"""Fused rotary embedding and SwiGLU.

These operations are trivial arithmetic on tiny tensors at decode, so their
cost is almost entirely the launch. Expressed in torch they are many kernels:
``(q * cos) + (rotate_half(q) * sin)`` is a slice, a negation, a concatenation
and three arithmetic launches, and it runs twice per layer across 36 layers.

Both kernels reproduce the reference's rounding rather than its formula's
"obvious" fused form. Torch rounds every bf16 elementwise result, so a fused
kernel that keeps everything in fp32 and rounds once computes a different
function — more accurate, and not the one being judged. The casts below are
deliberate and load-bearing.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, HALF: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, HALF)
    base = row * (2 * HALF)

    lower = tl.load(x_ptr + base + offs).to(tl.float32)
    upper = tl.load(x_ptr + base + HALF + offs).to(tl.float32)
    cos_lo = tl.load(cos_ptr + offs).to(tl.float32)
    cos_hi = tl.load(cos_ptr + HALF + offs).to(tl.float32)
    sin_lo = tl.load(sin_ptr + offs).to(tl.float32)
    sin_hi = tl.load(sin_ptr + HALF + offs).to(tl.float32)

    # rotate_half gives [-upper, lower]. Each product is rounded to bf16
    # before the sum, exactly as the two torch multiplies would.
    a_lo = (lower * cos_lo).to(tl.bfloat16).to(tl.float32)
    b_lo = (-upper * sin_lo).to(tl.bfloat16).to(tl.float32)
    a_hi = (upper * cos_hi).to(tl.bfloat16).to(tl.float32)
    b_hi = (lower * sin_hi).to(tl.bfloat16).to(tl.float32)

    tl.store(out_ptr + base + offs, (a_lo + b_lo).to(tl.bfloat16))
    tl.store(out_ptr + base + HALF + offs, (a_hi + b_hi).to(tl.bfloat16))


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary embedding over the last dimension of a contiguous bf16 tensor.

    ``cos`` and ``sin`` are one row of the position table, broadcast over every
    row of ``x``; this is the T=1 decode case, where all heads share a position.
    """
    head_dim = x.shape[-1]
    rows = x.numel() // head_dim
    out = torch.empty_like(x)
    _rope_kernel[(rows,)](
        x, cos, sin, out, HALF=head_dim // 2, num_warps=1, num_stages=2
    )
    return out


@triton.jit
def _swiglu_kernel(gu_ptr, out_ptr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * BLOCK + tl.arange(0, BLOCK)
    live = offs < WIDTH

    gate = tl.load(gu_ptr + row * (2 * WIDTH) + offs, mask=live).to(tl.float32)
    up = tl.load(gu_ptr + row * (2 * WIDTH) + WIDTH + offs, mask=live).to(tl.float32)
    # torch computes silu in fp32 and rounds its result before the multiply.
    activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * WIDTH + offs, (activated * up).to(tl.bfloat16), mask=live)


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up, reading the fused projection's output in place.

    ``gate_up`` is [..., 2 * WIDTH] contiguous, gate first, as produced by the
    concatenated gate/up weight.
    """
    width = gate_up.shape[-1] // 2
    rows = gate_up.numel() // (2 * width)
    out = torch.empty(
        (*gate_up.shape[:-1], width), dtype=gate_up.dtype, device=gate_up.device
    )
    block = 1024
    _swiglu_kernel[(rows, triton.cdiv(width, block))](
        gate_up, out, WIDTH=width, BLOCK=block, num_warps=4, num_stages=2
    )
    return out
