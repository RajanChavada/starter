"""One kernel for everything between the QKV projection and attention.

At decode the six operations after the fused QKV matmul — per-head RMSNorm on
Q and on K, rotary on each, and the K and V cache writes — all touch a few
kilobytes and are pure launch cost. This does them in a single launch, one
program per (sequence, head), with each program owning one 128-wide head.

Rounding follows the reference exactly. RMSNorm reduces in fp32, rounds the
normalized value to bf16, and only then multiplies by the learned gain; rotary
rounds each product before summing. Writing it the tidy way, carrying fp32
through and rounding once at the end, is a different function.

The cache write lands at a position read from a device tensor, so the step
stays replayable from a CUDA graph without recapture.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope(src, gain, cos_row, sin_row, eps, HALF: tl.constexpr, D: tl.constexpr):
    """RMSNorm then rotary over one head, returned as its two halves."""
    lo = tl.arange(0, HALF)
    hi = HALF + lo

    x_lo = tl.load(src + lo).to(tl.float32)
    x_hi = tl.load(src + hi).to(tl.float32)
    variance = (tl.sum(x_lo * x_lo) + tl.sum(x_hi * x_hi)) / D
    inv = tl.math.rsqrt(variance + eps)

    # Round before the gain multiply, as Qwen3RMSNorm does.
    n_lo = (x_lo * inv).to(tl.bfloat16)
    n_hi = (x_hi * inv).to(tl.bfloat16)
    g_lo = tl.load(gain + lo).to(tl.float32)
    g_hi = tl.load(gain + hi).to(tl.float32)
    q_lo = (n_lo.to(tl.float32) * g_lo).to(tl.bfloat16).to(tl.float32)
    q_hi = (n_hi.to(tl.float32) * g_hi).to(tl.bfloat16).to(tl.float32)

    cos_lo = tl.load(cos_row + lo).to(tl.float32)
    cos_hi = tl.load(cos_row + hi).to(tl.float32)
    sin_lo = tl.load(sin_row + lo).to(tl.float32)
    sin_hi = tl.load(sin_row + hi).to(tl.float32)

    # rotate_half gives [-upper, lower]; each product rounds before the sum.
    a_lo = (q_lo * cos_lo).to(tl.bfloat16).to(tl.float32)
    b_lo = (-q_hi * sin_lo).to(tl.bfloat16).to(tl.float32)
    a_hi = (q_hi * cos_hi).to(tl.bfloat16).to(tl.float32)
    b_hi = (q_lo * sin_hi).to(tl.bfloat16).to(tl.float32)
    return (a_lo + b_lo).to(tl.bfloat16), (a_hi + b_hi).to(tl.bfloat16)


@triton.jit
def _qkv_finish(
    qkv_ptr,
    q_gain_ptr,
    k_gain_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    q_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    eps,
    WIDTH: tl.constexpr,
    N_Q: tl.constexpr,
    N_KV: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    CAP: tl.constexpr,
):
    seq = tl.program_id(0)
    slot = tl.program_id(1)
    lo = tl.arange(0, HALF)
    hi = HALF + lo
    row = seq * WIDTH
    position = tl.load(pos_ptr)

    if slot < N_Q:
        src = qkv_ptr + row + slot * D
        cos_row = cos_ptr + position * D
        sin_row = sin_ptr + position * D
        out_lo, out_hi = _norm_rope(
            src, q_gain_ptr, cos_row, sin_row, eps, HALF, D
        )
        dst = q_out_ptr + seq * (N_Q * D) + slot * D
        tl.store(dst + lo, out_lo)
        tl.store(dst + hi, out_hi)
    elif slot < N_Q + N_KV:
        head = slot - N_Q
        src = qkv_ptr + row + N_Q * D + head * D
        cos_row = cos_ptr + position * D
        sin_row = sin_ptr + position * D
        out_lo, out_hi = _norm_rope(
            src, k_gain_ptr, cos_row, sin_row, eps, HALF, D
        )
        dst = k_cache_ptr + seq * (N_KV * CAP * D) + head * (CAP * D) + position * D
        tl.store(dst + lo, out_lo)
        tl.store(dst + hi, out_hi)
    else:
        head = slot - N_Q - N_KV
        src = qkv_ptr + row + (N_Q + N_KV) * D + head * D
        dst = v_cache_ptr + seq * (N_KV * CAP * D) + head * (CAP * D) + position * D
        tl.store(dst + lo, tl.load(src + lo))
        tl.store(dst + hi, tl.load(src + hi))


def qkv_finish(
    qkv, q_gain, k_gain, cos_table, sin_table, position, q_out, k_cache, v_cache, eps
) -> None:
    """Normalize, rotate and cache one decode step's projections.

    ``qkv`` is [B, N_Q*D + 2*N_KV*D] contiguous; ``q_out`` is [B, N_KV, G, D],
    which is the same byte layout as [B, N_Q, D] and is what the decode
    attention kernel expects. ``position`` is a one-element device tensor.
    """
    batch = qkv.shape[0]
    n_kv, capacity, head_dim = k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]
    n_q = (qkv.shape[-1] - 2 * n_kv * head_dim) // head_dim
    _qkv_finish[(batch, n_q + 2 * n_kv)](
        qkv, q_gain, k_gain, cos_table, sin_table, position,
        q_out, k_cache, v_cache, eps,
        WIDTH=qkv.shape[-1], N_Q=n_q, N_KV=n_kv, D=head_dim,
        HALF=head_dim // 2, CAP=capacity,
        num_warps=2, num_stages=2,
    )
