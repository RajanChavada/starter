"""Fused final RMSNorm, LM-head projection, and greedy token selection.

The regular decode tail writes a full vocabulary row, then launches argmax to
read it back.  Greedy generation only needs its largest element.  This kernel
keeps each vocabulary tile's maximum in a small temporary buffer, followed by
a reduction over those maxima.  The projection arithmetic deliberately mirrors
the decode GEMM: fp32 accumulation, a bf16 product store point, and Qwen's
bf16 RMSNorm cast placement.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _lm_head_tiles(
    x_ptr,
    weight_ptr,
    gain_ptr,
    values_ptr,
    indices_ptr,
    M,
    N,
    K,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    n_live = offs_n < N
    m_live = offs_m < M

    squares = tl.zeros((BLOCK_M,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_live = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=m_live[:, None] & k_live[None, :],
            other=0.0,
        ).to(tl.float32)
        squares += tl.sum(x * x, axis=1)
    inv = tl.math.rsqrt(squares / K + eps)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_live = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=m_live[:, None] & k_live[None, :],
            other=0.0,
        )
        normed = (x.to(tl.float32) * inv[:, None]).to(tl.bfloat16)
        gain = tl.load(gain_ptr + offs_k, mask=k_live, other=0.0)
        x = (normed.to(tl.float32) * gain[None, :].to(tl.float32)).to(tl.bfloat16)
        weight = tl.load(
            weight_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=n_live[:, None] & k_live[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(weight))

    # This bf16 round is the value seen by torch.argmax in the existing path.
    logits = acc.to(tl.bfloat16).to(tl.float32)
    logits = tl.where(n_live[None, :], logits, float("-inf"))
    local_values = tl.max(logits, axis=1)
    local_indices = tl.argmax(logits, axis=1).to(tl.int32) + tile * BLOCK_N
    tiles = tl.cdiv(N, BLOCK_N)
    tl.store(
        values_ptr + offs_m * tiles + tile,
        local_values,
        mask=m_live,
    )
    tl.store(
        indices_ptr + offs_m * tiles + tile,
        local_indices,
        mask=m_live,
    )


@triton.jit
def _reduce_argmax(
    values_ptr,
    indices_ptr,
    output_ptr,
    TILES,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_TILES)
    live = offsets < TILES
    values = tl.load(
        values_ptr + row * TILES + offsets,
        mask=live,
        other=float("-inf"),
    )
    indices = tl.load(indices_ptr + row * TILES + offsets, mask=live, other=0)
    best = tl.max(values, axis=0)
    # Preserve torch.argmax's first-index tie break, though exact ties are
    # exceptionally rare for this checkpoint.
    candidates = tl.where(values == best, indices, 2147483647)
    output = tl.min(candidates, axis=0)
    tl.store(output_ptr + row, output.to(tl.int64))


def run(x, weight, gain, eps, values, indices, output, config) -> None:
    """Write the greedy LM-head token for each [M, K] input row."""
    rows, k = x.shape
    n = weight.shape[0]
    block_n, block_k, warps, stages = config
    tiles = triton.cdiv(n, block_n)
    _lm_head_tiles[(tiles,)](
        x,
        weight,
        gain,
        values,
        indices,
        rows,
        n,
        k,
        eps,
        BLOCK_M=max(16, triton.next_power_of_2(rows)),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    _reduce_argmax[(rows,)](
        values,
        indices,
        output.reshape(-1),
        tiles,
        BLOCK_TILES=triton.next_power_of_2(tiles),
        num_warps=4,
        num_stages=2,
    )
