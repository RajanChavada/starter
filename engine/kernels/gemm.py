"""Skinny GEMM for the decode step's projections.

Decode multiplies one row (or a handful) against every weight matrix in the
model, so each projection is bound by streaming its weight out of HBM, not by
arithmetic. cuBLAS reaches roughly a third of peak bandwidth on that shape —
its tiling is built for square problems — and at 8 GB of weights per token
that ceiling is the single largest cost in the step.

This kernel tiles only the output dimension and streams K contiguously, which
is the layout torch keeps weights in ([N, K] row major, so W[n, :] is
contiguous). Accumulation is fp32 from bf16 inputs and the result is rounded
once on store, which is what cuBLAS does; reassociating the K reduction is a
reordering, not a reformulation.

Nothing here is trusted blind. The engine benchmarks every configuration
against ``F.linear`` during warmup and keeps cuBLAS unless a configuration is
both correct and faster, so adopting this can only help.
"""

import torch
import triton
import triton.language as tl

#: tl.dot's minimum tile. The row block must also cover the whole batch, or
#: rows past it are silently never computed, so it is derived per launch.
MIN_BLOCK_M = 16


def block_m_for(rows: int) -> int:
    return max(MIN_BLOCK_M, triton.next_power_of_2(rows))

#: (BLOCK_N, BLOCK_K, num_warps). N and K are runtime arguments, so each tuple
#: compiles once and is reused for every projection shape.
CONFIGS = (
    (32, 256, 4),
    (64, 128, 4),
    (64, 256, 8),
    (128, 128, 8),
    (128, 64, 4),
    (256, 64, 8),
)


@triton.jit
def _skinny_gemm(
    x_ptr,
    w_ptr,
    o_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    n_live = offs_n < N
    m_live = offs_m < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_live = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=m_live[:, None] & k_live[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=n_live[:, None] & k_live[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))

    tl.store(
        o_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=m_live[:, None] & n_live[None, :],
    )


def run(x, weight, out, config) -> None:
    """x is [M, K], weight is [N, K], out is [M, N]; all contiguous bf16."""
    block_n, block_k, warps = config
    rows, k = x.shape
    n = weight.shape[0]
    _skinny_gemm[(triton.cdiv(n, block_n),)](
        x, weight, out, rows, n, k,
        BLOCK_M=block_m_for(rows), BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=warps, num_stages=3,
    )
