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

#: (BLOCK_N, BLOCK_K, num_warps, num_stages). N and K are runtime arguments,
#: so each tuple compiles once and is reused for every projection shape --
#: a wide search costs compile time once, not per projection.
#:
#: The narrow-output projections (o_proj and down_proj both emit 2560) need
#: small BLOCK_N to get enough programs to fill the device, while the wide
#: ones (gate/up at 19456, the LM head at 151936) want large tiles and deep
#: pipelining. One list covers both; warmup picks per shape.
CONFIGS = (
    (32, 256, 4, 3),
    (64, 128, 4, 4),
    (64, 256, 8, 3),
    (128, 128, 8, 3),
    (128, 64, 4, 4),
    (256, 64, 8, 3),
)

#: (BLOCK_N, BLOCK_K, num_warps, num_stages, SPLIT_K), offered only for the
#: narrow-output projections. o_proj and down_proj both emit 2560, so tiling
#: the output alone yields ~80 programs for 132 SMs -- the device is more than
#: a third idle while streaming a third of each layer's weights. Splitting the
#: reduction instead multiplies the program count by SPLIT_K.
#:
#: The partial sums stay in fp32 and are reduced in a second pass rather than
#: accumulated with atomics, so the result does not depend on the order blocks
#: happen to finish in. It is the same fp32 reduction the single-pass kernel
#: does, reassociated.
SPLIT_CONFIGS = (
    (16, 128, 4, 3, 4),
    (16, 256, 4, 3, 8),
    (32, 64, 4, 4, 8),
    (32, 128, 4, 3, 4),
    (64, 64, 4, 3, 4),
)

#: Above this output width there are already enough programs and the partial
#: buffer would be large for no benefit.
SPLIT_MAX_N = 4096


@triton.jit
def _split_gemm(
    x_ptr,
    w_ptr,
    part_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    n_live = offs_n < N
    m_live = offs_m < M

    per_split = tl.cdiv(K, SPLIT_K)
    start = pid_k * per_split
    stop = tl.minimum(start + per_split, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(start, stop, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_live = offs_k < stop
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
        part_ptr + pid_k * (M * N) + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=m_live[:, None] & n_live[None, :],
    )


@triton.jit
def _split_gemm_fixup(
    x_ptr,
    w_ptr,
    r_ptr,
    part_ptr,
    lock_ptr,
    o_ptr,
    M,
    N,
    K,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Split-K with the reduction folded into the last program to finish.

    Every program stores its fp32 partial, then increments the tile's counter
    with release semantics. The program that observes the final count reads
    all SPLIT_K partials back, in split order, and runs the epilogue, so the
    sum is the same in-order fp32 sum ``_reduce_splits`` performs and the
    second launch disappears. The counter is reset for the next replay.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    n_live = offs_n < N
    m_live = offs_m < M
    tile_mask = m_live[:, None] & n_live[None, :]
    tile_offs = offs_m[:, None] * N + offs_n[None, :]

    per_split = tl.cdiv(K, SPLIT_K)
    start = pid_k * per_split
    stop = tl.minimum(start + per_split, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(start, stop, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_live = offs_k < stop
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

    tl.store(part_ptr + pid_k * (M * N) + tile_offs, acc, mask=tile_mask)
    tl.debug_barrier()
    arrived = tl.atomic_add(lock_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    tl.debug_barrier()

    if arrived == SPLIT_K - 1:
        total = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for split in range(SPLIT_K):
            total += tl.load(
                part_ptr + split * (M * N) + tile_offs,
                mask=tile_mask,
                other=0.0,
                volatile=True,
            )
        result = total.to(o_ptr.dtype.element_ty)
        if HAS_RESIDUAL:
            residual = tl.load(r_ptr + tile_offs, mask=tile_mask, other=0.0)
            result = (result.to(tl.float32) + residual.to(tl.float32)).to(
                o_ptr.dtype.element_ty
            )
        tl.store(o_ptr + tile_offs, result, mask=tile_mask)
        tl.debug_barrier()
        tl.atomic_xchg(lock_ptr + pid_n, 0, sem="release", scope="gpu")


#: Set False to fall back to the separate reduction launch.
SPLIT_FIXUP = True

_locks = {}


def _lock_buffer(device) -> torch.Tensor:
    """Per-tile arrival counters, zero at rest; each launch leaves them zero."""
    key = str(device)
    locks = _locks.get(key)
    if locks is None:
        locks = torch.zeros(4096, dtype=torch.int32, device=device)
        _locks[key] = locks
    return locks


@triton.jit
def _reduce_splits(
    part_ptr,
    r_ptr,
    o_ptr,
    TOTAL,
    HAS_RESIDUAL: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offs < TOTAL
    total = tl.zeros((BLOCK,), tl.float32)
    for split in range(SPLIT_K):
        total += tl.load(part_ptr + split * TOTAL + offs, mask=live, other=0.0)

    result = total.to(o_ptr.dtype.element_ty)
    if HAS_RESIDUAL:
        residual = tl.load(r_ptr + offs, mask=live, other=0.0)
        result = (result.to(tl.float32) + residual.to(tl.float32)).to(
            o_ptr.dtype.element_ty
        )
    tl.store(o_ptr + offs, result, mask=live)


@triton.jit
def _skinny_gemm(
    x_ptr,
    w_ptr,
    r_ptr,
    g_ptr,
    o_ptr,
    M,
    N,
    K,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    n_live = offs_n < N
    m_live = offs_m < M

    inv = tl.zeros((BLOCK_M,), tl.float32)
    if NORMALIZE:
        # The RMS reduction needs the whole row before any of it can be
        # scaled, so the input is streamed twice. It is a few kilobytes
        # against tens of megabytes of weights, and it saves a launch and a
        # round trip through HBM.
        squares = tl.zeros((BLOCK_M,), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_live = offs_k < K
            chunk = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=m_live[:, None] & k_live[None, :],
                other=0.0,
            ).to(tl.float32)
            squares += tl.sum(chunk * chunk, axis=1)
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
        if NORMALIZE:
            # Round the normalized value to bf16 before the gain multiply,
            # which is where Qwen3RMSNorm puts its cast.
            normed = (x.to(tl.float32) * inv[:, None]).to(tl.bfloat16)
            gain = tl.load(g_ptr + offs_k, mask=k_live, other=0.0)
            x = (normed.to(tl.float32) * gain[None, :].to(tl.float32)).to(tl.bfloat16)
        w = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=n_live[:, None] & k_live[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))

    result = acc.to(o_ptr.dtype.element_ty)
    if HAS_RESIDUAL:
        # torch rounds the matmul to bf16 before the residual add, then rounds
        # the sum; rounding once here would compute a different function.
        residual = tl.load(
            r_ptr + offs_m[:, None] * N + offs_n[None, :],
            mask=m_live[:, None] & n_live[None, :],
            other=0.0,
        )
        result = (result.to(tl.float32) + residual.to(tl.float32)).to(
            o_ptr.dtype.element_ty
        )

    tl.store(
        o_ptr + offs_m[:, None] * N + offs_n[None, :],
        result,
        mask=m_live[:, None] & n_live[None, :],
    )


@triton.jit
def _gateup_swiglu(
    x_ptr,
    w_ptr,
    g_ptr,
    o_ptr,
    M,
    N,
    K,
    eps,
    NORMALIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """gate/up projection with silu(gate) * up applied in the epilogue.

    ``w_ptr`` is the concatenated [2N, K] gate/up weight; a program owns the
    same BLOCK_N columns of both halves, so the activation never round-trips
    through HBM and the separate SwiGLU launch disappears. Each intermediate
    is rounded to bf16 exactly where the unfused path rounds it.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    n_live = offs_n < N
    m_live = offs_m < M

    inv = tl.zeros((BLOCK_M,), tl.float32)
    if NORMALIZE:
        squares = tl.zeros((BLOCK_M,), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_live = offs_k < K
            chunk = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=m_live[:, None] & k_live[None, :],
                other=0.0,
            ).to(tl.float32)
            squares += tl.sum(chunk * chunk, axis=1)
        inv = tl.math.rsqrt(squares / K + eps)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_live = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=m_live[:, None] & k_live[None, :],
            other=0.0,
        )
        if NORMALIZE:
            normed = (x.to(tl.float32) * inv[:, None]).to(tl.bfloat16)
            gain = tl.load(g_ptr + offs_k, mask=k_live, other=0.0)
            x = (normed.to(tl.float32) * gain[None, :].to(tl.float32)).to(tl.bfloat16)
        tile_mask = n_live[:, None] & k_live[None, :]
        w_gate = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=tile_mask, other=0.0
        )
        w_up = tl.load(
            w_ptr + (offs_n[:, None] + N) * K + offs_k[None, :],
            mask=tile_mask,
            other=0.0,
        )
        acc_gate += tl.dot(x, tl.trans(w_gate))
        acc_up += tl.dot(x, tl.trans(w_up))

    gate = acc_gate.to(tl.bfloat16).to(tl.float32)
    up = acc_up.to(tl.bfloat16).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(
        o_ptr + offs_m[:, None] * N + offs_n[None, :],
        (activated * up).to(o_ptr.dtype.element_ty),
        mask=m_live[:, None] & n_live[None, :],
    )


#: (BLOCK_N, BLOCK_K, num_warps, num_stages) for the fused gate/up + SwiGLU
#: kernel. A program streams two BLOCK_N-row slabs, so BLOCK_N=32 gives the
#: same 304 programs as the pinned 64-wide plain projection.
GATEUP_SWIGLU_CONFIGS = (
    (32, 128, 4, 4),
    (32, 256, 4, 3),
    (16, 256, 4, 3),
    (64, 128, 4, 4),
    (64, 64, 4, 4),
)


def run_gateup_swiglu(x, weight, out, config, gain=None, eps=0.0) -> None:
    """x is [M, K], weight is the [2N, K] gate/up concat, out is [M, N]."""
    rows, k = x.shape
    n = weight.shape[0] // 2
    block_n, block_k, warps, stages = config
    _gateup_swiglu[(triton.cdiv(n, block_n),)](
        x,
        weight,
        gain if gain is not None else x,
        out,
        rows, n, k, eps,
        NORMALIZE=gain is not None,
        BLOCK_M=block_m_for(rows), BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=warps, num_stages=stages,
    )


def run(x, weight, out, config, residual=None, gain=None, eps=0.0) -> None:
    """x is [M, K], weight is [N, K], out is [M, N]; all contiguous bf16.

    ``residual``, if given, is [M, N] and is added in the epilogue, folding
    what would otherwise be a separate launch per residual branch per layer.
    ``gain``, if given, is the [K] RMSNorm weight applied to ``x`` before the
    product, folding the pre-matmul norm in the same way.
    """
    rows, k = x.shape
    n = weight.shape[0]

    if len(config) == 5:
        block_n, block_k, warps, stages, split_k = config
        partials = torch.empty(
            (split_k, rows, n), dtype=torch.float32, device=x.device
        )
        if SPLIT_FIXUP:
            _split_gemm_fixup[(triton.cdiv(n, block_n), split_k)](
                x, weight, residual if residual is not None else out,
                partials, _lock_buffer(x.device), out, rows, n, k,
                HAS_RESIDUAL=residual is not None,
                BLOCK_M=block_m_for(rows), BLOCK_N=block_n,
                BLOCK_K=block_k, SPLIT_K=split_k,
                num_warps=warps, num_stages=stages,
            )
            return
        _split_gemm[(triton.cdiv(n, block_n), split_k)](
            x, weight, partials, rows, n, k,
            BLOCK_M=block_m_for(rows), BLOCK_N=block_n,
            BLOCK_K=block_k, SPLIT_K=split_k,
            num_warps=warps, num_stages=stages,
        )
        total = rows * n
        _reduce_splits[(triton.cdiv(total, 1024),)](
            partials, residual if residual is not None else out, out, total,
            HAS_RESIDUAL=residual is not None, SPLIT_K=split_k, BLOCK=1024,
            num_warps=4, num_stages=2,
        )
        return

    block_n, block_k, warps, stages = config
    _skinny_gemm[(triton.cdiv(n, block_n),)](
        x,
        weight,
        residual if residual is not None else x,
        gain if gain is not None else x,
        out,
        rows, n, k, eps,
        HAS_RESIDUAL=residual is not None,
        NORMALIZE=gain is not None,
        BLOCK_M=block_m_for(rows), BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=warps, num_stages=stages,
    )
