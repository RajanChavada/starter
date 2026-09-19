"""Flash-decoding attention for the T=1 step, in Triton.

Why this exists. The SDPA decode path has two problems at these shapes. It
needs a validity mask, because a fixed-capacity cache holds stale slots past
the write head, and a mask rules out the flash backend. And with a 4-position
query over 8 KV heads, batch 1 gives the memory-efficient kernel 8 units of
work to spread over 132 SMs.

This kernel splits the key range instead, so parallelism is B * H * SPLITS and
does not collapse at batch 1, and it takes the valid length from a device
tensor, so it reads only live cache and needs no mask. Being device-side, the
length can change between CUDA graph replays without recapture.

Arithmetic follows FlashAttention as the reference backends implement it:
bf16 inputs into fp32 tensor-core accumulators, online softmax in fp32,
probabilities rounded to bf16 before the PV product, one final divide. That is
a reordering of the reference reduction, not a reformulation of it.
"""

import torch
import triton
import triton.language as tl

#: tl.dot needs at least 16 rows. Qwen3 has 4 query heads per KV head, so the
#: query block is padded and the extra rows are masked off on store.
BLOCK_M = 16


@triton.jit
def _split_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    len_ptr,
    acc_ptr,
    max_ptr,
    sum_ptr,
    out_ptr,
    scale,
    CAP: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLITS: tl.constexpr,
    SINGLE: tl.constexpr,
):
    head = tl.program_id(0)
    split = tl.program_id(1)

    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < G

    query = tl.load(
        q_ptr + head * (G * D) + rows[:, None] * D + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )

    valid = tl.load(len_ptr)
    per_split = (CAP + SPLITS - 1) // SPLITS
    start = split * per_split
    stop = tl.minimum(start + per_split, valid)

    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    base = head * (CAP * D)
    # Dynamic bounds, deliberately: a statically sized loop would process
    # blocks with no live column, whose all -inf row makes the online softmax
    # rescale 0/0. Every block entered here has at least one live column.
    for block in range(start, stop, BLOCK_N):
        keys = block + cols
        live = keys < stop
        k = tl.load(
            k_ptr + base + keys[:, None] * D + dims[None, :],
            mask=live[:, None],
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(k)) * scale
        scores = tl.where(live[None, :], scores, float("-inf"))

        block_max = tl.maximum(running_max, tl.max(scores, 1))
        rescale = tl.exp(running_max - block_max)
        probs = tl.exp(scores - block_max[:, None])

        running_sum = running_sum * rescale + tl.sum(probs, 1)
        acc = acc * rescale[:, None]

        v = tl.load(
            v_ptr + base + keys[:, None] * D + dims[None, :],
            mask=live[:, None],
            other=0.0,
        )
        acc += tl.dot(probs.to(v.dtype), v)
        running_max = block_max

    if SINGLE:
        # One split covers every key, so there is nothing to combine and the
        # divide belongs here rather than in a second launch.
        tl.store(
            out_ptr + head * (G * D) + rows[:, None] * D + dims[None, :],
            (acc / running_sum[:, None]).to(out_ptr.dtype.element_ty),
            mask=row_mask[:, None],
        )
        return

    # Partials stay unnormalized; the combine pass holds the only divide.
    out_base = head * (SPLITS * BLOCK_M * D) + split * (BLOCK_M * D)
    tl.store(
        acc_ptr + out_base + rows[:, None] * D + dims[None, :],
        acc,
        mask=row_mask[:, None],
    )
    stat_base = head * (SPLITS * BLOCK_M) + split * BLOCK_M
    tl.store(max_ptr + stat_base + rows, running_max, mask=row_mask)
    tl.store(sum_ptr + stat_base + rows, running_sum, mask=row_mask)


@triton.jit
def _combine_splits(
    acc_ptr,
    max_ptr,
    sum_ptr,
    out_ptr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLITS: tl.constexpr,
):
    head = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    row_mask = rows < G

    overall = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    for split in range(SPLITS):
        stat = head * (SPLITS * BLOCK_M) + split * BLOCK_M
        overall = tl.maximum(overall, tl.load(max_ptr + stat + rows, mask=row_mask))

    numerator = tl.zeros((BLOCK_M, D), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    for split in range(SPLITS):
        stat = head * (SPLITS * BLOCK_M) + split * BLOCK_M
        split_max = tl.load(max_ptr + stat + rows, mask=row_mask, other=float("-inf"))
        split_sum = tl.load(sum_ptr + stat + rows, mask=row_mask, other=0.0)
        # A split past the valid length contributes exp(-inf - finite) = 0.
        weight = tl.exp(split_max - overall)
        chunk = tl.load(
            acc_ptr
            + head * (SPLITS * BLOCK_M * D)
            + split * (BLOCK_M * D)
            + rows[:, None] * D
            + dims[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        numerator += chunk * weight[:, None]
        denominator += split_sum * weight

    result = numerator / denominator[:, None]
    tl.store(
        out_ptr + head * (G * D) + rows[:, None] * D + dims[None, :],
        result.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )


def choose_splits(heads: int, capacity: int, block_n: int) -> int:
    """Enough splits to fill the device, never so many that blocks go empty.

    Fixed at allocation time and passed as a constexpr, so exactly one
    specialization is compiled and no measured step can trigger a new one.
    """
    by_occupancy = max(1, 512 // max(heads, 1))
    by_capacity = max(1, capacity // block_n)
    return max(1, min(by_occupancy, by_capacity, 64))


def flash_decode(query, keys, values, length, out, acc_buf, max_buf, sum_buf,
                 scale: float, splits: int, block_n: int) -> None:
    """Allocation-free, so the whole call is safe to capture in a graph.

    query/out are [B, H, G, D]; keys/values are [B, H, CAP, D]; length is a
    one-element device tensor holding the count of live cache slots. All
    tensors must be contiguous and on the same device.
    """
    batch, heads, groups, head_dim = query.shape
    capacity = keys.shape[2]
    programs = batch * heads

    _split_attention[(programs, splits)](
        query, keys, values, length, acc_buf, max_buf, sum_buf, out, scale,
        CAP=capacity, G=groups, D=head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=block_n, SPLITS=splits, SINGLE=splits == 1,
        num_warps=4, num_stages=2,
    )
    if splits == 1:
        return
    _combine_splits[(programs,)](
        acc_buf, max_buf, sum_buf, out,
        G=groups, D=head_dim, BLOCK_M=BLOCK_M, SPLITS=splits,
        num_warps=4, num_stages=2,
    )
