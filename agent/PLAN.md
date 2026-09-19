# Optimization plan — Qwen3 4B decode

Not submitted. Notes and staging for `engine/engine.py`.

## Live spec vs. repo docs (API is authoritative)

`GET /api/v1/challenges` on the running server, challenge version 2,
specDigest `b103e41b…`, disagrees with the checked-in markdown:

| | Repo docs say | Live API says |
| --- | --- | --- |
| Workloads | 6 (3 public + 3 hidden) | **9 (3 public + 6 private)** |
| Scored | geomean of 3 hidden | **geomean of 6 private** |
| Compile budget | not mentioned | `max_compile_seconds: 600` |
| GPU / run budget | not mentioned | `max_gpu_seconds: 2100`, `max_run_seconds: 2400` |

Trust the API. Six private shapes, not three, and their shapes are unpublished.
The public shapes carry a `regime` tag (1, 2, 3) and all share `family: 1` with
weight 1.0, which suggests the private set spans the same or wider regimes.

**Design consequence: nothing may be tuned to batch 1/4/16 or prompt 512/2048.**
Every shape-dependent decision has to be made from the shapes observed at
warmup, not hardcoded. This is free if we build buffers and capture graphs in
the warmup call, which is what the plan already does.

## There is no training

Weights are frozen at revision `cdbee75f…`; `outputValidation` is
`greedy_exact_v1` with `tieMarginLogits: 2.0`. Fine-tuning, quantizing, or
distilling fails by construction. This is inference engineering only.

## The dev loop runs on their H100

Public runs are the test harness. They execute on the official hardware, take
about two minutes, never touch the leaderboard, and — critically — they still
run the teacher-forced correctness replay. They differ from official runs only
in that they use one sample instead of five and *report* the TTFT/TPOT ratios
instead of enforcing them.

So a public run answers both questions we care about: is it correct, and is it
faster. No local or rented GPU is needed. Log diagnostics to stdout and read
them from the bounded log tail via `./bin/dryft logs <run-id> --follow`.

What a public run cannot tell us, because it runs one sample: the 25% spread
gate. That risk stays invisible until an official run.

## Where the time goes

Model facts that drive everything below:

| Quantity | Value |
| --- | ---: |
| Non-embedding params | 3.63 B |
| Embedding / LM head (tied) | 0.39 B |
| Total BF16 weight bytes | ~8.05 GB |
| H100 HBM3 bandwidth | ~3.3 TB/s |
| **Memory-bound floor per decode step** | **~2.4 ms** |

A decode step reads every weight once no matter the batch size, so the floor is
batch-independent. Prefill is compute-bound instead: `2 * 3.63e9 * tokens` FLOP.

### Baseline model vs. observed leaderboard

Assuming ~18-20 ms/step for Transformers dispatch at these sizes:

| Workload | B | prompt | out | prefill | decode | est. tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| public-0 | 1 | 512 | 32 | ~10 ms | 32 x 18 ms | ~55 |
| public-1 | 4 | 2048 | 32 | ~120 ms | 32 x 19 ms | ~176 |
| public-2 | 16 | 512 | 128 | ~120 ms | 128 x 20 ms | ~764 |

Geometric mean ~195 tok/s. The bottom of the leaderboard sits at 198.5, so this
is the baseline and the estimate is calibrated. Current leader: 541.6 tok/s.

### Target, and the wall past it

Same shapes with the decode step at the ~2.4 ms bandwidth floor:

| Workload | est. tok/s |
| --- | ---: |
| public-0 | ~368 |
| public-1 | ~640 |
| public-2 | ~4795 |

Geometric mean **~1041 tok/s**. Removing per-step overhead is worth ~5.3x and
gets us to roughly the leaderboard's current top (two teams >1000 tok/s as of
Sep 19). That is not a coincidence: **the leaders are at the hardware floor.**

This reframes everything. Overhead removal is table stakes, not an edge. No
amount of further dispatch tuning beats a bandwidth floor — at 8.05 GB of
weight reads per forward, 2.4 ms per token is physics.

The only way past the floor is to produce more than one token per weight read:

- **Speculative decoding with exact verification.** Verify k draft tokens in a
  single forward and the same 8 GB read yields up to k tokens. At public-0,
  decode is 77 of 87 ms; a 2.5x acceptance rate takes the workload from ~368 to
  roughly ~780 tok/s. This is the only lever with headroom left.
- **Prefill**, which is compute-bound and therefore not subject to this floor.
  It is ~60% of public-1's total time at the floor, so it stops being a
  rounding error once decode is fast.

Batch size does not help us here — the floor is batch-independent, which is why
public-2 already scores 13x public-0. Low-batch private workloads are where the
score is won or lost.

## Reading the gates correctly

TTFT and TPOT at 1.10x native are *lower* bounds on speed — they fail you for
being slower than native, not faster. They are regression guards. The only
plausible way to trip them is damaging prefill, since TTFT is essentially
prefill + one step.

The gate that actually threatens a good engine is **spread < 25% across five
samples**. Anything with a data-dependent slow path — lazy Triton compilation,
graph recapture, a cache that occasionally reallocates — passes a one-sample
public run and fails the official five. Everything shape-dependent must be
built during warmup.

Memory is a non-issue: 8.05 GB weights plus at most ~1.5 GB of KV cache
(public-2: `147456 * 16 * 640` bytes) against a 72 GB ceiling.

## Stages

Each stage is independently submittable and independently revertible. Measure
after each; do not stack unmeasured changes.

### Stage 1 — Bypass the wrapper (small, safe, unblocks the rest)

Replace `self.model(...)` with a direct `qwen_forward` over `base.layers`,
per the guide. Kills `Qwen3ForCausalLM` output-object construction and generic
dispatch. Expect low single-digit percent. Its real value is that it puts the
forward path under our control so Stages 2-3 are possible.

Risk: low. Arithmetic is unchanged — same modules, same order.

### Stage 2 — Static preallocated KV cache

Per-layer K/V as `[B, 8, prompt_len + max_new_tokens, 128]`, allocated during
warmup when shapes are known. Write at `[:, :, L:L+T, :]`. Removes per-step
allocation and the `DynamicCache` concatenation.

Must reset logical length to zero at the top of every `generate` call — warmup
and measured prompts must never share cached content.

Critical detail from the guide: `attention_mask=None` is only valid for a full
unpadded prefill into an empty cache. With a fixed-capacity cache we must supply
an explicit mask permitting key `j` only when `j <= L + query_index` **and** the
slot is initialized. Getting this wrong reads uninitialized memory and fails
correctness non-deterministically — the worst possible failure mode.

Risk: medium. This is where a subtle masking bug hides.

### Stage 3 — CUDA graph capture of the decode step (the big one)

Capture one graph for the `T=1` decode step. Requirements:

- Fixed input/output addresses; update token IDs and positions **in place**.
- `.tolist()` and `yield` stay outside the capture.
- Attention shape must not vary. A growing `:L+T` slice changes shape every
  step, so either run over full capacity with a mask derived from a
  device-side step counter, or use a custom kernel that reads the length from
  a device tensor.

Full-capacity attention costs extra HBM traffic (~1.5 GB/step at public-2,
roughly 19% over the 8 GB floor). Acceptable for a first cut; Stage 5 removes
it.

Capture during warmup, never during a measured sample. Keep prefill on the
eager path — it runs once and has a different shape.

Expect the bulk of the 4.2x here.

Risk: medium-high, but failures are loud (wrong shapes, stale addresses)
rather than silent.

### Stage 4 — Fuse the projections

Concatenate `q_proj`/`k_proj`/`v_proj` into one `[6144, 2560]` weight and
`gate_proj`/`up_proj` into one `[19456, 2560]`, done once in `__init__`. Three
GEMMs become one, twice per layer, 36 layers. Under a CUDA graph the launch
saving is already gone, so this wins on GEMM efficiency only — larger, better
shaped matrices. Helps prefill more than decode.

This is a pure re-association of independent matmuls; it does not change any
individual output value. Numerically the safest change on the list.

### Stage 5 — Custom Triton decode attention

Flash-decoding style kernel taking a device-side valid length, so we stop
reading uninitialized cache capacity. Removes the ~19% Stage 3 overhead and
drops the mask entirely. Also the natural place to apply the `h // 4` query-to-
KV head mapping directly instead of materializing the 8 -> 32 head expansion
Transformers does before SDPA.

Launch every specialization during warmup — Triton compiles on first use, and
a compile inside a measured sample is an instant spread-gate failure.

### Stage 6 — Fused elementwise (low priority)

RMSNorm (144 launches/step), RoPE, SwiGLU. At batch 1 decode these are
negligible against 8 GB of weight traffic; they matter at public-2 prefill.
`engine/kernels/rmsnorm.py` is the worked example and already has the cast
placement right.

**The cast rule:** reduce and normalize in FP32, cast the normalized value to
BF16, *then* multiply by the learned weight. Keeping the product in FP32 and
rounding once is more accurate and is wrong — it computes a different function.
Reorder arithmetic freely; never reformulate it.

### Stage 7 — Speculative decoding with exact verification (only if needed)

Passes rule 3 by construction. Biggest remaining win at batch 1, where we are
furthest from saturating the GPU. Prompt-lookup / n-gram self-speculation
avoids shipping a draft model.

Deferred because acceptance rate is prompt-dependent, which is exactly the
shape of risk the 25% spread gate is built to catch. Do not attempt until
Stages 1-5 are banked and stable.

## Do not

Quantization, cache eviction, approximate or sparse attention, a draft model
without verification. All shift logits by whole units and fail the 2.0 tie
margin.

## torch.compile — considered, rejected for now

`mode="reduce-overhead"` would do much of Stages 1-3 automatically. Rejected as
the primary path because compilation of a 36-layer 4B model can consume most of
the 300 s load+warmup budget, and any recompilation triggered inside a measured
sample is a spread-gate failure. Manual graph capture is more predictable and
warms up faster. Worth a side experiment once the manual path is banked.

## Verification before every submission

Per the guide's "Check each change", compare against an untouched baseline
under inference mode:

1. Prefill logits at each public shape.
2. Several cached decode steps.
3. **Two consecutive `generate` calls with different prompts** — catches cache
   reset bugs, which are the most likely silent failure.
4. TTFT, TPOT, throughput and peak memory at all three public shapes.

Run these as assertions inside a public run and read the results from the log
tail. A public run executes the real teacher-forced replay on real hardware,
so it is the verification step — no local or rented GPU required.

## Log

| Stage | Submitted | public-0 | public-1 | public-2 | geomean | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| static KV + direct layers + graph | d5334ff / 558ab255 | 104.1 | 229.0 | 1381.6 | 416.4 | Banked starting point from parent loop. TPOT ratios 0.44/0.52/0.45x native. |
| flash decode + fused projections + fallbacks | 5d2189a / f587d356 | 116.1 | 268.2 | 1514.3 | 479.4 | New banked best. TPOT ratios 0.22/0.24/0.24x native. Correctness and latency gates passed. |
| skinny GEMM + chunked sync | c78533f / 8f04a7b3 | 125.2 | 273.2 | 1558.4 | 495.7 | New banked best. Candidate ms improved to 255.6/468.5/1314.2. |
| fused RMSNorm + RoPE + SwiGLU | f4b49b6 / 3578a1ff | 188.5 | 383.9 | 2399.7 | 743.5 | New banked best. Candidate ms 169.7/333.4/853.4, TTFT 0.65/0.74/0.73x native. |
| residual epilogue + 12-config autotune | 7b48455 | | | | 719.1 | noise band |
| norm+rotary+cache write in one kernel | 9fb9dbd | | | | 780.8 | +5% over f4b49b6 |
| attention autotune + RMSNorm into GEMM | 7c2a739 | | | | 772.5 | noise band |
| autotune narrowed to 6 configs | 3c5eb28 | | | | 796.5 | p0 tpot 4.47 |
| residual epilogue off | 9db3ed8 | | | | 762.3 | noise band |
| SwiGLU epilogue + sync 16 | 23496cb | | | | 747.0 | noise band |
| speculative decoding | 47335e6 | | | | 507.6 | **real regression**, p0 tpot 7.39. Abandoned. |
| verification via fused path | 5db42b5 | | | | 513.6 | stacked on the above |
| residual epilogue restored | 629e156 | | | | 499.8 | stacked on the above |
| revert to 3c5eb28 tree, byte identical | c9a7ac6 | | | | 766.3 | **p0 tpot 5.55 vs 4.47 for the same code** |
| split-K for narrow projections | 4f7c38f | 204.4 | 412.6 | 2483.6 | **799.7** | p0 tpot 4.51. Best. |
| reproducible autotune | 84c5d62 | | | | pending | |
| replay-timed warmup autotune | 59425ba | | | | pending | queued |
| aligned / evict-first GEMM loads | pending | | | | pending | submitting next |

## Commit -> score ladder (p0 tpot is the number to drive down)

| Commit | Change | Score | p0 tpot | Verdict |
| --- | --- | ---: | ---: | --- |
| f4b49b6 | fused RMSNorm/rotary/SwiGLU | 743.5 | 4.86 | win |
| 9fb9dbd | norm+rotary+cache in one kernel | 780.8 | 4.60 | win |
| 3c5eb28 | autotune narrowed to 6 configs | 796.5 | 4.47 | |
| c9a7ac6 | 3c5eb28 tree, byte identical | 766.3 | 5.55 | same code, 24% apart |
| **4f7c38f** | **split-K for narrow projections** | **799.7** | **4.51** | **BEST** |
| 84c5d62 | reproducible autotune | 742.8 | 5.64 | reverted |
| d8bee03 | swiglu epilogue + single-split + deep-K | 595.5 | 6.59 | reverted |
| 005bc30 | pre-swizzled weight layout | 727.5 | 5.79 | reverted |
| 1444d26 | restore of 4f7c38f tree | pending | | |
| 59425ba | replay-timed warmup autotune | pending | | queued |

| 1444d26 | restore of 4f7c38f tree | **failed** | 4.66 | `unstable_timing` |

## Machine health audit: only one run was affected

Prefill is identical in every experiment, so TTFT is a clean indicator of how
the box was performing. Across all 19 runs it sits at 15-22 / 148-152 /
136-141 ms, with exactly one exception: 1444d26 at 21 / 319 / 317, the run
that failed `unstable_timing`.

So the platform was healthy for every other measurement, and the earlier
reverts were justified after all -- 84c5d62 at 5.64, d8bee03 at 6.59 and
005bc30 at 5.79 were all measured on healthy boxes and were real regressions.

What the audit does confirm is the internal variance: 3c5eb28 and c9a7ac6 are
byte-identical, both ran on healthy boxes, and measured 4.47 and 5.55 ms. A
24% swing with the machine ruled out points back at the warmup autotune
choosing different GEMM configs from one run to the next. That is the largest
single lever left, and it is worth more than any kernel change still on the
list: the fast configuration already exists, we just do not select it
reliably.

The fix attempted in 84c5d62 (minimum-of-sweeps timing plus a 3% adoption
margin) made things worse, most likely because the margin biased selection
back toward cuBLAS. The better approach is to remove the timing from the
decision entirely and pin configs by a deterministic rule on shape, so the
same tree always produces the same engine.

## The platform is occasionally unstable too

Run 6d2dd51e ran the byte-identical best tree and **failed the 25% spread
gate**. Its public-0 step was 4.66 ms, right at our best, but public-1 and
public-2 came back with TTFT of 319 ms and 317 ms against roughly 148 ms and
136 ms on every earlier run of the same code. Prefill did not change; the
machine did. An earlier run of the same tree was also cancelled with
`harness_error`.

This reframes the recent history. Identical code has now produced 4.47, 4.51,
4.66 and 5.55 ms on public-0, and a run that failed outright. Some of the
"regressions" reverted after 4f7c38f were probably partly this, which means
single-run attribution is unreliable at the moment and a change needs to be
worth well over 20% before one measurement can distinguish it from the
infrastructure.

Practical consequence: prefer re-measuring the same tree to get a second
sample before believing any result, and treat `unstable_timing` and
`harness_error` on a known-good tree as platform faults to retry rather than
signals to change code.

Everything attempted after 4f7c38f has been a regression. The cheap
launch-collapsing wins are spent, and the three structural attempts since
(autotune margin, further epilogue fusion, weight relayout) all made the step
slower rather than faster. Pre-swizzling is the most surprising: giving each
program one contiguous tile instead of BLOCK_N strided runs should help
streaming, and it cost 28%. The likely explanation is that the relayout adds
~8GB of resident weights on top of the originals and the fused copies, and
the extra footprint hurts more than the access pattern helps -- worth
retrying only with the originals freed.

## Two things that govern how results must be read

**Run-to-run noise on the score is about 4%.** Anything smaller carries no
information. Several reverts above were chasing noise and cost cycles.

**The autotune is itself a variance source, and a bigger one.** c9a7ac6 was
byte-identical to 3c5eb28 and measured 766.3 / 5.55 ms against 796.5 / 4.47 ms.
A 24% swing in time per token from identical code means warmup was picking
different GEMM configs on different runs -- it times launches lasting tens of
microseconds, so interference during warmup can lock in a worse config for
every sample in that workload. Fixing the measurement (minimum of several
sweeps, plus a margin before switching) is worth more than most kernel work,
because the good configuration already existed; we were just failing to find
it reliably.

**Speculative decoding lost badly here** and is abandoned: 747 -> 507, with
the step going 4.73 -> 7.39 ms. Verification runs the full model over five
positions, and the per-step host work -- building drafts, uploading them,
reading predictions back -- reintroduces exactly the synchronisation the
graphed single-token path had removed. At 32-128 output tokens there is not
enough repetition for acceptance to pay for that.
| residual epilogue + 12-config autotune | 7b48455 | | | | 719.1 | Regression. Later split: the 12 configs were the loss, the epilogue a win. |
| one kernel for norm+rotary+cache write | 9fb9dbd | | | | 780.8 | +8.5% while still carrying the 12-config loss. |
| attention autotune + RMSNorm into GEMM | 7c2a739 | | | | 772.5 | Within noise. |
| autotune narrowed back to 6 configs | 3c5eb28 | | | | **796.5** | Best. Confirms the widening was the regression. |
| residual epilogue disabled | 9db3ed8 | | | | 762.3 | -4%, TPOT 4.47 -> 5.55. Epilogue earns its place; restored. |
| SwiGLU epilogue + sync chunk 16 | 23496cb | | | | 747.0 | Every public TPOT improved yet score fell. See noise note. |
| speculative decoding | 47335e6 | | | | pending | |
| verification through the fused path | 5db42b5 | | | | pending | |
| residual epilogue restored | 629e156 | | | | pending | |

**Score noise is roughly +/-2-3%.** 23496cb improved all three public TPOTs
versus its parent and still scored 2% lower, so anything under about 5% on the
private geomean is not signal. Only changes that move TPOT visibly, or the
score by more than ~5%, have been trusted.
| residual in GEMM epilogue + 12-config autotune | 7b48455 / b3f0f39a | 190.6 | 377.3 | 2246.8 | 719.1 | Lost to f4b49b6 despite public-0 gain. Candidate ms 167.9/339.3/911.5; engine reverted. |
| one kernel for norm+rotary+cache write | 9fb9dbd / db4bf30d | 201.8 | 402.7 | 2464.6 | 780.8 | New banked best. Candidate ms 158.6/317.9/831.0. |

## What the measurements actually said

TPOT on public-0 is the cleanest decode signal: 9.17 -> 7.74 -> 7.55 -> 4.86 ms.

**Host-side stall was never the problem.** Batching the per-step device sync
eight ways moved TPOT only 7.74 -> 7.55 ms. Had the per-step `.tolist()`
stall been significant, that change alone would have been large. It was not,
so the step is real GPU work and there is no point chasing the host further.

**Launch count was the problem.** Removing roughly 1100 launches took 7.55 ->
4.86 ms, about 2.3 us apiece. That is too much to be launch overhead alone:
these kernels are small enough that memory latency sets their cost, so fusing
removes a round trip as well as a launch. This is why fusion kept paying when
the custom GEMM did not.

**The projections were never as far off peak as they looked.** At 4.86 ms with
~550 launches left, overhead is ~1.3 ms and the remaining ~3.6 ms streams
8.05 GB, which is ~2200 GB/s or about two thirds of peak. cuBLAS won most of
the autotune. The custom GEMM's real value turned out to be the fused
epilogue, not beating cuBLAS at the matmul.

**Native's own timings move a lot between runs.** Reference TPOT came back at
20.7, 35.6 and 19.7 ms across three runs of the same reference. The latency
gates are ratios against that, so they have slack, but no single native
number should be treated as ground truth.

## Remaining budget

Against the ~2.4 ms bandwidth floor, a realistic best is ~3.5 ms once launches
are minimal: ~0.9 ms of irreducible overhead plus ~2.6 ms of weight streaming.
On this scoring that is roughly 1030. Going past it requires more than one
token per weight read, i.e. speculative decoding with exact verification —
which is also the one optimization whose cost depends on the prompt, and so
sits in direct tension with the 25% spread gate we currently clear at 0.3%.
