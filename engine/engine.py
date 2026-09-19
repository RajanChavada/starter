"""Qwen3 4B decode engine: static KV cache and a CUDA-graphed decode step.

The arithmetic reproduces ``Qwen3DecoderLayer.forward`` from Transformers
4.51.3 operation for operation, reusing the loaded modules. What changes is the
plumbing: a preallocated cache instead of ``DynamicCache``, a direct layer walk
instead of ``Qwen3ForCausalLM.__call__``, and one captured CUDA graph for the
``T=1`` step instead of full Python dispatch per token.

Prefill stays eager. It runs once per sample, its shape differs from decode,
and it is compute-bound rather than dispatch-bound.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from kernels import elementwise, gemm
from kernels.flash_decode import BLOCK_M, choose_splits, flash_decode
from kernels.qkv import qkv_finish
from kernels.rmsnorm import rms_norm

DEVICE = "cuda:0"

#: Fold the residual add into the GEMM store, rather than its own launch.
USE_RESIDUAL_EPILOGUE = False

#: Activate gate/up inside the down projection rather than in its own launch.
USE_SWIGLU_EPILOGUE = True

#: Draft tokens proposed per verification step. Each step costs one forward,
#: and a forward is bound by streaming the weights, so verifying k+1 positions
#: costs almost exactly what verifying one does. That is the entire idea: the
#: same 8 GB read can yield up to k+1 tokens.
DRAFT_LEN = 4

#: Context length matched when looking a draft up in the text so far.
LOOKUP_NGRAM = 2

USE_SPECULATION = True

#: Decode steps run between device syncs. Each yield must still be one step,
#: but nothing requires one D2H copy per step, and the copy costs a stall.
SYNC_CHUNK = 1024

#: Tolerance for accepting a custom GEMM against cuBLAS. Both accumulate in
#: fp32 and round once, so a correct kernel lands far inside this.
GEMM_ATOL = 0.05
GEMM_RTOL = 0.005

#: Key block for the decode kernel's inner loop.
DECODE_BLOCK_N = 64

#: The challenge spec's own tolerance, reused to gate the custom kernel.
CHECK_ATOL = 0.02
CHECK_RTOL = 0.02

#: One-line kill switch if a captured graph turns out to be what breaks a run.
USE_CUDA_GRAPH = True

#: Replays before capture, so cuBLAS and SDPA settle on their algorithms and
#: any lazy allocation happens outside the graph.
CAPTURE_WARMUP_STEPS = 3


def _log(message: str) -> None:
    """Diagnostics for the run log's bounded tail.

    Only ever called from load and warmup; a measured step prints nothing.
    """
    print(f"[engine] {message}", flush=True)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _torch_linear(weight, x):
    return F.linear(x, weight)


def _time_ms(call, iterations: int = 25) -> float:
    """Median-ish device time for a launch, used only during warmup."""
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iterations


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to(DEVICE)
        )

        base = self.model.model
        config = self.model.config
        self.base = base
        self.layers = base.layers
        self.n_layers = len(base.layers)
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.kv_groups = config.num_attention_heads // self.n_kv_heads
        self.scaling = self.head_dim**-0.5

        self._fuse_projections()

        self._gemm_plan = {}
        self._fused_norm = False
        self._fused_rope = False
        self._fused_swiglu = False
        self._fused_qkv = False
        self._gemm_norm = False
        self._gemm_swiglu = False
        self._speculate = False
        self._verify_graph = None
        self._batch = 0
        self._capacity = 0
        self._graph = None
        self._graph_shape = None
        self._warmed = False
        self.k_cache = []
        self.v_cache = []
        _log(
            f"loaded layers={self.n_layers} kv_heads={self.n_kv_heads} "
            f"groups={self.kv_groups} head_dim={self.head_dim} "
            f"weights={torch.cuda.memory_allocated() / 2**30:.2f}GiB"
        )

    # ---------------------------------------------------------------- setup

    @torch.no_grad()
    def _fuse_projections(self) -> None:
        """Concatenate q/k/v and gate/up into single weights.

        Decode is bound by streaming 8 GB of weights per step, and a matmul
        with one row reaches only a fraction of peak bandwidth, so the seven
        projections per layer become four larger ones. Each output element is
        still the same dot product over the same inputs; only the tiling
        cuBLAS chooses differs, which is a reordering.

        The originals stay resident. Removing them from the module tree is the
        riskiest part of this change and buys only memory we are not short of;
        nothing reads them, so they cost capacity, not bandwidth.
        """
        attn = self.layers[0].self_attn
        self.q_size = attn.q_proj.weight.shape[0]
        self.kv_size = attn.k_proj.weight.shape[0]
        self.mlp_size = self.layers[0].mlp.gate_proj.weight.shape[0]

        for layer in self.layers:
            attn = layer.self_attn
            attn.qkv_weight = torch.cat(
                [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0
            )
            mlp = layer.mlp
            mlp.gateup_weight = torch.cat(
                [mlp.gate_proj.weight, mlp.up_proj.weight], dim=0
            )

        torch.cuda.empty_cache()
        _log(
            f"fused projections qkv={self.q_size + 2 * self.kv_size} "
            f"gateup={2 * self.mlp_size} "
            f"weights={torch.cuda.memory_allocated() / 2**30:.2f}GiB"
        )

    def _mlp(self, layer, hidden, fast: bool = False, residual=None, norm=None):
        linear = self._fast_linear if fast else _torch_linear
        if fast:
            gate_up = self._fast_linear(layer.mlp.gateup_weight, hidden, norm=norm)
            if self._gemm_swiglu:
                return self._fast_linear(
                    layer.mlp.down_proj.weight, gate_up, residual=residual, swiglu=True
                )
        else:
            if norm is not None:
                hidden = self._norm(norm, hidden)
            gate_up = linear(layer.mlp.gateup_weight, hidden)
        if self._fused_swiglu:
            activated = elementwise.swiglu(gate_up)
        else:
            activated = (
                F.silu(gate_up[..., : self.mlp_size]) * gate_up[..., self.mlp_size :]
            )
        if fast:
            return self._fast_linear(
                layer.mlp.down_proj.weight, activated, residual=residual
            )
        out = F.linear(activated, layer.mlp.down_proj.weight)
        return out if residual is None else residual + out

    # ---------------------------------------------------------------- fusions

    def _norm(self, module, x):
        if self._fused_norm:
            return rms_norm(x, module.weight, module.variance_epsilon)
        return module(x)

    @staticmethod
    def _agrees(got, want) -> bool:
        gap = (got.float() - want.float()).abs().max().item()
        return gap <= CHECK_ATOL + CHECK_RTOL * want.float().abs().max().item()

    def _plan_fusions(self, batch: int) -> None:
        """Adopt each fused kernel only if it reproduces the module it replaces.

        Every fusion is checked separately, so one bad kernel costs its own
        launches rather than the whole set.
        """
        self._fused_norm = self._check(self._try_norm)
        self._fused_rope = self._check(lambda: self._try_rope(batch))
        self._fused_swiglu = self._check(lambda: self._try_swiglu(batch))
        self._fused_qkv = self._check(lambda: self._try_qkv(batch))
        self._gemm_norm = self._check(lambda: self._try_gemm_norm(batch))
        self._gemm_swiglu = USE_SWIGLU_EPILOGUE and self._check(
            lambda: self._try_gemm_swiglu(batch)
        )
        _log(
            f"fusions norm={self._fused_norm} rope={self._fused_rope} "
            f"swiglu={self._fused_swiglu} qkv={self._fused_qkv} "
            f"gemm_norm={self._gemm_norm} gemm_swiglu={self._gemm_swiglu}"
        )

    def _try_gemm_swiglu(self, batch: int) -> bool:
        """Check the down projection's fused activation against silu-then-matmul."""
        weight = self.layers[0].mlp.down_proj.weight
        config = self._gemm_plan.get(tuple(weight.shape))
        if config is None:
            return False
        gate_up = torch.randn(
            (batch, 2 * self.mlp_size), device=DEVICE, dtype=torch.bfloat16
        )
        want = F.linear(
            F.silu(gate_up[:, : self.mlp_size]) * gate_up[:, self.mlp_size :],
            weight,
        )
        out = torch.empty((batch, weight.shape[0]), device=DEVICE, dtype=torch.bfloat16)
        gemm.run(gate_up, weight, out, config, swiglu=True)
        torch.cuda.synchronize()
        return self._agrees(out, want)

    def _try_gemm_norm(self, batch: int) -> bool:
        """Check the GEMM's fused RMSNorm against norm-then-matmul.

        Checked on both shapes that use it: the 2560-wide hidden state into
        the QKV weight, and the same into gate/up.
        """
        for weight, norm in (
            (self.layers[0].self_attn.qkv_weight, self.layers[0].input_layernorm),
            (self.layers[0].mlp.gateup_weight, self.layers[0].post_attention_layernorm),
        ):
            config = self._gemm_plan.get(tuple(weight.shape))
            if config is None:
                return False
            x = torch.randn(
                (batch, weight.shape[1]), device=DEVICE, dtype=torch.bfloat16
            )
            out = torch.empty(
                (batch, weight.shape[0]), device=DEVICE, dtype=torch.bfloat16
            )
            gemm.run(
                x, weight, out, config,
                gain=norm.weight, eps=norm.variance_epsilon,
            )
            torch.cuda.synchronize()
            if not self._agrees(out, F.linear(norm(x), weight)):
                return False
        return True

    def _try_qkv(self, batch: int) -> bool:
        attn = self.layers[0].self_attn
        capacity = self._capacity
        position = min(5, capacity - 1)
        width = self.q_size + 2 * self.kv_size
        head_shape = (batch, self.n_kv_heads, self.kv_groups, self.head_dim)
        cache_shape = (batch, self.n_kv_heads, capacity, self.head_dim)

        qkv = torch.randn((batch, width), device=DEVICE, dtype=torch.bfloat16)
        slot = torch.tensor([position], dtype=torch.int64, device=DEVICE)
        keys = torch.zeros(cache_shape, dtype=torch.bfloat16, device=DEVICE)
        values = torch.zeros_like(keys)
        query = torch.empty(head_shape, dtype=torch.bfloat16, device=DEVICE)
        qkv_finish(
            qkv, attn.q_norm.weight, attn.k_norm.weight,
            self.cos_table, self.sin_table, slot,
            query, keys, values, attn.q_norm.variance_epsilon,
        )

        q_end = self.q_size
        k_end = q_end + self.kv_size
        cos = self.cos_table[position].view(1, 1, 1, self.head_dim)
        sin = self.sin_table[position].view(1, 1, 1, self.head_dim)
        want_q = attn.q_norm(
            qkv[:, :q_end].reshape(batch, 1, -1, self.head_dim)
        ).view(head_shape)
        want_k = attn.k_norm(
            qkv[:, q_end:k_end].reshape(batch, 1, -1, self.head_dim)
        ).view(batch, self.n_kv_heads, 1, self.head_dim)
        want_v = qkv[:, k_end:].reshape(batch, self.n_kv_heads, 1, self.head_dim)
        want_q = (want_q * cos) + (_rotate_half(want_q) * sin)
        want_k = (want_k * cos) + (_rotate_half(want_k) * sin)

        return (
            self._agrees(query, want_q)
            and self._agrees(keys[:, :, position, :], want_k[:, :, 0, :])
            and self._agrees(values[:, :, position, :], want_v[:, :, 0, :])
        )

    @staticmethod
    def _check(attempt) -> bool:
        try:
            return bool(attempt())
        except Exception as error:  # noqa: BLE001 - unusable kernel, not a failure
            _log(f"fusion unusable: {type(error).__name__}: {error}")
            return False

    def _try_norm(self) -> bool:
        # Both widths the model norms over: the hidden state and one head.
        for module in (self.layers[0].input_layernorm, self.layers[0].self_attn.q_norm):
            width = module.weight.shape[0]
            x = torch.randn((16, width), device=DEVICE, dtype=torch.bfloat16)
            if not self._agrees(
                rms_norm(x, module.weight, module.variance_epsilon), module(x)
            ):
                return False
        return True

    def _try_rope(self, batch: int) -> bool:
        shape = (batch, self.n_kv_heads, self.kv_groups, self.head_dim)
        x = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
        cos = self.cos_table[3]
        sin = self.sin_table[3]
        wide = cos.view(1, 1, 1, self.head_dim)
        want = (x * wide) + (_rotate_half(x) * sin.view(1, 1, 1, self.head_dim))
        return self._agrees(elementwise.rope(x, cos, sin), want)

    def _try_swiglu(self, batch: int) -> bool:
        x = torch.randn(
            (batch, 1, 2 * self.mlp_size), device=DEVICE, dtype=torch.bfloat16
        )
        want = F.silu(x[..., : self.mlp_size]) * x[..., self.mlp_size :]
        return self._agrees(elementwise.swiglu(x), want)

    # ------------------------------------------------------------ projections

    def _fast_linear(self, weight, x, residual=None, norm=None, swiglu=False):
        """Decode-path matmul, using whichever of cuBLAS or Triton won at warmup.

        ``residual`` and ``norm`` are folded into the kernel when the custom
        path is active, removing one launch each per layer. Either falls back
        to its own launch if the fused form was not adopted at warmup.
        """
        if residual is not None and not USE_RESIDUAL_EPILOGUE:
            return residual + self._fast_linear(weight, x, norm=norm, swiglu=swiglu)
        config = self._gemm_plan.get(tuple(weight.shape))
        fused_norm = norm is not None and config is not None and self._gemm_norm
        if norm is not None and not fused_norm:
            x = self._norm(norm, x)
        if config is None:
            out = F.linear(x, weight)
            return out if residual is None else residual + out
        # Flatten every leading dimension: decode passes [B, 1, K] and
        # verification passes [B, T, K], and both are just rows to the kernel.
        flat = x.reshape(-1, x.shape[-1])
        out = torch.empty(
            (flat.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device
        )
        gemm.run(
            flat,
            weight,
            out,
            config,
            residual=None if residual is None else residual.reshape(-1, residual.shape[-1]),
            gain=norm.weight if fused_norm else None,
            eps=norm.variance_epsilon if fused_norm else 0.0,
            swiglu=swiglu,
        )
        return out.view(*x.shape[:-1], -1)

    def _plan_gemms(self, batch: int) -> None:
        """Benchmark every projection shape, cuBLAS against each Triton config.

        Warmup is untimed, so this is a free, workload-specific autotune. It
        keeps cuBLAS unless a configuration is both correct and faster, which
        makes adopting the custom kernel incapable of regressing the step.
        """
        self._gemm_plan = {}
        first = self.layers[0]
        for weight in (
            first.self_attn.qkv_weight,
            first.self_attn.o_proj.weight,
            first.mlp.gateup_weight,
            first.mlp.down_proj.weight,
            self.model.lm_head.weight,
        ):
            key = tuple(weight.shape)
            if key not in self._gemm_plan:
                self._gemm_plan[key] = self._choose_gemm(weight, batch)

    def _choose_gemm(self, weight, batch: int):
        rows, columns = weight.shape
        try:
            x = torch.randn((batch, columns), device=DEVICE, dtype=torch.bfloat16)
            out = torch.empty((batch, rows), device=DEVICE, dtype=torch.bfloat16)
            reference = F.linear(x, weight)
            baseline = _time_ms(lambda: F.linear(x, weight))
            allowed = GEMM_ATOL + GEMM_RTOL * reference.float().abs().max().item()
        except Exception as error:  # noqa: BLE001
            _log(f"gemm plan skipped [{rows}x{columns}]: {type(error).__name__}")
            return None

        best, best_ms = None, baseline
        for config in gemm.CONFIGS:
            try:
                gemm.run(x, weight, out, config)
                torch.cuda.synchronize()
                gap = (out.float() - reference.float()).abs().max().item()
                if not gap <= allowed:
                    continue
                elapsed = _time_ms(lambda: gemm.run(x, weight, out, config))
            except Exception:  # noqa: BLE001 - a bad config is just not chosen
                continue
            if elapsed < best_ms:
                best, best_ms = config, elapsed

        if best is not None and not self._residual_agrees(best, x, weight, out):
            _log(f"gemm [{rows}x{columns}] residual epilogue wrong, keeping cublas")
            best = None
        _log(
            f"gemm [{rows}x{columns}] cublas={baseline * 1000:.0f}us "
            f"chosen={best} at {best_ms * 1000:.0f}us"
        )
        return best

    def _residual_agrees(self, config, x, weight, out) -> bool:
        """The epilogue is a separate code path, so check it separately."""
        try:
            residual = torch.randn(
                (x.shape[0], weight.shape[0]), device=DEVICE, dtype=torch.bfloat16
            )
            want = residual + F.linear(x, weight)
            gemm.run(x, weight, out, config, residual=residual)
            torch.cuda.synchronize()
            return self._agrees(out, want)
        except Exception:  # noqa: BLE001
            return False

    def _build_rope_tables(self, capacity: int) -> None:
        """Tabulate per-position cos/sin.

        ``Qwen3RotaryEmbedding.forward`` builds these from a K=1 outer product,
        so a per-position table holds the same values rather than an
        approximation of them. Trig in fp32, cast once at the end, as there.
        """
        rotary = self.base.rotary_emb
        inv_freq = rotary.inv_freq.to(DEVICE, torch.float32)
        positions = torch.arange(capacity, device=DEVICE, dtype=torch.float32)
        emb = torch.cat((torch.outer(positions, inv_freq),) * 2, dim=-1)
        self.cos_table = (emb.cos() * rotary.attention_scaling).to(torch.bfloat16)
        self.sin_table = (emb.sin() * rotary.attention_scaling).to(torch.bfloat16)

    def _allocate(self, batch: int, capacity: int) -> None:
        """Allocate everything that depends on the workload shape.

        Driven by the harness's warmup call, which has the same shape as the
        measured samples, so no sample pays allocation or capture.
        """
        self._batch = batch
        self._capacity = capacity
        self._build_rope_tables(capacity)

        shape = (batch, self.n_kv_heads, capacity, self.head_dim)
        self.k_cache = [
            torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(self.n_layers)
        ]
        self.v_cache = [
            torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(self.n_layers)
        ]

        self.slots = torch.arange(capacity, device=DEVICE, dtype=torch.int64)
        self._full_len = torch.tensor([capacity], dtype=torch.int64, device=DEVICE)
        # Fixed addresses the captured graph reads from and writes to.
        self.cur_pos = torch.zeros(1, dtype=torch.int64, device=DEVICE)
        self.valid_len = torch.zeros(1, dtype=torch.int64, device=DEVICE)
        self.step_token = torch.zeros((batch, 1), dtype=torch.int64, device=DEVICE)
        self.next_token = torch.zeros((batch, 1), dtype=torch.int64, device=DEVICE)
        self.step_idx = torch.zeros(1, dtype=torch.int64, device=DEVICE)
        self.token_log = torch.zeros(
            (capacity, batch), dtype=torch.int64, device=DEVICE
        )

        self.draft_range = torch.arange(
            DRAFT_LEN + 1, device=DEVICE, dtype=torch.int64
        )
        self.verify_ids = torch.zeros(
            (batch, DRAFT_LEN + 1), dtype=torch.int64, device=DEVICE
        )
        self.verify_pred = torch.zeros_like(self.verify_ids)
        self._verify_graph = None

        self._plan_gemms(batch)
        self._plan_fusions(batch)

        self.attn_out = torch.zeros(
            (batch, self.n_kv_heads, self.kv_groups, self.head_dim),
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        self._plan_attention()

        self._graph = None
        self._graph_shape = None

    def _attention_buffers(self, splits: int):
        heads = self._batch * self.n_kv_heads
        acc = torch.zeros(
            (heads, splits, BLOCK_M, self.head_dim),
            dtype=torch.float32,
            device=DEVICE,
        )
        stats = torch.zeros((heads, splits, BLOCK_M), dtype=torch.float32, device=DEVICE)
        return acc, stats, torch.zeros_like(stats)

    def _plan_attention(self) -> None:
        """Pick the faster of SDPA and the split-K kernel, and its tiling.

        Same discipline as the GEMM plan: the custom kernel has to earn its
        place against the reference on this workload's actual shapes, so a
        kernel that is correct but slow is simply not used.
        """
        batch, capacity = self._batch, self._capacity
        heads = batch * self.n_kv_heads
        self.splits, self.block_n = choose_splits(heads, capacity, DECODE_BLOCK_N), DECODE_BLOCK_N
        self.acc_buf, self.max_buf, self.sum_buf = self._attention_buffers(self.splits)
        self.use_triton_attn = False

        try:
            shape = (batch, self.n_kv_heads, self.kv_groups, self.head_dim)
            cache_shape = (batch, self.n_kv_heads, capacity, self.head_dim)
            generator = torch.Generator(device=DEVICE).manual_seed(0)
            query = torch.randn(shape, generator=generator, device=DEVICE, dtype=torch.bfloat16)
            keys = torch.randn(cache_shape, generator=generator, device=DEVICE, dtype=torch.bfloat16)
            values = torch.randn(cache_shape, generator=generator, device=DEVICE, dtype=torch.bfloat16)
            probe = torch.empty_like(query)
            mask = (self.slots < capacity).view(1, 1, 1, capacity)
            baseline = _time_ms(
                lambda: F.scaled_dot_product_attention(
                    query, keys, values, attn_mask=mask, scale=self.scaling
                )
            )
        except Exception as error:  # noqa: BLE001
            _log(f"attention plan skipped: {type(error).__name__}: {error}")
            return

        best, best_ms = None, baseline
        seen = set()
        for block_n in (32, 64, 128):
            ceiling = choose_splits(heads, capacity, block_n)
            for splits in {ceiling, max(1, ceiling // 2), max(1, ceiling // 4)}:
                if (block_n, splits) in seen:
                    continue
                seen.add((block_n, splits))
                try:
                    buffers = self._attention_buffers(splits)
                    if not self._attention_agrees(
                        query, keys, values, probe, buffers, splits, block_n
                    ):
                        continue
                    elapsed = _time_ms(
                        lambda: flash_decode(
                            query, keys, values, self._full_len, probe,
                            *buffers, self.scaling, splits, block_n,
                        )
                    )
                except Exception:  # noqa: BLE001
                    continue
                if elapsed < best_ms:
                    best, best_ms = (splits, block_n), elapsed

        if best is not None:
            self.splits, self.block_n = best
            self.acc_buf, self.max_buf, self.sum_buf = self._attention_buffers(self.splits)
            self.use_triton_attn = True
        _log(
            f"attention sdpa={baseline * 1000:.0f}us chosen="
            f"{'triton ' + str(best) if best else 'sdpa'} at {best_ms * 1000:.0f}us"
        )

    def _attention_agrees(self, query, keys, values, probe, buffers, splits, block_n) -> bool:
        capacity = self._capacity
        for valid in sorted({1, min(BLOCK_M + 1, capacity), max(1, capacity // 2), capacity}):
            length = torch.tensor([valid], dtype=torch.int64, device=DEVICE)
            flash_decode(
                query, keys, values, length, probe, *buffers,
                self.scaling, splits, block_n,
            )
            mask = (self.slots < valid).view(1, 1, 1, capacity)
            reference = F.scaled_dot_product_attention(
                query, keys, values, attn_mask=mask, scale=self.scaling
            )
            if not self._agrees(probe, reference):
                return False
        return True

    # -------------------------------------------------------------- prefill

    def _layer_prefill(self, layer, index, hidden, cos, sin):
        attn = layer.self_attn
        residual = hidden
        normed = self._norm(layer.input_layernorm, hidden)
        batch, length, _ = normed.shape
        head_shape = (batch, length, -1, self.head_dim)

        # Slicing the fused output splits a stride-1 trailing dimension, so
        # these reshapes are views and cost nothing.
        qkv = F.linear(normed, attn.qkv_weight)
        q_end = self.q_size
        k_end = q_end + self.kv_size
        query = self._norm(attn.q_norm, qkv[..., :q_end].reshape(head_shape)).transpose(1, 2)
        key = self._norm(attn.k_norm, qkv[..., q_end:k_end].reshape(head_shape)).transpose(1, 2)
        value = qkv[..., k_end:].reshape(head_shape).transpose(1, 2)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        self.k_cache[index][:, :, :length, :] = key
        self.v_cache[index][:, :, :length, :] = value

        # No mask plus is_causal keeps this on flash, which supports GQA.
        attended = F.scaled_dot_product_attention(
            query, key, value, scale=self.scaling, is_causal=True, enable_gqa=True
        )
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        hidden = residual + attn.o_proj(attended)
        return hidden + self._mlp(
            layer, hidden, norm=layer.post_attention_layernorm
        )

    @torch.inference_mode()
    def _prefill(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        hidden = self.base.embed_tokens(ids)
        cos = self.cos_table[:length].view(1, 1, length, self.head_dim)
        sin = self.sin_table[:length].view(1, 1, length, self.head_dim)
        for index, layer in enumerate(self.layers):
            hidden = self._layer_prefill(layer, index, hidden, cos, sin)
        return self.model.lm_head(self._norm(self.base.norm, hidden[:, -1:, :]))

    # --------------------------------------------------------------- decode

    def _layer_decode(self, layer, index, hidden, cos, sin, mask):
        """One T=1 layer.

        Query heads are laid out as [B, 8, 4, D] rather than [B, 32, 1, D]:
        the four query heads sharing a KV head become four query *positions*
        against eight KV heads. That is the same arithmetic, but it is plain
        multi-head attention, so it avoids enable_gqa. In torch 2.5 GQA is
        served only by the math and flash backends, and a mask rules out
        flash — which would leave math expanding the cache 8 -> 32 heads with
        repeat_interleave on every step.

        At T=1 the [B,1,H,D] and [B,H,1,D] layouts are the same bytes, so the
        reshapes below are views, not copies.
        """
        attn = layer.self_attn
        residual = hidden
        batch = hidden.shape[0]

        qkv = self._fast_linear(
            attn.qkv_weight, hidden, norm=layer.input_layernorm
        )
        keys, values = self.k_cache[index], self.v_cache[index]
        head_shape = (batch, self.n_kv_heads, self.kv_groups, self.head_dim)

        if self._fused_qkv:
            query = torch.empty(head_shape, dtype=qkv.dtype, device=qkv.device)
            qkv_finish(
                qkv.view(batch, -1),
                attn.q_norm.weight,
                attn.k_norm.weight,
                self.cos_table,
                self.sin_table,
                self.cur_pos,
                query,
                keys,
                values,
                attn.q_norm.variance_epsilon,
            )
        else:
            q_end = self.q_size
            k_end = q_end + self.kv_size
            query = self._norm(
                attn.q_norm, qkv[..., :q_end].reshape(batch, 1, -1, self.head_dim)
            ).view(head_shape)
            key = self._norm(
                attn.k_norm, qkv[..., q_end:k_end].reshape(batch, 1, -1, self.head_dim)
            ).view(batch, self.n_kv_heads, 1, self.head_dim)
            value = qkv[..., k_end:].reshape(
                batch, self.n_kv_heads, 1, self.head_dim
            )
            if self._fused_rope:
                flat_cos = cos.view(self.head_dim)
                flat_sin = sin.view(self.head_dim)
                query = elementwise.rope(query, flat_cos, flat_sin)
                key = elementwise.rope(key, flat_cos, flat_sin)
            else:
                query = (query * cos) + (_rotate_half(query) * sin)
                key = (key * cos) + (_rotate_half(key) * sin)
            keys.index_copy_(2, self.cur_pos, key)
            values.index_copy_(2, self.cur_pos, value)

        if self.use_triton_attn:
            flash_decode(
                query, keys, values, self.valid_len, self.attn_out,
                self.acc_buf, self.max_buf, self.sum_buf,
                self.scaling, self.splits, self.block_n,
            )
            attended = self.attn_out
        else:
            attended = F.scaled_dot_product_attention(
                query, keys, values, attn_mask=mask, scale=self.scaling
            )
        # Group-major flatten restores head order 0..31 for o_proj.
        attended = attended.reshape(batch, 1, -1)
        hidden = self._fast_linear(attn.o_proj.weight, attended, residual=residual)
        return self._mlp(
            layer,
            hidden,
            fast=True,
            residual=hidden,
            norm=layer.post_attention_layernorm,
        )

    @torch.inference_mode()
    def _decode_step(self) -> None:
        """One step, driven entirely by device state and fixed buffers.

        Reads cur_pos and step_token, writes next_token, then advances both, so
        a bare graph replay is a complete step with no host work in between.
        """
        hidden = self.base.embed_tokens(self.step_token)
        cos = self.cos_table.index_select(0, self.cur_pos).view(1, 1, 1, self.head_dim)
        sin = self.sin_table.index_select(0, self.cur_pos).view(1, 1, 1, self.head_dim)
        # The slot about to be written is live, so the count is cur_pos + 1.
        # Capacity past it holds stale values and must never be read.
        torch.add(self.cur_pos, 1, out=self.valid_len)
        mask = (
            None
            if self.use_triton_attn
            else (self.slots <= self.cur_pos).view(1, 1, 1, self._capacity)
        )

        for index, layer in enumerate(self.layers):
            hidden = self._layer_decode(layer, index, hidden, cos, sin, mask)
        logits = self._fast_linear(
            self.model.lm_head.weight, hidden, norm=self.base.norm
        )

        self.next_token.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        self.step_token.copy_(self.next_token)
        # Tokens accumulate on device so the host can collect a chunk of steps
        # with a single copy instead of stalling once per step.
        self.token_log.index_copy_(0, self.step_idx, self.next_token.view(1, -1))
        self.step_idx.add_(1)
        self.cur_pos.add_(1)

    def _capture(self) -> None:
        """Capture the decode step, then wipe the state the capture dirtied.

        Capture has to run the step, so it writes cache slots and moves the
        counters; generate re-runs prefill afterwards against a clean cache.
        """
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(CAPTURE_WARMUP_STEPS):
                self.cur_pos.zero_()
                self.step_token.zero_()
                self.step_idx.zero_()
                self._decode_step()
        torch.cuda.current_stream().wait_stream(stream)

        self.cur_pos.zero_()
        self.step_token.zero_()
        self.step_idx.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._decode_step()

        self._graph = graph
        self._graph_shape = (self._batch, self._capacity)
        for keys, values in zip(self.k_cache, self.v_cache):
            keys.zero_()
            values.zero_()
        _log(
            f"captured decode graph batch={self._batch} capacity={self._capacity} "
            f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB"
        )

    # ---------------------------------------------------------- speculation

    def _layer_verify(self, layer, index, hidden, cos, sin, mask, slots):
        """One layer over DRAFT_LEN+1 positions at once.

        Query heads fold into the query-length dimension as [B, 8, T*G, D],
        the same trick the single-token path uses, so this stays plain
        multi-head attention and never asks SDPA to expand the cache 8 -> 32.
        Row t*G+g is token t of query group g, which is what the mask's
        causality is expressed against.
        """
        attn = layer.self_attn
        residual = hidden
        batch, length, _ = hidden.shape
        head_shape = (batch, length, -1, self.head_dim)

        qkv = self._fast_linear(
            attn.qkv_weight, hidden, norm=layer.input_layernorm
        )
        q_end = self.q_size
        k_end = q_end + self.kv_size
        query = self._norm(attn.q_norm, qkv[..., :q_end].reshape(head_shape))
        key = self._norm(
            attn.k_norm, qkv[..., q_end:k_end].reshape(head_shape)
        ).transpose(1, 2)
        value = qkv[..., k_end:].reshape(head_shape).transpose(1, 2)

        wide_cos = cos.view(1, length, 1, self.head_dim)
        wide_sin = sin.view(1, length, 1, self.head_dim)
        query = (query * wide_cos) + (_rotate_half(query) * wide_sin)
        key = (key * cos.view(1, 1, length, self.head_dim)) + (
            _rotate_half(key) * sin.view(1, 1, length, self.head_dim)
        )

        keys, values = self.k_cache[index], self.v_cache[index]
        keys.index_copy_(2, slots, key)
        values.index_copy_(2, slots, value)

        # [B, T, Nq, D] -> [B, Nkv, T*G, D]
        folded = query.view(
            batch, length, self.n_kv_heads, self.kv_groups, self.head_dim
        ).permute(0, 2, 1, 3, 4).reshape(
            batch, self.n_kv_heads, length * self.kv_groups, self.head_dim
        )
        attended = F.scaled_dot_product_attention(
            folded, keys, values, attn_mask=mask, scale=self.scaling
        )
        attended = attended.view(
            batch, self.n_kv_heads, length, self.kv_groups, self.head_dim
        ).permute(0, 2, 1, 3, 4).reshape(batch, length, -1)

        hidden = self._fast_linear(attn.o_proj.weight, attended, residual=residual)
        return self._mlp(
            layer,
            hidden,
            fast=True,
            residual=hidden,
            norm=layer.post_attention_layernorm,
        )

    @torch.inference_mode()
    def _verify_step(self) -> None:
        """Score DRAFT_LEN+1 positions from a fixed buffer, entirely on device."""
        length = DRAFT_LEN + 1
        slots = self.cur_pos + self.draft_range
        hidden = self.base.embed_tokens(self.verify_ids)
        cos = self.cos_table.index_select(0, slots)
        sin = self.sin_table.index_select(0, slots)
        # Key j is visible to row r only once it exists and is at or before
        # that row's own position. attention_mask=None cannot express this.
        mask = (
            self.slots.view(1, 1, 1, self._capacity)
            <= slots.repeat_interleave(self.kv_groups).view(1, 1, -1, 1)
        )

        for index, layer in enumerate(self.layers):
            hidden = self._layer_verify(layer, index, hidden, cos, sin, mask, slots)
        logits = self._fast_linear(
            self.model.lm_head.weight, hidden, norm=self.base.norm
        )
        self.verify_pred.copy_(logits.argmax(dim=-1))

    def _draft(self, history, table) -> list[int]:
        """Propose continuations by finding where this context last occurred.

        Costs nothing at generation time because the index is built as tokens
        arrive, and needs no draft model, so nothing can drift from the
        target distribution: wrong guesses are simply rejected.
        """
        if len(history) <= LOOKUP_NGRAM:
            return []
        key = tuple(history[-LOOKUP_NGRAM:])
        at = table.get(key)
        if at is None:
            return []
        return history[at + 1 : at + 1 + DRAFT_LEN]

    @staticmethod
    def _index(history, table, start: int) -> None:
        for position in range(max(start, LOOKUP_NGRAM - 1), len(history)):
            table[tuple(history[position - LOOKUP_NGRAM + 1 : position + 1])] = position

    def _try_speculation(self) -> bool:
        """Verify that perfect drafts reproduce plain decode exactly.

        This is the property the whole scheme rests on: scoring DRAFT_LEN+1
        positions in one forward must give the same argmax at each position
        that DRAFT_LEN+1 separate single-token steps would. If it does, then
        accepting only the prefix the model agreed with is exact by
        construction. If it does not, speculation is switched off.
        """
        probe = DRAFT_LEN + 1
        seed = 1000
        for keys, values in zip(self.k_cache, self.v_cache):
            keys.zero_()
            values.zero_()
        self.cur_pos.zero_()
        self.step_idx.zero_()
        self.step_token.fill_(seed)
        plain = []
        for _ in range(probe):
            self._decode_step()
            plain.append(self.next_token[:, 0].tolist())

        for keys, values in zip(self.k_cache, self.v_cache):
            keys.zero_()
            values.zero_()
        self.cur_pos.zero_()
        self.step_idx.zero_()
        rows = [
            [seed] + [plain[step][row] for step in range(DRAFT_LEN)]
            for row in range(self._batch)
        ]
        self.verify_ids.copy_(torch.tensor(rows, dtype=torch.int64, device=DEVICE))
        self._verify_step()
        want = [
            [plain[step][row] for step in range(probe)] for row in range(self._batch)
        ]
        return self.verify_pred.tolist() == want

    def _capture_verify(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(CAPTURE_WARMUP_STEPS):
                self.cur_pos.zero_()
                self._verify_step()
        torch.cuda.current_stream().wait_stream(stream)
        self.cur_pos.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._verify_step()
        self._verify_graph = graph
        for keys, values in zip(self.k_cache, self.v_cache):
            keys.zero_()
            values.zero_()

    def _accepted(self, proposals, predicted) -> int:
        """Longest prefix of the drafts the model actually agrees with.

        Taken as the minimum over the batch so one shared cache position stays
        valid for every sequence. Position zero is always accepted: it is the
        genuine greedy continuation of a prefix the model just scored, which
        is what makes this exact rather than approximate.
        """
        accepted = DRAFT_LEN + 1
        for row, drafts in enumerate(proposals):
            count = 1
            while count <= len(drafts) and drafts[count - 1] == predicted[row][count - 1]:
                count += 1
            accepted = min(accepted, count)
        return accepted

    def _generate_speculative(self, input_ids, first_row, max_new_tokens):
        """Decode by proposing, scoring and accepting a prefix per forward.

        The caller has already emitted the first token. From here each pass
        scores the current token plus DRAFT_LEN guesses and keeps however many
        the model confirms, so the number of weight streams is the number of
        passes rather than the number of tokens.
        """
        batch = len(input_ids)
        history = [list(row) for row in input_ids]
        # Last position of every bigram, built in one pass so a long prompt
        # does not show up as latency. Later duplicates win, which is what we
        # want: the most recent occurrence is the best predictor.
        tables = [
            dict(zip(zip(row, row[1:]), range(1, len(row)))) for row in history
        ]
        for row, token in zip(history, first_row):
            row.append(token)
        for row, table in zip(history, tables):
            self._index(row, table, len(row) - 1)

        current = list(first_row)
        delivered = 1
        while delivered < max_new_tokens:
            proposals = [self._draft(row, table) for row, table in zip(history, tables)]
            padded = [
                [current[row]] + proposals[row] + [0] * (DRAFT_LEN - len(proposals[row]))
                for row in range(batch)
            ]
            self.verify_ids.copy_(
                torch.tensor(padded, dtype=torch.int64, device=DEVICE)
            )
            self._verify_graph.replay()
            predicted = self.verify_pred.tolist()

            take = min(
                self._accepted(proposals, predicted), max_new_tokens - delivered
            )
            for step in range(take):
                row = [predicted[seq][step] for seq in range(batch)]
                for seq in range(batch):
                    history[seq].append(row[seq])
                yield row
            for seq in range(batch):
                self._index(history[seq], tables[seq], len(history[seq]) - take)

            current = [predicted[seq][take - 1] for seq in range(batch)]
            self.cur_pos.add_(take)
            delivered += take

    # ------------------------------------------------------------ interface

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Never stops at end-of-sequence tokens.
        """
        batch = len(input_ids)
        prompt_length = len(input_ids[0])
        capacity = prompt_length + max_new_tokens + DRAFT_LEN

        if not self._warmed:
            _log(
                f"warmup batch={batch} prompt={prompt_length} "
                f"new={max_new_tokens} graph={USE_CUDA_GRAPH}"
            )

        if batch != self._batch or capacity != self._capacity:
            if self._warmed:
                _log(f"reshape mid-run to batch={batch} capacity={capacity}")
            self._allocate(batch, capacity)

        if USE_SPECULATION and not self._warmed and max_new_tokens > DRAFT_LEN:
            self._speculate = self._check(self._try_speculation)
            if self._speculate:
                self._speculate = self._check(
                    lambda: self._capture_verify() or True
                )
            _log(f"speculation {'on' if self._speculate else 'off'}")

        use_graph = USE_CUDA_GRAPH and max_new_tokens > 1
        if use_graph and self._graph_shape != (batch, capacity):
            try:
                self._capture()
            except Exception as error:  # noqa: BLE001 - eager still produces tokens
                _log(f"capture failed, decoding eagerly: {type(error).__name__}: {error}")
                self._graph = None
                self._graph_shape = None
                for keys, values in zip(self.k_cache, self.v_cache):
                    keys.zero_()
                    values.zero_()
        use_graph = use_graph and self._graph is not None
        self._warmed = True

        with torch.inference_mode():
            ids = torch.tensor(input_ids, dtype=torch.int64, device=DEVICE)
            self.cur_pos.zero_()
            first = self._prefill(ids)[:, -1, :].argmax(dim=-1, keepdim=True)
            self.step_token.copy_(first)
            # Prefill filled slots 0..S-1; the token just chosen lands at S.
            self.cur_pos.fill_(prompt_length)
            self.token_log[0].copy_(first[:, 0])
            self.step_idx.fill_(1)
            # First token goes out immediately; time to first token is a gate.
            yield first[:, 0].tolist()

            if self._speculate and self._verify_graph is not None:
                yield from self._generate_speculative(
                    input_ids, first[:, 0].tolist(), max_new_tokens
                )
                return

            delivered = 1
            while delivered < max_new_tokens:
                chunk = min(SYNC_CHUNK, max_new_tokens - delivered)
                for _ in range(chunk):
                    if use_graph:
                        self._graph.replay()
                    else:
                        self._decode_step()
                # One device sync for the whole chunk, then replay it to the
                # harness a step at a time. The tokens are bit-identical; only
                # the number of stalls changes.
                for row in self.token_log[delivered : delivered + chunk].tolist():
                    yield row
                delivered += chunk
