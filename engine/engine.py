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

from kernels import gemm
from kernels.flash_decode import BLOCK_M, choose_splits, flash_decode

DEVICE = "cuda:0"

#: Decode steps run between device syncs. Each yield must still be one step,
#: but nothing requires one D2H copy per step, and the copy costs a stall.
SYNC_CHUNK = 8

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

    def _mlp(self, layer, hidden, fast: bool = False):
        linear = self._fast_linear if fast else _torch_linear
        gate_up = linear(layer.mlp.gateup_weight, hidden)
        gate = gate_up[..., : self.mlp_size]
        up = gate_up[..., self.mlp_size :]
        return linear(layer.mlp.down_proj.weight, F.silu(gate) * up)

    # ------------------------------------------------------------ projections

    def _fast_linear(self, weight, x):
        """Decode-path matmul, using whichever of cuBLAS or Triton won at warmup."""
        config = self._gemm_plan.get(tuple(weight.shape))
        if config is None:
            return F.linear(x, weight)
        rows = x.shape[0]
        out = torch.empty((rows, weight.shape[0]), dtype=x.dtype, device=x.device)
        gemm.run(x.reshape(rows, -1), weight, out, config)
        return out.view(rows, 1, -1)

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
        _log(
            f"gemm [{rows}x{columns}] cublas={baseline * 1000:.0f}us "
            f"chosen={best} at {best_ms * 1000:.0f}us"
        )
        return best

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
        # Fixed addresses the captured graph reads from and writes to.
        self.cur_pos = torch.zeros(1, dtype=torch.int64, device=DEVICE)
        self.valid_len = torch.zeros(1, dtype=torch.int64, device=DEVICE)
        self.step_token = torch.zeros((batch, 1), dtype=torch.int64, device=DEVICE)
        self.next_token = torch.zeros((batch, 1), dtype=torch.int64, device=DEVICE)
        self.step_idx = torch.zeros(1, dtype=torch.int64, device=DEVICE)
        self.token_log = torch.zeros(
            (capacity, batch), dtype=torch.int64, device=DEVICE
        )

        self._plan_gemms(batch)

        heads = batch * self.n_kv_heads
        self.splits = choose_splits(heads, capacity, DECODE_BLOCK_N)
        self.attn_out = torch.zeros(
            (batch, self.n_kv_heads, self.kv_groups, self.head_dim),
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        self.acc_buf = torch.zeros(
            (heads, self.splits, BLOCK_M, self.head_dim),
            dtype=torch.float32,
            device=DEVICE,
        )
        self.max_buf = torch.zeros(
            (heads, self.splits, BLOCK_M), dtype=torch.float32, device=DEVICE
        )
        self.sum_buf = torch.zeros_like(self.max_buf)

        self.use_triton_attn = self._validate_attention()
        _log(
            f"decode attention: {'triton' if self.use_triton_attn else 'sdpa'} "
            f"splits={self.splits} block_n={DECODE_BLOCK_N}"
        )

        self._graph = None
        self._graph_shape = None

    def _validate_attention(self) -> bool:
        """Check the custom kernel against SDPA before trusting it.

        Warmup is untimed, so this is free. Any disagreement, and any
        exception at all — a Triton compile failure, a bad launch grid, an
        unsupported construct in this Triton build — falls back to the SDPA
        path. A broken kernel then costs throughput instead of the run.
        """
        try:
            return self._compare_attention()
        except Exception as error:  # noqa: BLE001 - fall back on anything
            _log(f"kernel unusable, falling back to sdpa: {type(error).__name__}: {error}")
            return False

    def _compare_attention(self) -> bool:
        batch, capacity = self._batch, self._capacity
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        shape = (batch, self.n_kv_heads, self.kv_groups, self.head_dim)
        cache_shape = (batch, self.n_kv_heads, capacity, self.head_dim)
        query = torch.randn(shape, generator=generator, device=DEVICE, dtype=torch.bfloat16)
        keys = torch.randn(cache_shape, generator=generator, device=DEVICE, dtype=torch.bfloat16)
        values = torch.randn(cache_shape, generator=generator, device=DEVICE, dtype=torch.bfloat16)
        probe = torch.empty_like(query)

        lengths = {1, min(BLOCK_M + 1, capacity), max(1, capacity // 2), capacity}
        for valid in sorted(lengths):
            length = torch.tensor([valid], dtype=torch.int64, device=DEVICE)
            flash_decode(
                query, keys, values, length, probe,
                self.acc_buf, self.max_buf, self.sum_buf,
                self.scaling, self.splits, DECODE_BLOCK_N,
            )
            mask = (self.slots < valid).view(1, 1, 1, capacity)
            reference = F.scaled_dot_product_attention(
                query, keys, values, attn_mask=mask, scale=self.scaling
            )
            gap = (probe.float() - reference.float()).abs().max().item()
            allowed = CHECK_ATOL + CHECK_RTOL * reference.float().abs().max().item()
            if not gap <= allowed:
                _log(f"kernel check FAILED at valid={valid}: {gap:.5f} > {allowed:.5f}")
                return False
        return True

    # -------------------------------------------------------------- prefill

    def _layer_prefill(self, layer, index, hidden, cos, sin):
        attn = layer.self_attn
        residual = hidden
        normed = layer.input_layernorm(hidden)
        batch, length, _ = normed.shape
        head_shape = (batch, length, -1, self.head_dim)

        # Slicing the fused output splits a stride-1 trailing dimension, so
        # these reshapes are views and cost nothing.
        qkv = F.linear(normed, attn.qkv_weight)
        q_end = self.q_size
        k_end = q_end + self.kv_size
        query = attn.q_norm(qkv[..., :q_end].reshape(head_shape)).transpose(1, 2)
        key = attn.k_norm(qkv[..., q_end:k_end].reshape(head_shape)).transpose(1, 2)
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
        return hidden + self._mlp(layer, layer.post_attention_layernorm(hidden))

    @torch.inference_mode()
    def _prefill(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        hidden = self.base.embed_tokens(ids)
        cos = self.cos_table[:length].view(1, 1, length, self.head_dim)
        sin = self.sin_table[:length].view(1, 1, length, self.head_dim)
        for index, layer in enumerate(self.layers):
            hidden = self._layer_prefill(layer, index, hidden, cos, sin)
        return self.model.lm_head(self.base.norm(hidden[:, -1:, :]))

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
        normed = layer.input_layernorm(hidden)
        batch = normed.shape[0]

        qkv = self._fast_linear(attn.qkv_weight, normed)
        q_end = self.q_size
        k_end = q_end + self.kv_size
        query = attn.q_norm(
            qkv[..., :q_end].reshape(batch, 1, -1, self.head_dim)
        ).view(batch, self.n_kv_heads, self.kv_groups, self.head_dim)
        key = attn.k_norm(
            qkv[..., q_end:k_end].reshape(batch, 1, -1, self.head_dim)
        ).view(batch, self.n_kv_heads, 1, self.head_dim)
        value = qkv[..., k_end:].reshape(batch, self.n_kv_heads, 1, self.head_dim)

        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        keys, values = self.k_cache[index], self.v_cache[index]
        keys.index_copy_(2, self.cur_pos, key)
        values.index_copy_(2, self.cur_pos, value)

        if self.use_triton_attn:
            flash_decode(
                query, keys, values, self.valid_len, self.attn_out,
                self.acc_buf, self.max_buf, self.sum_buf,
                self.scaling, self.splits, DECODE_BLOCK_N,
            )
            attended = self.attn_out
        else:
            attended = F.scaled_dot_product_attention(
                query, keys, values, attn_mask=mask, scale=self.scaling
            )
        # Group-major flatten restores head order 0..31 for o_proj.
        attended = attended.reshape(batch, 1, -1)
        hidden = residual + self._fast_linear(attn.o_proj.weight, attended)
        return hidden + self._mlp(
            layer, layer.post_attention_layernorm(hidden), fast=True
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
        logits = self._fast_linear(self.model.lm_head.weight, self.base.norm(hidden))

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

    # ------------------------------------------------------------ interface

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Never stops at end-of-sequence tokens.
        """
        batch = len(input_ids)
        prompt_length = len(input_ids[0])
        capacity = prompt_length + max_new_tokens

        if not self._warmed:
            _log(
                f"warmup batch={batch} prompt={prompt_length} "
                f"new={max_new_tokens} graph={USE_CUDA_GRAPH}"
            )

        if batch != self._batch or capacity != self._capacity:
            if self._warmed:
                _log(f"reshape mid-run to batch={batch} capacity={capacity}")
            self._allocate(batch, capacity)

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
