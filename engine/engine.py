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

from kernels.flash_decode import BLOCK_M, choose_splits, flash_decode

DEVICE = "cuda:0"

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

        Originals are dropped so the fused copies do not double weight memory.
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
            del attn.q_proj, attn.k_proj, attn.v_proj

            mlp = layer.mlp
            mlp.gateup_weight = torch.cat(
                [mlp.gate_proj.weight, mlp.up_proj.weight], dim=0
            )
            del mlp.gate_proj, mlp.up_proj

        torch.cuda.empty_cache()
        _log(
            f"fused projections qkv={self.q_size + 2 * self.kv_size} "
            f"gateup={2 * self.mlp_size} "
            f"weights={torch.cuda.memory_allocated() / 2**30:.2f}GiB"
        )

    def _mlp(self, layer, hidden):
        gate_up = F.linear(hidden, layer.mlp.gateup_weight)
        gate = gate_up[..., : self.mlp_size]
        up = gate_up[..., self.mlp_size :]
        return layer.mlp.down_proj(F.silu(gate) * up)

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

        Warmup is untimed, so this is free. A kernel that disagrees costs us
        speed by falling back; one that silently disagreed would cost the run.
        """
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

        qkv = F.linear(normed, attn.qkv_weight)
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
        hidden = residual + attn.o_proj(attended)
        return hidden + self._mlp(layer, layer.post_attention_layernorm(hidden))

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
        logits = self.model.lm_head(self.base.norm(hidden))

        self.next_token.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        self.step_token.copy_(self.next_token)
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
                self._decode_step()
        torch.cuda.current_stream().wait_stream(stream)

        self.cur_pos.zero_()
        self.step_token.zero_()
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
            self._capture()
        self._warmed = True

        with torch.inference_mode():
            ids = torch.tensor(input_ids, dtype=torch.int64, device=DEVICE)
            self.cur_pos.zero_()
            first = self._prefill(ids)[:, -1, :].argmax(dim=-1, keepdim=True)
            self.step_token.copy_(first)
            # Prefill filled slots 0..S-1; the token just chosen lands at S.
            self.cur_pos.fill_(prompt_length)
            yield first[:, 0].tolist()

            for _ in range(max_new_tokens - 1):
                if use_graph:
                    self._graph.replay()
                else:
                    self._decode_step()
                yield self.next_token[:, 0].tolist()
