# SPDX-License-Identifier: Apache-2.0
"""vLLM metadata adapter for the FlagGems INT8 sparse indexer."""

import os
import flag_gems.fused.int8_sparse_indexer as _hy4_fg_int8

import torch
from torch import nn

from flag_gems.fused.int8_sparse_indexer import (
    int8_indexer_cache_store,
    int8_indexer_paged_logits,
)
from vllm.forward_context import get_forward_context


class HYV4Int8Indexer(nn.Module):
    def __init__(self, cache, topk, max_model_len, indices, parallel_config):
        super().__init__()
        if (
            parallel_config.decode_context_parallel_size != 1
            or parallel_config.prefill_context_parallel_size != 1
        ):
            raise ValueError(
                "HYV4 INT8 indexer currently supports TP/PP without context parallelism"
            )
        self.cache, self.topk = cache, topk
        self.max_model_len, self.indices = max_model_len, indices
        self.dense_mha_metadata_layer_name = ""
        # Freeze the logits provider before warmup/capture; original FG remains the default.
        self._hy4_int8_paged_logits_module = _hy4_fg_int8
        self._hy4_int8_paged_logits_symbol = "int8_indexer_paged_logits"
        self._hy4_int8_empty_tile_enabled = (
            os.getenv("HY4_INT8_PAGED_LOGITS_EMPTY_TILE", "0") == "1"
        )
        if self._hy4_int8_empty_tile_enabled:
            import flaggems_vllm.ops.int8_indexer_empty_tile as _hy4_selected_logits

            self._hy4_int8_paged_logits_module = _hy4_selected_logits
            self._hy4_int8_paged_logits_symbol = "int8_indexer_paged_logits_empty_tile"
        # Resolve once before native graph warmup/capture; no replay-time toggle.
        self._hy4_int8_prefix_topk_enabled = (
            os.getenv("HY4_INT8_PREFIX_TOPK", "0") == "1"
        )
        self._hy4_int8_prefix_topk = None
        self._hy4_supports_int8_prefix_topk = None
        if self._hy4_int8_prefix_topk_enabled:
            from flaggems_vllm.ops.indexer_prefix_topk import (
                indexer_prefix_topk,
                supports_indexer_prefix_topk,
            )

            self._hy4_int8_prefix_topk = indexer_prefix_topk
            self._hy4_supports_int8_prefix_topk = supports_indexer_prefix_topk
        if os.environ.get("HY4_INT8_DECODE_METADATA_FUSION") == "1":
            from vllm.config import get_current_vllm_config
            from vllm.v1.attention.backends.mla.indexer import (
                DeepSeekV32IndexerDecodeMetadata,
                DeepseekV32IndexerMetadata,
            )

            config = get_current_vllm_config()
            parallel = config.parallel_config
            self._hy4_int8_decode_metadata_selector_version = 1
            self._hy4_int8_decode_metadata_variant = "baseline"
            self._hy4_int8_decode_metadata_configuration = (
                parallel.tensor_parallel_size,
                parallel.enable_expert_parallel,
                parallel.data_parallel_size,
                parallel.pipeline_parallel_size,
                parallel.decode_context_parallel_size,
                parallel.prefill_context_parallel_size,
                config.speculative_config is None,
                config.lora_config is None,
                config.scheduler_config.max_num_batched_tokens,
            )
            self._hy4_int8_decode_metadata_types = (
                DeepseekV32IndexerMetadata,
                DeepSeekV32IndexerDecodeMetadata,
            )

    @property
    def topk_indices_buffer(self):
        """Expose the buffer name used by native MTP sharing."""
        return self.indices

    @topk_indices_buffer.setter
    def topk_indices_buffer(self, buffer):
        self.indices = buffer

    def _score(self, q, weights, table, rows, lengths, output):
        import flag_gems

        page_size = self.cache.kv_cache.shape[1]
        width = min(self.max_model_len, table.shape[1] * page_size)
        lengths = lengths.contiguous()
        logits = getattr(
            self._hy4_int8_paged_logits_module, self._hy4_int8_paged_logits_symbol
        )(
            q,
            self.cache.kv_cache,
            weights,
            table,
            rows.contiguous(),
            lengths,
            width,
            page_size,
        )
        count = min(self.topk, width)
        if count:
            if self._hy4_int8_prefix_topk_enabled:
                topk_output = output[:, :count]
                if self._hy4_supports_int8_prefix_topk(
                    logits, lengths, count, topk_output
                ):
                    self._hy4_int8_prefix_topk(logits, lengths, count, topk_output)
                    return
            values, indices = flag_gems.topk(logits, count, dim=-1)
            output[:, :count] = torch.where(torch.isfinite(values), indices, -1).to(
                torch.int32
            )

    def forward(self, hidden_states, q_quant, k, weights):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return self.indices
        meta = metadata[self.cache.prefix]
        int8_indexer_cache_store(
            k, self.cache.kv_cache, meta.slot_mapping, self.cache.kv_cache.shape[1]
        )
        self.indices[: hidden_states.shape[0]].fill_(-1)
        if meta.num_prefills:
            for chunk in meta.prefill.chunks:
                start, end = chunk.token_start, chunk.token_end
                # cu_seqlen_ks points to each query's sequence in the flattened
                # prefill key workspace. We score directly from paged INT8 K.
                rows = torch.searchsorted(
                    chunk.local_cu_seq_lens[1:], chunk.cu_seqlen_ks, right=True
                )
                lengths = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
                self._score(
                    q_quant[start:end],
                    weights[start:end],
                    chunk.block_table,
                    rows,
                    lengths,
                    self.indices[start:end, : self.topk],
                )
        if meta.num_decode_tokens:
            decode = meta.decode
            n = meta.num_decode_tokens
            counts = decode.decode_lens
            prepared_metadata = None
            if os.environ.get("HY4_INT8_FULL_DECODE_METADATA", "0") == "1":
                from vllm.config import CUDAGraphMode

                if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
                    if meta.num_prefills != 0 or hidden_states.shape[0] != n:
                        raise ValueError(
                            "FULL producer requires a complete padded pure-decode batch"
                        )
                    from .decode_metadata import prepare_full_decode_metadata

                    prepared_metadata = prepare_full_decode_metadata(
                        counts, decode.seq_lens, n, q_quant.device
                    )
            if (
                prepared_metadata is None
                and os.environ.get("HY4_INT8_DECODE_METADATA_FUSION") == "1"
                and getattr(self, "_hy4_int8_decode_metadata_variant", "baseline")
                != "baseline"
            ):
                from .decode_metadata_scratch import maybe_prepare_decode_metadata

                prepared_metadata = maybe_prepare_decode_metadata(
                    self, meta, counts, decode.seq_lens, n, q_quant.device
                )
            if prepared_metadata is None:
                rows = torch.repeat_interleave(
                    torch.arange(counts.shape[0], device=q_quant.device),
                    counts,
                    output_size=n,
                )
                starts = counts.cumsum(0) - counts
                steps = torch.arange(n, device=q_quant.device) - starts[rows]
                seq_lens = decode.seq_lens
                if seq_lens.ndim == 2 and seq_lens.shape[1] > 1:
                    lengths = seq_lens[rows, steps]
                else:
                    lengths = seq_lens.reshape(-1)[rows] - counts[rows] + steps + 1
            else:
                rows, lengths = prepared_metadata
            self._score(
                q_quant[:n],
                weights[:n],
                decode.block_table,
                rows,
                lengths,
                self.indices[:n, : self.topk],
            )
        return self.indices
