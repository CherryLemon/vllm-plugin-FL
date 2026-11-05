# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 INT8 metadata using the stock vLLM paged-indexer building blocks."""

from __future__ import annotations

import torch

from vllm import envs
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.attention.backends.mla import indexer as native_indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from .runtime_proof import (
    CPUSequenceLengthsProof,
    get_current_cpu_seq_lens_proof,
    validated_prefill_chunk_bound,
)


class HYV4Int8IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """Build INT8 metadata without requesting the native FP8 logits scheduler.

    Request partitioning, chunk construction and decode tensor preparation use
    stock helpers. The Hy4 contract is CP1 with uncompressed KV. Native module
    globals and native builder classes remain unchanged.
    """

    def __init__(
        self,
        kv_cache_spec,
        layer_names,
        vllm_config,
        device,
        *,
        block_table_width,
        **kwargs,
    ):
        parallel = vllm_config.parallel_config
        if (
            parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or getattr(kv_cache_spec, "compress_ratio", 1) != 1
        ):
            raise NotImplementedError(
                "Hy4 INT8 metadata requires CP1 without KV compression"
            )
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            block_table_width=block_table_width,
            **kwargs,
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        from .runtime_hooks import bind_common_runtime_metadata

        common = common_attn_metadata
        bind_common_runtime_metadata(common)
        query_cpu = common.query_start_loc_cpu
        if (
            query_cpu.device.type != "cpu"
            or query_cpu.ndim != 1
            or query_cpu.dtype not in (torch.int32, torch.int64)
            or query_cpu.shape[0] != common.num_reqs + 1
        ):
            raise ValueError("query starts must be a CPU integer vector")
        if int(query_cpu[0].item()) != 0:
            raise ValueError("query starts must begin at zero")
        query_lens_cpu = torch.diff(query_cpu)
        if (common.num_reqs and int(query_lens_cpu.min().item()) < 0) or int(
            query_lens_cpu.sum().item()
        ) > common.num_actual_tokens:
            raise ValueError("decode lengths exceed their token span")
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common,
                decode_threshold=self.decode_threshold,
                require_uniform=not (self.use_flattening or self.supports_varlen),
                treat_short_extends_as_decodes=True,
            )
        )
        assert num_decodes + num_prefills == common.num_reqs
        assert num_decode_tokens + num_prefill_tokens == common.num_actual_tokens
        decode_lens_cpu = None
        min_decode_len = max_decode_len = 0
        if num_decodes:
            decode_lens_cpu = query_lens_cpu[:num_decodes]
            min_decode_len = int(decode_lens_cpu.min().item())
            max_decode_len = int(decode_lens_cpu.max().item())
            if (
                min_decode_len < 0
                or int(decode_lens_cpu.sum().item()) > num_decode_tokens
            ):
                raise ValueError("decode lengths exceed their token span")
        proof = getattr(common, "cpu_seq_lens_proof", None)
        proof_live = (
            isinstance(proof, CPUSequenceLengthsProof)
            and proof is get_current_cpu_seq_lens_proof()
        )
        prefill_metadata = None
        if num_prefills:
            upper = common.seq_lens_cpu_upper_bound
            assert upper is not None
            query_lens = torch.diff(
                common.query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
            )
            specs = native_indexer.split_indexer_prefill_chunks(
                upper[num_decodes:],
                query_lens,
                self.max_prefill_buffer_size,
                envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024,
                request_offset=num_decodes,
            )
            chunks = []
            for req_slice, query_slice in specs:
                chunk = native_indexer.build_prefill_chunk_metadata(
                    req_slice.start,
                    req_slice.stop,
                    common.query_start_loc,
                    common.query_start_loc_cpu,
                    common.seq_lens,
                    common.seq_lens,
                    upper,
                    common.block_table_tensor,
                    1,
                    query_slice=query_slice,
                    skip_kv_gather=query_slice.start > 0,
                    dcp_rank=0,
                    dcp_world_size=1,
                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                )
                if chunk is None:
                    continue
                chunk.cpu_seq_len_bound = (
                    validated_prefill_chunk_bound(
                        common,
                        chunk,
                        req_slice.start,
                        req_slice.stop,
                        num_decodes,
                        self.decode_threshold,
                        1,
                        1,
                        1,
                    )
                    if proof_live
                    else None
                )
                chunks.append(chunk)
            prefill_metadata = native_indexer.DeepseekV32IndexerPrefillMetadata(chunks)

        decode_metadata = None
        if num_decodes:
            torch.diff(
                common.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            next_n = self.num_speculative_tokens + 1
            step_next_n_ok = (
                max_decode_len <= 1
                or native_indexer._supports_native_decode(max_decode_len)
            )
            use_native = (
                not (self.use_flattening or self.supports_varlen)
                and max_decode_len <= next_n
                and step_next_n_ok
                # Keep native FULL-graph views for uniform live requests and
                # zero-length padding. Real nonuniform rows use stock flattening.
                and max_decode_len > 0
                and bool(
                    torch.all(
                        (decode_lens_cpu == max_decode_len) | (decode_lens_cpu == 0)
                    ).item()
                )
                and num_decode_tokens <= num_decodes * max_decode_len
            )
            indices = (
                self._build_varlen_decode_indices(
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    num_decode_tokens=num_decode_tokens,
                )
                if self.supports_varlen
                else None
            )
            seq_lens, table, decode_lens, _, requires_padding = (
                self._prepare_decode_tensors(
                    seq_lens=common.seq_lens[:num_decodes],
                    block_table=common.block_table_tensor[:num_decodes],
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    query_start_loc=common.query_start_loc[:num_decodes],
                    num_decodes=num_decodes,
                    num_decode_tokens=num_decode_tokens,
                    use_native=use_native,
                    next_n=next_n,
                    max_decode_len=max_decode_len,
                )
            )
            if seq_lens.ndim == 1:
                seq_lens = seq_lens.unsqueeze(-1)
            decode_metadata = native_indexer.DeepSeekV32IndexerDecodeMetadata(
                block_table=table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=self.scheduler_metadata_buffer,
                indices=indices,
                global_seq_lens=None,
            )
        metadata = native_indexer.DeepseekV32IndexerMetadata(
            seq_lens=common.seq_lens,
            max_seq_len=common.max_seq_len,
            slot_mapping=common.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )
        metadata.cpu_seq_lens_proof_generation = (
            proof.generation if proof_live else None
        )
        return metadata


class HYV4Int8IndexerBackend(DeepseekV32IndexerBackend):
    # The inherited name preserves native prefill-kernel warmup classification.
    @staticmethod
    def get_builder_cls() -> type[HYV4Int8IndexerMetadataBuilder]:
        return HYV4Int8IndexerMetadataBuilder


class HYV4Int8IndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self) -> type[HYV4Int8IndexerBackend]:
        return HYV4Int8IndexerBackend
