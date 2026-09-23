# SPDX-License-Identifier: Apache-2.0
"""Bounded causal Prefill over the same complete request-state layout.

Each chunk can cross a ring or compression boundary. Attention reads previous
state plus the current chunk before committing the new ring. Future compressed
positions are masked separately for each query.
"""

import torch

from .collectives import all_reduce_
from .models.deepseek_v41 import model as ref
from .models.deepseek_v41.ops import act_quant, fp4_act_quant, grouped_output_projection


class ChunkPrefill:
    def __init__(self, start, count, prompt_length, device):
        if not 0 <= start < start + count <= prompt_length:
            raise ValueError("invalid Prefill chunk bounds")
        self.start, self.count, self.prompt_length = start, count, prompt_length
        self.end = start + count
        self.positions = torch.arange(start, self.end, device=device)[None]
        self.pages = torch.zeros(1, dtype=torch.long, device=device)
        self.compressed = self.index_keys = self.topk = self.candidates = None

    def hash(self, module, input_ids, token_mask=None):
        return module(input_ids, self.start, token_mask)

    def commit_ring(self, cache, values):
        width = cache.shape[1]
        count = min(values.shape[1], width)
        slots = torch.arange(self.end - count, self.end, device=values.device) % width
        cache.index_copy_(1, slots, values[:, -count:])

    def compress(self, module, x):
        ratio = module.compress_ratio
        if ratio == 1:
            return module.norm(module.wkv(x))
        kv, score = module.wkv(x.float()), module.wgate(x.float())
        pending = self.start % ratio
        if pending:
            kv = torch.cat([module.kv_state[:, :pending], kv], 1)
            score = torch.cat([module.score_state[:, :pending], score], 1)
        complete = kv.shape[1] // ratio * ratio
        tail_kv, tail_score = kv[:, complete:].clone(), score[:, complete:].clone()
        # Canonical incomplete-group state, matching a fresh full Prefill.
        module.kv_state.zero_()
        module.score_state.fill_(-torch.inf)
        if tail_kv.shape[1]:
            module.kv_state[:, : tail_kv.shape[1]].copy_(tail_kv)
            module.score_state[:, : tail_score.shape[1]].copy_(tail_score)
        if not complete:
            return None
        grouped_kv = kv[:, :complete].unflatten(1, (-1, ratio))
        grouped_score = score[:, :complete].unflatten(1, (-1, ratio))
        latent = (grouped_kv * grouped_score.softmax(2)).sum(2)
        return module.norm(latent.to(x.dtype))

    def compressed_frequencies(self, attn, count):
        ratio = attn.compress_ratio
        begin = self.start // ratio * ratio
        return attn.freqs_cis[begin : begin + count * ratio : ratio]

    def index(self, indexer, attn, x, qr, latent, offset):
        from flag_gems.fused.dsv41_decode_state import paged_index_scores

        ratio, rd = indexer.compress_ratio, indexer.rope_head_dim
        length = self.end // ratio
        if indexer.owns_k:
            if latent is not None:
                key = indexer.k_norm(indexer.wk(latent))
                ref.apply_rotary_emb(
                    key[..., -rd:], self.compressed_frequencies(attn, key.shape[1])
                )
                fp4_act_quant(key, inplace=True)
                indexer.k_cache[:, self.start // ratio : length].copy_(key)
            self.index_keys = indexer.k_cache
        if self.index_keys is None:
            raise RuntimeError("index source must publish its state before consumers")
        q = indexer.wq_b(qr).unflatten(
            -1, (indexer.n_local_heads, indexer.index_head_dim)
        )
        ref.apply_rotary_emb(q[..., -rd:], attn.freqs_cis[self.start : self.end])
        fp4_act_quant(q, inplace=True)
        weights = indexer.weights_proj(x) * (
            indexer.softmax_scale * indexer.n_heads**-0.5
        )
        visible = (self.positions + 1) // ratio
        scores = paged_index_scores(
            q, self.index_keys[:, :length], weights, self.pages, visible
        )
        if ref.world_size > 1 and length:
            all_reduce_(scores)
        if length:
            if indexer.is_candidate_source:
                self.candidates = ref.select_candidate_blocks(
                    scores,
                    visible.unsqueeze(-1),
                    indexer.candidate_topk_blocks,
                    indexer.candidate_block_size,
                )
            elif indexer.uses_candidates:
                scores.masked_fill_(~self.candidates, -torch.inf)
            ids = (
                scores.topk(min(indexer.index_topk, length), -1, sorted=False)
                .indices.sort(-1)
                .values
            )
            ids = torch.where(ids < visible.unsqueeze(-1), ids + offset, -1).int()
        else:
            ids = torch.empty(1, self.count, 0, dtype=torch.int32, device=x.device)
        # Keep the whole-prompt reference's window/compressed tile boundary.
        target_width = min(indexer.index_topk, self.prompt_length // ratio)
        return torch.nn.functional.pad(ids, (0, target_width - ids.shape[-1]), value=-1)

    def attention(self, attn, x):
        from flag_gems.fused.dsv41_reference_ops import paged_sparse_attention_with_sink

        if x.shape[:2] != (1, self.count):
            raise ValueError(
                "chunked Prefill expects one request with the declared chunk length"
            )
        rd, win = attn.rope_head_dim, attn.window_size
        frequencies = attn.freqs_cis[self.start : self.end]
        qr = attn.q_norm(attn.wq_a(x))
        q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
        ref.apply_rotary_emb(q[..., -rd:], frequencies)
        kv = attn.kv_norm(attn.wkv(x))
        ref.apply_rotary_emb(kv[..., -rd:], frequencies)
        act_quant(kv, inplace=True)
        history = min(self.start, win - 1)
        history_begin = self.start - history
        if history:
            slots = torch.arange(history_begin, self.start, device=x.device) % win
            window = torch.cat([attn.window_kv_cache[:, slots], kv], 1)
        else:
            window = kv
        oldest = (self.positions - win + 1).clamp_min(0)
        absolute = oldest.unsqueeze(-1) + torch.arange(
            min(win, self.prompt_length), device=x.device
        )
        ids = torch.where(
            absolute <= self.positions.unsqueeze(-1), absolute - history_begin, -1
        ).int()
        offset = window.shape[1]
        compressed = None
        if attn.compress_ratio:
            ratio, latent = attn.compress_ratio, None
            if attn.is_kv_source:
                latent = self.compress(attn.compressor, x)
                self.compressed = attn.compress_kv_cache
            if attn.is_index_source:
                self.topk = self.index(attn.indexer, attn, x, qr, latent, offset)
            if latent is not None:
                ref.apply_rotary_emb(
                    latent[..., -rd:],
                    self.compressed_frequencies(attn, latent.shape[1]),
                )
                fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
                self.compressed[:, self.start // ratio : self.end // ratio].copy_(
                    latent
                )
            compressed = self.compressed
            ids = torch.cat([ids, self.topk], -1)
        out = paged_sparse_attention_with_sink(
            q,
            window,
            compressed,
            attn.attn_sink,
            ids.contiguous(),
            self.pages,
            attn.softmax_scale,
        )
        self.commit_ring(attn.window_kv_cache, kv)
        ref.apply_rotary_emb(out[..., -rd:], frequencies, inverse=True)
        out = out.view(1, self.count, attn.n_local_groups, -1)
        weight = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
        return attn.wo_b(
            prefill_grouped_projection(out, weight, self.prompt_length).flatten(2)
        )

    def store_draft_context(self, attn, hidden):
        kv = attn.kv_norm(attn.wkv(hidden))
        ref.apply_rotary_emb(
            kv[..., -attn.rope_head_dim :], attn.freqs_cis[self.start : self.end]
        )
        act_quant(kv, inplace=True)
        self.commit_ring(attn.window_kv_cache, kv)


def prefill_grouped_projection(x, weight, prompt_length):
    """Keep small chunks on the complete prefix's BF16 GEMM geometry.

    On H100, M=32/128 and M>=256 select different accumulation paths. Real
    wo_a weights expose BF16 rounding ties, which amplify in later layers.
    Zero rows do not contribute to other outputs. Long chunks already use
    the large-prefix path, so this adds at most 255 padding rows.
    """
    count = x.shape[1]
    rows = min(prompt_length, 256)
    if count < rows:
        x = torch.cat([x, x.new_zeros((x.shape[0], rows - count, *x.shape[2:]))], dim=1)
    return grouped_output_projection(x, weight)[:, :count]
