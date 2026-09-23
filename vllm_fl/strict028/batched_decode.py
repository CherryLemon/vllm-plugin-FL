# SPDX-License-Identifier: Apache-2.0
"""Device-position decode over stable, scheduler-owned request-state pages.

This composes the same model weights and numerical operations as the serial
reference. Large KV/index pools stay in place; only selected sparse rows are
read by FlagGems. Metadata can change without changing the captured graph.
"""

import torch

from .collectives import all_reduce_
from .models.deepseek_v41 import model as ref
from .models.deepseek_v41.ops import act_quant, fp4_act_quant, grouped_output_projection


def rotary(x, frequencies, inverse=False):
    """Reference complex RoPE with independent positions for every request."""
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        frequencies = frequencies.conj()
    if z.ndim == 4:
        frequencies = frequencies.unsqueeze(2)
    x.copy_(torch.view_as_real(z * frequencies).flatten(-2))
    return x


class DecodeBatch:
    def __init__(self, state, pages, positions, active):
        if (
            pages.ndim != 1
            or pages.dtype != torch.int64
            or positions.shape != (pages.numel(), 1)
            or positions.dtype != torch.int64
            or active.shape != positions.shape
            or active.dtype != torch.bool
        ):
            raise ValueError("expected pages[B], positions[B,1], active[B,1]")
        self.state = state
        self.pages = pages
        self.positions = positions
        self.active = active
        self.compressed = None
        self.index_keys = None
        self.topk = None
        self.candidates = None

    def pool(self, module, name):
        return self.state.pool(module, name)

    def write(self, pool, values, positions, active=None):
        from flag_gems.fused.dsv41_decode_state import write_request_rows

        write_request_rows(
            pool,
            values.contiguous(),
            self.pages,
            positions.contiguous(),
            self.active if active is None else active.contiguous(),
        )

    def hash(self, module, input_ids, token_mask=None):
        compressed = module.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, module.DEAD)
        pool = self.pool(module, "cache")
        self.write(pool.unsqueeze(-1), compressed.unsqueeze(-1), self.positions)
        tokens, blocked = [], torch.zeros_like(self.positions, dtype=torch.bool)
        for shift in range(module.layout.max_ngram_size):
            source = pool[self.pages[:, None], (self.positions - shift).clamp_min(0)]
            blocked = blocked | (self.positions < shift) | (source == module.DEAD)
            tokens.append(torch.where(blocked, module.pad_id, source))
        tokens = torch.stack(tokens, -1)
        products = tokens.unsqueeze(2) * module.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, module.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % module.primes[:, i - 1])
        return torch.cat(hashes, -1) + module.offsets

    def window(self, attn, x, frequencies):
        kv = attn.kv_norm(attn.wkv(x))
        rotary(kv[..., -attn.rope_head_dim :], frequencies)
        act_quant(kv, inplace=True)
        pool = self.pool(attn, "window_kv_cache")
        self.write(pool, kv, self.positions % attn.window_size)
        # Preserve the serial ring's oldest-first ordering, including -1 holes.
        ids = (
            (self.positions % attn.window_size + 1).unsqueeze(-1)
            + torch.arange(attn.window_size, device=x.device)
        ) % attn.window_size
        ids = torch.where(ids <= self.positions.unsqueeze(-1), ids, -1).int()
        return pool, ids

    def compress(self, module, x):
        ratio = module.compress_ratio
        if ratio == 1:
            return module.norm(module.wkv(x))
        kv, score = module.wkv(x.float()), module.wgate(x.float())
        kv_pool, score_pool = (
            self.pool(module, "kv_state"),
            self.pool(module, "score_state"),
        )
        self.write(kv_pool, kv, self.positions % ratio)
        self.write(score_pool, score, self.positions % ratio)
        # The small incomplete group is gathered; the context-length pools are not.
        kv, score = kv_pool[self.pages], score_pool[self.pages]
        latent = (kv * score.softmax(dim=1)).sum(dim=1, keepdim=True)
        return module.norm(latent.to(x.dtype))

    def index(self, indexer, x, qr, latent, frequencies, offset):
        from flag_gems.fused.dsv41_decode_state import paged_index_scores

        ratio, rd = indexer.compress_ratio, indexer.rope_head_dim
        lengths = (self.positions + 1) // ratio
        publish = self.active & ((self.positions + 1) % ratio == 0)
        if indexer.owns_k:
            k = indexer.k_norm(indexer.wk(latent))
            rotary(k[..., -rd:], frequencies[(self.positions + 1 - ratio).clamp_min(0)])
            fp4_act_quant(k, inplace=True)
            self.index_keys = self.pool(indexer, "k_cache")
            self.write(self.index_keys, k, self.positions // ratio, publish)
        if self.index_keys is None:
            raise RuntimeError("index source must publish its state before consumers")
        q = indexer.wq_b(qr).unflatten(
            -1, (indexer.n_local_heads, indexer.index_head_dim)
        )
        rotary(q[..., -rd:], frequencies[self.positions])
        fp4_act_quant(q, inplace=True)
        weights = indexer.weights_proj(x) * (
            indexer.softmax_scale * indexer.n_heads**-0.5
        )
        scores = paged_index_scores(q, self.index_keys, weights, self.pages, lengths)
        if ref.world_size > 1:
            all_reduce_(scores)
        if indexer.is_candidate_source:
            self.candidates = ref.select_candidate_blocks(
                scores,
                lengths.unsqueeze(-1),
                indexer.candidate_topk_blocks,
                indexer.candidate_block_size,
            )
        elif indexer.uses_candidates:
            scores.masked_fill_(~self.candidates, -torch.inf)
        k = min(indexer.index_topk, scores.shape[-1])
        ids = scores.topk(k, -1, sorted=False).indices.sort(-1).values
        return torch.where(ids < lengths.unsqueeze(-1), ids + offset, -1).int()

    def attention(self, attn, x):
        from flag_gems.fused.dsv41_reference_ops import paged_sparse_attention_with_sink

        b, s, _ = x.shape
        rd = attn.rope_head_dim
        frequencies = attn.freqs_cis[self.positions]
        qr = attn.q_norm(attn.wq_a(x))
        q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
        rotary(q[..., -rd:], frequencies)
        window, ids = self.window(attn, x, frequencies)
        compressed = None
        if attn.compress_ratio:
            ratio, latent = attn.compress_ratio, None
            if attn.is_kv_source:
                latent = self.compress(attn.compressor, x)
                self.compressed = self.pool(attn, "compress_kv_cache")
            if attn.is_index_source:
                self.topk = self.index(
                    attn.indexer, x, qr, latent, attn.freqs_cis, attn.window_size
                )
            if latent is not None:
                rotary(
                    latent[..., -rd:],
                    attn.freqs_cis[(self.positions + 1 - ratio).clamp_min(0)],
                )
                fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
                self.write(
                    self.compressed,
                    latent,
                    self.positions // ratio,
                    self.active & ((self.positions + 1) % ratio == 0),
                )
            compressed = self.compressed
            ids = torch.cat([ids, self.topk], -1)
        ids = torch.where(self.active.unsqueeze(-1), ids, -1).contiguous()
        out = paged_sparse_attention_with_sink(
            q, window, compressed, attn.attn_sink, ids, self.pages, attn.softmax_scale
        )
        rotary(out[..., -rd:], frequencies, inverse=True)
        out = out.view(b, s, attn.n_local_groups, -1)
        weight = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
        return attn.wo_b(grouped_output_projection(out, weight).flatten(2))

    def store_draft_context(self, attn, hidden):
        frequencies = attn.freqs_cis[self.positions]
        kv = attn.kv_norm(attn.wkv(hidden))
        rotary(kv[..., -attn.rope_head_dim :], frequencies)
        act_quant(kv, inplace=True)
        self.write(
            self.pool(attn, "window_kv_cache"), kv, self.positions % attn.window_size
        )

    def draft_attention(self, attn, x, hidden):
        self.store_draft_context(attn, hidden)
        b, block, _ = x.shape
        win, rd = attn.window_size, attn.rope_head_dim
        positions = self.positions + 1 + torch.arange(block, device=x.device)
        frequencies = attn.freqs_cis[positions]
        qr = attn.q_norm(attn.wq_a(x))
        q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
        rotary(q[..., -rd:], frequencies)
        kv = attn.kv_norm(attn.wkv(x))
        rotary(kv[..., -rd:], frequencies)
        act_quant(kv, inplace=True)
        kv = torch.cat([self.pool(attn, "window_kv_cache")[self.pages], kv], 1)
        # Compact the visible window and draft IDs into the same order as the
        # serial path; trailing -1 slots do not shift the softmax tile boundary.
        length = (self.positions + 1).clamp_max(win)
        slots = torch.arange(win + block, device=x.device).view(1, -1)
        ids = torch.where(slots < length, slots, win + slots - length)
        ids = torch.where((slots < length + block) & self.active, ids, -1)
        ids = ids.int().unsqueeze(1).expand(b, block, -1).contiguous()
        out = ref.sparse_attn(q, kv, attn.attn_sink, ids, attn.softmax_scale)
        rotary(out[..., -rd:], frequencies, inverse=True)
        out = out.view(b, block, attn.n_local_groups, -1)
        weight = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
        return attn.wo_b(grouped_output_projection(out, weight).flatten(2))
