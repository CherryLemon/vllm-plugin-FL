# SPDX-License-Identifier: Apache-2.0
"""Real paged logits/top-k must address pools while returning token indices."""

import pytest
import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)

from vllm_fl.kernels.glm5_next.sparse_attn_indexer_kpool import (
    INDEXER_BACKEND,
    sparse_attn_indexer_kpool,
)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "pool,max_tokens", [(1, 4096), (4, 4096), (4, 4095), (4, 4097)]
)
def test_real_decode_uses_pool_capacity_and_preserves_token_selection(
    monkeypatch, pool, max_tokens
):
    page, dim, heads, batch = 64, 128, 32, 4
    capacity = (max_tokens + pool - 1) // pool
    # Actual K-pool indexer: 512 logical tokens per block, translated into
    # 64-pool kernel pages, including the final partially used logical block.
    pages = (max_tokens + 511) // 512 * (512 // pool // page)
    backing = torch.full((pages, 16896), 193, dtype=torch.uint8, device="cuda")
    cache = torch.as_strided(backing, (pages, page, dim + 4), (16896, dim + 4, 1))
    keys = torch.ones(pages, page, dim, device="cuda").to(torch.float8_e4m3fn)
    scales = (torch.arange(pages * page, device="cuda") + 1).float() / 128
    backing[:, : page * dim].copy_(keys.view(torch.uint8).reshape(pages, -1))
    backing[:, page * dim : page * (dim + 4)].copy_(
        scales.view(pages, page).view(torch.uint8)
    )
    before = backing.clone()
    table = torch.stack(
        [torch.arange(pages, device="cuda").roll(row) for row in range(batch)]
    ).int()
    lengths = torch.tensor(
        [0, 1, 65, max_tokens // pool], dtype=torch.int32, device="cuda"
    )
    tails = [1, 1, 1, max_tokens % pool] if pool > 1 else [0] * batch
    positions = lengths.long() * pool - 1
    positions += torch.tensor(tails, device="cuda")
    q = torch.ones(batch, heads, dim, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.ones(batch, heads, device="cuda")
    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=table,
        seq_lens=lengths[:, None],
        decode_lens=torch.ones(batch, dtype=torch.int32, device="cuda"),
        requires_padding=False,
        schedule_metadata=torch.empty(0, dtype=torch.int32, device="cuda"),
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=lengths,
        max_seq_len=max_tokens,
        slot_mapping=torch.full((batch,), -1, dtype=torch.int64, device="cuda"),
        num_decodes=batch,
        num_decode_tokens=batch,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    topk_tokens = 512 * pool
    width = (topk_tokens + pool - 1 + 127) // 128 * 128
    output = torch.full((batch, width), 123, dtype=torch.int32, device="cuda")
    reference_logits = torch.zeros(batch, capacity, device="cuda")
    expected_indices = []
    for row, length in enumerate(lengths.tolist()):
        page_ids = table[row].long()
        row_keys = keys.float()[page_ids].reshape(-1, dim)[:length]
        row_scales = scales.view(pages, page)[page_ids].reshape(-1)[:length]
        values = (
            torch.relu((q[row].float() @ row_keys.T) * row_scales)
            * weights[row, :, None]
        ).sum(0)
        reference_logits[row, :length] = values
        pools = torch.topk(values, min(512, length)).indices.tolist()
        tokens = [index * pool + offset for index in pools for offset in range(pool)]
        tokens.extend(length * pool + offset for offset in range(tails[row]))
        expected_indices.append(sorted(tokens))

    calls = []
    actual_paged = INDEXER_BACKEND.paged_mqa_logits

    def observe_paged(*args, **kwargs):
        logits = actual_paged(*args, **kwargs)
        assert kwargs["max_model_len"] == capacity
        assert args[4] is table
        assert args[3].data_ptr() == decode.seq_lens.data_ptr()
        assert args[3].shape == decode.seq_lens.shape
        assert logits.shape == (batch, capacity)
        torch.testing.assert_close(logits, reference_logits, atol=2e-5, rtol=2e-5)
        calls.append(logits)
        return logits

    # A forwarding observer around the real public resolver/kernel; no
    # numerical, dependency, metadata or top-k implementation is replaced.
    monkeypatch.setattr(INDEXER_BACKEND, "paged_mqa_logits", observe_paged)
    with set_forward_context({"indexer": metadata}, VllmConfig()):
        result = sparse_attn_indexer_kpool(
            torch.empty(batch, 1, device="cuda"),
            "indexer",
            cache,
            q,
            None,
            torch.empty(batch, dim, device="cuda"),
            weights,
            128,
            None,
            topk_tokens,
            dim,
            max_tokens,
            batch,
            output,
            True,
            index_kpool=pool,
            positions=positions,
        )
    assert len(calls) == 1
    assert result is output
    for row in range(batch):
        actual = result[row][result[row] >= 0].tolist()
        assert sorted(actual) == expected_indices[row]
    assert torch.equal(backing, before)
