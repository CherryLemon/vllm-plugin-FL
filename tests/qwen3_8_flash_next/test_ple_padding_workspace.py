"""CPU checks for the PLE padding workspace.

Graph padding can hand the PLE layer token rows that belong to no real
request.  Those rows must never be written into a real request's storage, and
the real tokens' hash ids must be independent of how many padding rows the
runner added.
"""

from __future__ import annotations

import pytest
import torch

from vllm_fl.models.qwen3_8_flash_next.config import Qwen3_8FlashNextTextConfig
from vllm_fl.models.qwen3_8_flash_next.gpu import ple_layer


class _HashEchoEmbedding(torch.nn.Module):
    """Echoes hash ids so output differences track hash-id differences."""

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        head_dim = self.embedding_dim // ids.shape[-1]
        return ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, head_dim)


@pytest.fixture()
def build(monkeypatch):
    monkeypatch.setattr(ple_layer, "VocabParallelEmbedding", _HashEchoEmbedding)

    def _build(max_num_reqs: int = 8, max_total_tokens: int = 64):
        config = Qwen3_8FlashNextTextConfig(
            hc_count=4,
            hc_lowrank=8,
            ple_layer_ids=[1],
            ple_embed_dim=8,
            ple_conv_kernel_size=3,
            ngram_size=3,
            heads_per_ngram=2,
            ngram_vocab_size_base=97,
            make_ngram_vocab_size_divisible_by=8,
            output_gate_type="none",
            rope_parameters={"rope_theta": 10000.0},
            layer_types=["full_attention"],
            num_hidden_layers=1,
            vocab_size=256,
            eos_token_id=0,
            split_ngram_parts=8,
            seed=1234,
        )
        module = ple_layer.Qwen3_8FlashNextNGramEmbedding(
            config,
            embedding_dim=8,
            ple_dense_layer_id=0,
            max_total_tokens=max_total_tokens,
            max_num_reqs=max_num_reqs,
            prefix=".ple",
        )
        return module.eval()

    return _build


def _expected_rows(
    ids: torch.Tensor, query_start_loc: torch.Tensor, eos: int
) -> list[torch.Tensor]:
    rows = []
    for req in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[req])
        end = int(query_start_loc[req + 1])
        rows.append(ids.new_full((end - start,), eos) if end <= start else ids[start:end])
    return rows


def _real_case():
    # lengths 3 / 0 (empty request row) / 4
    ids = torch.tensor([5, 6, 7, 11, 12, 13, 14], dtype=torch.long)
    query_start_loc = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
    context = torch.tensor([[0, 0], [9, 9], [4, 8]], dtype=torch.long)
    return ids, query_start_loc, context


def test_workspace_reserves_a_sink_row(build):
    module = build(max_num_reqs=4)
    assert module.max_num_reqs == 4
    assert module.padded_buffer.shape[0] == 5


@pytest.mark.parametrize("num_padded_reqs", [4, 8])
def test_padding_does_not_change_real_token_outputs(build, num_padded_reqs):
    module = build(max_num_reqs=8)
    ids, query_start_loc, context = _real_case()
    num_real_reqs = query_start_loc.numel() - 1

    baseline = module(ids, query_start_loc, context)

    padded_qsl = torch.cat(
        [
            query_start_loc,
            query_start_loc.new_full((num_padded_reqs - num_real_reqs,), 7),
        ]
    )
    padded_context = torch.cat(
        [
            context,
            context.new_full((num_padded_reqs - num_real_reqs, context.shape[1]), 0),
        ]
    )
    padded = module(ids, padded_qsl, padded_context)
    assert torch.equal(baseline, padded[: ids.numel()])


def test_graph_padded_tokens_land_in_the_sink_row(build):
    module = build(max_num_reqs=8)
    ids, query_start_loc, _ = _real_case()
    # Three trailing token rows belong to no request: the runner padded the
    # token dimension up to a graph bucket.
    padded_ids = torch.cat([ids, torch.tensor([0, 0, 0], dtype=torch.long)])
    graph_qsl = torch.tensor([0, 3, 3, 7, 7], dtype=torch.int32)
    context = torch.zeros(4, 2, dtype=torch.long)

    module(padded_ids, graph_qsl, context)

    num_reqs = graph_qsl.numel() - 1
    # Request 0 owns positions 0..2; a stray write would land on [0, 0].
    assert [int(v) for v in module.padded_buffer[0, :3]] == [5, 6, 7]
    assert int(module.padded_buffer[0, 0]) == 5
    # The invalid rows are parked in the extra workspace row.
    assert [int(v) for v in module.padded_buffer[num_reqs, 7:10]] == [0, 0, 0]


def test_real_rows_match_independent_packing(build):
    module = build(max_num_reqs=8)
    ids, query_start_loc, context = _real_case()
    module(ids, query_start_loc, context)

    for req, expected in enumerate(_expected_rows(ids, query_start_loc, 0)):
        actual = module.padded_buffer[req, : expected.numel()]
        assert torch.equal(actual, expected)


def test_zero_requests_returns_empty_output(build):
    module = build(max_num_reqs=4)
    empty_ids = torch.empty(0, dtype=torch.long)
    out = module(empty_ids, torch.tensor([0], dtype=torch.int32), torch.empty(0, 2))
    assert out.shape == (0, module.embedding_dim)


def test_request_count_beyond_workspace_fails(build):
    module = build(max_num_reqs=4)
    ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    query_start_loc = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    with pytest.raises(ValueError, match="at most 4"):
        module(ids, query_start_loc, torch.zeros(5, 2, dtype=torch.long))


def test_workspace_reuse_does_not_leak_between_calls(build):
    module = build(max_num_reqs=8)
    ids, query_start_loc, context = _real_case()
    graph_qsl = torch.tensor([0, 3, 3, 7, 7], dtype=torch.int32)
    padded_ids = torch.cat([ids, torch.tensor([21, 22, 23], dtype=torch.long)])
    module(padded_ids, graph_qsl, torch.zeros(4, 2, dtype=torch.long))

    reused = module(ids, query_start_loc, context)
    fresh = build(max_num_reqs=8)(ids, query_start_loc, context)
    assert torch.equal(reused, fresh)
