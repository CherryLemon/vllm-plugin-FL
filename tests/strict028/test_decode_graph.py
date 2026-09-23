# SPDX-License-Identifier: Apache-2.0
"""Replay must read and write the selected request page, not capture its address."""

import pytest
import torch

from vllm_fl.strict028.decode_graph import DecodeGraphs


class _Pages:
    def __init__(self):
        self.storage = torch.zeros((2, 16), dtype=torch.uint8, device="cuda")
        self.graph_scratch = torch.empty_like(self.storage[0])
        self.page_bytes = 16
        self.active_block = None
        self.active = None

    def bind(self, block):
        self.active_block = block
        self.active = self.storage[block].view(torch.int64)

    def bind_graph_scratch(self):
        self.active_block = None
        self.active = self.graph_scratch.view(torch.int64)


class _Model:
    speculative_config = None

    def __init__(self, pages):
        self.pages = pages

    def forward_with_aux(self, token, *, start_pos):
        self.pages.active[0].add_(token[0] + start_pos)
        return self.pages.active[:1].float().clone(), None


class _DraftModel(_Model):
    speculative_config = object()

    def forward_with_aux(self, token, *, start_pos):
        logits, _ = super().forward_with_aux(token, start_pos=start_pos)
        return logits, logits.clone()

    def store_draft_context(self, hidden, position):
        self.pages.active[1].add_(hidden[0].long())

    def propose_draft(self, token, hidden, position):
        self.pages.active[1].add_(token[0] + hidden[0].long() + position)
        value = self.pages.active[1].clone()
        return value.view(1, 1), value.float(), value.float()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graphs_replay_across_positions_and_request_pages():
    pages = _Pages()
    model = _Model(pages)
    graphs = DecodeGraphs(model, pages, torch.device("cuda"))
    expected = [0, 0]
    for block, position, token in (
        (0, 16, 2),
        (1, 17, 3),
        (0, 17, 4),
        (1, 16, 5),
        (0, 16, 6),
    ):
        pages.bind(block)
        logits, _ = graphs.target(
            torch.tensor([token], device="cuda", dtype=torch.int64), position
        )
        expected[block] += position + token
        assert logits.item() == expected[block]
        assert pages.storage[block].view(torch.int64)[0].item() == expected[block]
        assert pages.storage[1 - block].view(torch.int64)[0].item() == expected[1 - block]
    assert graphs.stats()["target_graphs"] == 2
    assert graphs.stats()["target_replays"] == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_target_and_draft_graphs_share_pool_without_cross_request_state():
    pages = _Pages()
    graphs = DecodeGraphs(_DraftModel(pages), pages, torch.device("cuda"))
    expected = [[0, 0], [0, 0]]
    for block, position, token in (
        (0, 16, 2), (1, 17, 3), (0, 17, 4), (1, 16, 5)
    ):
        pages.bind(block)
        logits, hidden = graphs.target(
            torch.tensor([token], device="cuda", dtype=torch.int64), position
        )
        expected[block][0] += token + position
        expected[block][1] += expected[block][0]
        assert logits.item() == expected[block][0]
        result = graphs.draft(
            torch.tensor([token + 1], device="cuda", dtype=torch.int64),
            hidden,
            position,
        )
        expected[block][1] += token + 1 + expected[block][0] + position
        assert result[0].item() == expected[block][1]
        assert pages.storage[block].view(torch.int64).tolist() == expected[block]
        assert pages.storage[1 - block].view(torch.int64).tolist() == expected[1 - block]
    assert graphs.stats()["target_graphs"] == 2
    assert graphs.stats()["draft_graphs"] == 2
    assert graphs.stats()["target_replays"] == 4
    assert graphs.stats()["draft_replays"] == 4
