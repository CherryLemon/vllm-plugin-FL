# SPDX-License-Identifier: Apache-2.0
"""The real 0.28 output contract with a tiny deterministic target/draft oracle."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm_fl.strict028.worker import ModelRunnerFL028, Request


class Target:
    speculative_config = object()
    args = NS(max_seq_len=64)

    def __init__(self):
        self.inputs, self.commits = [], []

    def forward_with_aux(self, ids, *, start_pos):
        self.inputs.append((start_pos, ids.tolist()))
        # Absolute-position oracle: target output token at position p+1 is p+11.
        logits = torch.zeros(1, 128)
        logits[0, start_pos + len(ids) + 10] = 1
        return logits, torch.tensor([start_pos])

    def compute_logits(self, logits):
        return logits

    def store_draft_context(self, hidden, position):
        self.commits.append(position)

    def propose_draft(self, token, hidden, position):
        ids = torch.arange(int(token[0]), int(token[0]) + 6).view(1, 6)
        return ids, torch.ones(1, 5, 128), torch.ones(1, 5)


def scheduled(drafts, start=3):
    return NS(
        finished_req_ids=set(),
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={"a": drafts} if drafts else {},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=NS(
            req_ids=["a"],
            all_token_ids={},
            new_block_ids=[None],
            resumed_req_ids=set(),
            num_computed_tokens=[start],
        ),
        num_scheduled_tokens={"a": 1 + len(drafts)},
    )


@pytest.mark.parametrize("accepted", range(6))
def test_only_accepted_prefix_is_committed_and_scheduler_gets_bonus(accepted):
    target = Target()
    runner = ModelRunnerFL028(target, NS(bind=lambda *a, **kw: None), "cpu")
    runner.requests["a"] = Request([10, 11, 12, 13], 1, 3)
    drafts = [14, 15, 16, 17, 18]
    if accepted < 5:
        drafts[accepted] = 99
    output = runner.execute_model(scheduled(drafts))
    assert output.sampled_token_ids == [list(range(14, 15 + accepted))]
    assert target.commits == list(range(3, 4 + accepted))
    assert runner.requests["a"].computed == 4 + accepted
    assert runner.requests["a"].tokens[-1] == 14 + accepted
    assert all(99 not in ids for _, ids in target.inputs)
    assert runner.spec_stats["accepted_tokens"] == accepted
    # The host subtracts (drafts - accepted) from its optimistic computed span.
    assert 3 + 6 - (5 - accepted) == runner.requests["a"].computed
    proposed = runner.take_draft_token_ids()
    assert proposed.req_ids == ["a"]
    assert proposed.draft_token_ids == [list(range(15 + accepted, 20 + accepted))]
    assert runner.take_draft_token_ids() is None


def test_context_limit_disables_future_draft_queries():
    target = Target()
    target.args = NS(max_seq_len=8)
    runner = ModelRunnerFL028(target, NS(bind=lambda *a, **kw: None), "cpu")
    runner.requests["a"] = Request([10, 11, 12, 13], 1, 3)
    runner.execute_model(scheduled([]))
    assert runner.take_draft_token_ids().draft_token_ids == [[]]


def test_truncated_scheduler_draft_span_and_next_step():
    target = Target()
    runner = ModelRunnerFL028(target, NS(bind=lambda *a, **kw: None), "cpu")
    runner.requests["a"] = Request([10, 11, 12, 13], 1, 3)
    assert runner.execute_model(scheduled([14, 15])).sampled_token_ids == [[14, 15, 16]]
    assert runner.execute_model(scheduled([], start=6)).sampled_token_ids == [[17]]
    assert target.commits == [3, 4, 5, 6]


class BatchedOracle:
    def __init__(self):
        self.commits = {}
        self.calls = []

    def target_batch(self, tokens, pages, positions, active):
        self.calls.append((pages.tolist(), positions.tolist(), active.tolist()))
        logits = torch.zeros(len(tokens), 128)
        for i, (page, position, enabled) in enumerate(
            zip(pages.tolist(), positions.tolist(), active.tolist())
        ):
            if enabled:
                self.commits.setdefault(page, []).append((position, int(tokens[i])))
                logits[i, position + 11] = 1
        return logits, positions[:, None, None].float()

    def draft_batch(self, tokens, hidden, pages, positions, active):
        torch.testing.assert_close(hidden[active, 0, 0], positions[active].float())
        ids = tokens[:, None] + torch.arange(6)
        return ids, torch.ones(len(tokens), 5, 128), torch.ones(len(tokens), 5)


def test_batched_verification_masks_each_rejected_prefix_independently(monkeypatch):
    monkeypatch.setenv("VLLM_FL_DECODE_GRAPH", "1")
    monkeypatch.setenv("VLLM_FL_BATCHED_DECODE", "1")
    runner = ModelRunnerFL028(Target(), NS(), "cpu")
    runner.graphs = BatchedOracle()
    req_ids = [f"r{i}" for i in range(6)]
    output = scheduled([])
    output.scheduled_cached_reqs.req_ids = []
    output.num_scheduled_tokens = {}
    output.scheduled_spec_decode_tokens = {}
    for i, req_id in enumerate(req_ids):
        start = 3 + i
        runner.requests[req_id] = Request(list(range(10, start + 11)), i + 1, start)
        draft = list(range(start + 11, start + 16))
        if i < 5:
            draft[i] = 99
        output.num_scheduled_tokens[req_id] = 6
        output.scheduled_spec_decode_tokens[req_id] = draft
    result = runner.execute_model(output)
    assert len(runner.graphs.calls) == 6
    for i, req_id in enumerate(req_ids):
        start = 3 + i
        assert result.sampled_token_ids[i] == list(range(start + 11, start + 12 + i))
        assert runner.requests[req_id].computed == start + i + 1
        assert runner.graphs.commits[i + 1] == [
            (p, p + 10) for p in range(start, start + i + 1)
        ]
    assert runner.spec_stats["accepted_prefix_histogram"] == [1] * 6
    assert runner.spec_stats["target_forward_calls"] == 21
    proposals = runner.take_draft_token_ids()
    assert proposals.req_ids == req_ids
    assert all(len(ids) == 5 for ids in proposals.draft_token_ids)


@pytest.mark.parametrize("idle", [False, True])
def test_dp_rejection_or_idle_lane_keeps_remote_verification_aligned(monkeypatch, idle):
    monkeypatch.setenv("VLLM_FL_DECODE_GRAPH", "1")
    monkeypatch.setenv("VLLM_FL_BATCHED_DECODE", "1")
    runner = ModelRunnerFL028(Target(), NS(), "cpu")
    runner.data_parallel = True
    runner.graphs = BatchedOracle()
    output = NS(num_scheduled_tokens={}, scheduled_spec_decode_tokens={})
    if not idle:
        runner.requests["a"] = Request([10, 11, 12, 13], 1, 3)
        output = scheduled([99, 99, 99, 99, 99])
    local_activity = []
    remote_activity = iter([True, True, True, False])

    def control_reduce(flag, op):
        assert flag.device.type == "cpu"
        local_activity.append(bool(flag))
        flag.fill_(next(remote_activity))

    monkeypatch.setattr("vllm_fl.strict028.worker.dist.all_reduce", control_reduce)
    result = runner.execute_decode_batch(output)
    assert len(runner.graphs.calls) == 3
    assert local_activity == ([False] * 4 if idle else [True, False, False, False])
    assert result.sampled_token_ids == ([] if idle else [[14]])
    assert runner.graphs.commits == ({} if idle else {1: [(3, 13)]})
    assert runner.spec_stats["target_forward_calls"] == (0 if idle else 1)
