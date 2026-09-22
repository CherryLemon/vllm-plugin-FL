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
