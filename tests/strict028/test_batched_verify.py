# SPDX-License-Identifier: Apache-2.0
"""Real model state admission for multi-position target verification."""

from types import SimpleNamespace

import pytest
import torch
from test_batched_decode import make_model, serial_target

from vllm_fl.strict028.batched_graph import BatchedDecodeGraphs
from vllm_fl.strict028.models.deepseek_v41.model import set_dtype
from vllm_fl.strict028.worker import ModelRunnerFL028, Request

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.inference_mode()
@pytest.mark.parametrize("verify_width", [3, 6])
def test_verify_graph_commits_only_accepted_prefix_and_reuses_metadata(
    monkeypatch, verify_width
):
    model, state = make_model()
    graph = BatchedDecodeGraphs(
        model, state, torch.device("cuda"), batch_capacity=4, verify_width=verify_width
    )
    for page, length in ((1, 7), (2, 8), (3, 127)):
        state.bind(page, reset=True)
        with torch.device("cuda"), set_dtype(torch.bfloat16):
            _, _, hidden = model.core(torch.arange(length, device="cuda")[None] % 64, 0)
            model.core.store_spec_context(hidden, 0)
    initial = state.storage.clone()
    for order, starts, lengths, commits in (
        ([1, 2, 3], [7, 8, 127], [6, 6, 6], [1, 3, 6]),
        ([3, 1, 2], [127, 7, 8], [6, 6, 6], [6, 2, 4]),
        ([2, 3, 1], [8, 127, 7], [1, 0, 6], [1, 0, 1]),
    ):
        pages = torch.tensor(order, device="cuda")
        positions = torch.tensor(starts, device="cuda")[:, None] + torch.arange(
            6, device="cuda"
        )
        active = (
            torch.arange(6, device="cuda")[None]
            < torch.tensor(lengths, device="cuda")[:, None]
        )
        tokens = torch.zeros((3, 6), device="cuda", dtype=torch.long)
        oracle_ids, oracle_hidden = [], []
        # Construct greedy candidates from the serial reference, not random KV.
        for row in range(3):
            state.storage.copy_(initial)
            token = torch.tensor([11 + row], device="cuda")
            row_ids, row_hidden = [], []
            for offset in range(6):
                tokens[row, offset] = token[0]
                logits, hidden = serial_target(
                    model,
                    state,
                    token,
                    pages[row : row + 1],
                    positions[row : row + 1, offset],
                    torch.ones(1, device="cuda", dtype=torch.bool),
                )
                token = logits.argmax(-1)
                row_ids.append(token.clone())
                row_hidden.append(hidden.clone())
            oracle_ids.append(torch.cat(row_ids))
            oracle_hidden.append(row_hidden)
        for row, keep in enumerate(commits):
            if 0 < keep < lengths[row]:
                tokens[row, keep] = (tokens[row, keep] + 1) % 64

        state.storage.copy_(initial)
        for row, keep in enumerate(commits):
            for offset in range(keep):
                serial_target(
                    model,
                    state,
                    tokens[row : row + 1, offset],
                    pages[row : row + 1],
                    positions[row : row + 1, offset],
                    torch.ones(1, device="cuda", dtype=torch.bool),
                )
        expected = state.storage.clone()
        state.storage.copy_(initial)
        selected, kept, hidden, finite = graph.verify_batch(
            tokens, pages, positions, active
        )
        assert finite.item()
        assert kept.tolist() == commits
        for row, keep in enumerate(commits):
            if keep:
                torch.testing.assert_close(
                    selected[row, :keep], oracle_ids[row][:keep], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    hidden[row : row + 1], oracle_hidden[row][keep - 1], rtol=0, atol=0
                )
        for field in state.fields:
            actual_field = (
                state.storage[:, field.offset : field.offset + field.nbytes]
                .contiguous()
                .view(field.dtype)
            )
            expected_field = (
                expected[:, field.offset : field.offset + field.nbytes]
                .contiguous()
                .view(field.dtype)
            )
            torch.testing.assert_close(
                actual_field,
                expected_field,
                rtol=0,
                atol=0,
                msg=lambda msg, field=field: f"{field.module}.{field.name}: {msg}",
            )
        # The next draft must observe the committed target state and hidden.
        before_draft = state.storage.clone()
        expected_drafts = []
        for row, keep in enumerate(commits):
            if keep:
                state.bind(order[row])
                with torch.device("cuda"), set_dtype(torch.bfloat16):
                    expected_drafts.append(
                        model.core.forward_spec(
                            selected[row, keep - 1 : keep],
                            hidden[row : row + 1],
                            starts[row] + keep - 1,
                        )
                    )
        expected_after_draft = state.storage.clone()
        state.storage.copy_(before_draft)
        bonus = selected.gather(1, (kept - 1).clamp_min(0)[:, None]).flatten()
        result = graph.draft_batch(
            bonus, hidden, pages, positions[:, 0] + (kept - 1).clamp_min(0), kept > 0
        )
        for part, actual in enumerate(result):
            torch.testing.assert_close(
                actual[kept > 0],
                torch.cat([x[part] for x in expected_drafts]),
                rtol=1e-4,
                atol=1e-4,
            )
        torch.testing.assert_close(state.storage, expected_after_draft, rtol=0, atol=0)
    assert graph.stats()["verify_replays"] == 3 * (6 // verify_width)
    assert graph.stats()["verify_graphs"] == 1
    assert graph.stats()["verify_journal_bytes"] < state.storage.numel()
    assert graph.stats()["page_copy_bytes"] == 0
    for name in (
        "VLLM_FL_DECODE_GRAPH",
        "VLLM_FL_BATCHED_DECODE",
        "VLLM_FL_BATCHED_VERIFY",
    ):
        monkeypatch.setenv(name, "1")
    runner = ModelRunnerFL028(model, state, torch.device("cuda"))
    runner.graphs = graph
    draft_map = {}
    for row in (0, 2):
        req_id = f"r{row}"
        runner.requests[req_id] = Request(
            [0] * starts[row] + [int(tokens[row, 0])], order[row], starts[row]
        )
        draft_map[req_id] = tokens[row, 1 : lengths[row]].tolist()
    state.storage.copy_(initial)
    result = runner.execute_decode_batch(
        SimpleNamespace(
            num_scheduled_tokens={
                key: 1 + len(value) for key, value in draft_map.items()
            },
            scheduled_spec_decode_tokens=draft_map,
        )
    )
    assert result.sampled_token_ids == [
        oracle_ids[row][: commits[row]].tolist() for row in (0, 2)
    ]
    assert [runner.requests[f"r{row}"].computed for row in (0, 2)] == [
        starts[row] + commits[row] for row in (0, 2)
    ]
    # An idle DP engine still executes both fixed-size graphs and writes no page.
    idle_before = state.storage.clone()
    runner.data_parallel = True
    result = runner.execute_decode_batch(
        SimpleNamespace(num_scheduled_tokens={}, scheduled_spec_decode_tokens={})
    )
    assert result.sampled_token_ids == []
    torch.testing.assert_close(state.storage, idle_before, rtol=0, atol=0)
