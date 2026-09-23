#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Eight-rank mixed-axis model/state correctness on small dimensions."""

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/strict028"))
from test_batched_decode import make_model, serial_target

from vllm_fl.strict028.batched_graph import BatchedDecodeGraphs
from vllm_fl.strict028.collectives import init_tp_collectives, parallel_layout
from vllm_fl.strict028.models.deepseek_v41.model import set_dtype


@torch.inference_mode()
def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("gloo")
    init_tp_collectives(device, tensor_size=2)
    layout = parallel_layout()
    assert layout.world_size == 8
    model, state = make_model(parallel=layout)
    graph = BatchedDecodeGraphs(model, state, device, batch_capacity=3)
    for page, length in ((1, 7), (2, 8), (3, 127)):
        state.bind(page, reset=True)
        with torch.device(device), set_dtype(torch.bfloat16):
            ids = (torch.arange(length) + 11 * layout.data_rank).remainder(64)
            _, _, hidden = model.core(ids[None], 0)
            model.core.store_spec_context(hidden, 0)
    reports = []
    for order, positions_list in (([1, 2, 3], [7, 8, 127]), ([3, 1, 2], [128, 8, 9])):
        pages = torch.tensor(order, device=device)
        positions = torch.tensor(positions_list, device=device)
        active = torch.ones(3, device=device, dtype=torch.bool)
        tokens = (pages + positions + 11 * layout.data_rank).remainder(64)
        initial = state.storage.clone()
        expected_logits, expected_hidden = serial_target(
            model, state, tokens, pages, positions, active
        )
        expected_state = state.storage.clone()
        state.storage.copy_(initial)
        logits, hidden = graph.target_batch(tokens, pages, positions, active)
        torch.testing.assert_close(logits, expected_logits, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(hidden, expected_hidden, rtol=0, atol=0)
        torch.testing.assert_close(state.storage, expected_state, rtol=0, atol=0)
        initial.copy_(state.storage)
        drafts = []
        bonus = logits.argmax(-1)
        for i, page in enumerate(order):
            state.bind(page)
            with torch.device(device), set_dtype(torch.bfloat16):
                drafts.append(
                    model.core.forward_spec(
                        bonus[i : i + 1], hidden[i : i + 1], positions_list[i]
                    )
                )
        expected_state.copy_(state.storage)
        state.storage.copy_(initial)
        actual = graph.draft_batch(bonus, hidden, pages, positions, active)
        for part, value in enumerate(actual):
            torch.testing.assert_close(
                value, torch.cat([row[part] for row in drafts]), rtol=1e-4, atol=1e-4
            )
        torch.testing.assert_close(state.storage, expected_state, rtol=0, atol=0)
        reports.append({"positions": positions_list, "passed": True})
    # Uneven occupancy includes a completely idle DP group. Every group still
    # uses the same graph and EP collectives; only active request pages change.
    count = layout.data_rank
    positions += 1
    initial = state.storage.clone()
    expected_logits, expected_hidden = serial_target(
        model, state, tokens, pages, positions, active
    )
    expected_state = state.storage.clone()
    for page in pages[count:].tolist():
        expected_state[page].copy_(initial[page])
    state.storage.copy_(initial)
    logits, hidden = graph.target_batch(
        tokens[:count], pages[:count], positions[:count], active[:count]
    )
    torch.testing.assert_close(logits, expected_logits[:count], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(hidden, expected_hidden[:count], rtol=0, atol=0)
    torch.testing.assert_close(state.storage, expected_state, rtol=0, atol=0)
    reports.append({"local_requests": count, "idle_padding_passed": True})
    rows = [None] * 8
    dist.all_gather_object(
        rows,
        {
            "rank": layout.global_rank,
            "data_rank": layout.data_rank,
            "passes": reports,
            "graphs": graph.stats(),
        },
    )
    if layout.global_rank == 0:
        print(json.dumps({"results": rows}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
