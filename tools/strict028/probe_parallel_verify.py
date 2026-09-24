#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TP2/DP4/EP8 verification, row rollback and idle-lane Graph admission."""

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/strict028"))
from test_batched_decode import make_model

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
    graphs = BatchedDecodeGraphs(
        model, state, device, batch_capacity=3,
        verify_width=int(os.environ.get("VLLM_FL_VERIFY_WIDTH", "6")),
    )
    pages = torch.tensor([1, 2, 3], device=device)
    starts = torch.tensor([7, 8, 127], device=device)
    for page, length in zip(pages.tolist(), starts.tolist()):
        state.bind(page, reset=True)
        with torch.device(device), set_dtype(torch.bfloat16):
            ids = (torch.arange(length) + 11 * layout.data_rank).remainder(64)
            _, _, hidden = model.core(ids[None], 0)
            model.core.store_spec_context(hidden, 0)
    initial = state.storage.clone()
    tokens = (pages + starts + 11 * layout.data_rank).remainder(64)
    candidates, expected_ids, expected_hidden, snapshots = [], [], [], []
    active = torch.ones(3, device=device, dtype=torch.bool)
    for offset in range(6):
        candidates.append(tokens.clone())
        logits, hidden = graphs.target_batch(tokens, pages, starts + offset, active)
        tokens = logits.argmax(-1)
        expected_ids.append(tokens.clone())
        expected_hidden.append(hidden.clone())
        snapshots.append(state.storage.clone())
    candidates = torch.stack(candidates, 1)
    expected_ids = torch.stack(expected_ids, 1)
    positions = starts[:, None] + torch.arange(6, device=device)
    cases = []
    for idle in (False, True):
        counts = [1 + layout.data_rank, 6, 3]
        if idle:
            counts = [counts[i] if i < layout.data_rank else 0 for i in range(3)]
        ids = candidates.clone()
        active = torch.tensor(counts, device=device)[:, None].expand(-1, 6) > 0
        for row, count in enumerate(counts):
            if 0 < count < 6:
                ids[row, count] = (ids[row, count] + 1) % 64
        expected_state = initial.clone()
        for row, count in enumerate(counts):
            if count:
                expected_state[pages[row]].copy_(snapshots[count - 1][pages[row]])
        state.storage.copy_(initial)
        selected, kept, hidden, finite = graphs.verify_batch(
            ids, pages, positions, active
        )
        assert finite.item() and kept.tolist() == counts
        for row, count in enumerate(counts):
            if count:
                torch.testing.assert_close(
                    selected[row, :count], expected_ids[row, :count], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    hidden[row], expected_hidden[count - 1][row], rtol=0, atol=0
                )
        torch.testing.assert_close(state.storage, expected_state, rtol=0, atol=0)
        cases.append(
            dict(idle_padding=idle, counts=counts, exact_state=True, exact_hidden=True)
        )
    rows = [None] * 8
    dist.all_gather_object(
        rows,
        dict(
            rank=layout.global_rank,
            data_rank=layout.data_rank,
            cases=cases,
            graphs=graphs.stats(),
        ),
    )
    if layout.global_rank == 0:
        print(json.dumps(dict(passed=True, results=rows)), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
