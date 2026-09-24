# SPDX-License-Identifier: Apache-2.0
"""Idle, real-weight admission for the speculative verification graph."""

import torch

from .models.deepseek_v41.model import set_dtype


@torch.inference_mode()
def verify_differential(worker, prompt_ids):
    import json

    from .validation import global_rank_report

    if isinstance(prompt_ids, str):
        prompt_ids = json.loads(prompt_ids)
    runner, model = worker.model_runner, worker.get_model()
    if not runner.batched_verify_enabled or runner.requests:
        raise ValueError("an idle batched verification service is required")
    if not 3 <= len(prompt_ids) < model.args.max_seq_len - 11:
        raise ValueError("prompt must leave six verification and five draft positions")
    state = runner.state
    if state.storage.shape[0] < 4:
        raise ValueError("three request pages and the null page are required")
    previous = state.active_block
    saved = state.storage[1:4].cpu()
    report = {"rank": worker.global_rank, "passes": []}
    try:
        with torch.device(worker.device), set_dtype(torch.bfloat16):
            lengths = [len(prompt_ids) - 2, len(prompt_ids) - 1, len(prompt_ids)]
            for page, length in zip((1, 2, 3), lengths):
                state.bind(page, reset=True)
                _, _, hidden = model.core(torch.tensor([prompt_ids[:length]]), 0)
                model.core.store_spec_context(hidden, 0)
            initial = state.storage[1:4].cpu()
            expected = initial.clone()
            candidates, outputs, hiddens = [], [], []
            counts = [1, 3, 6]
            for page, length, keep in zip((1, 2, 3), lengths, counts):
                state.bind(page)
                token = torch.tensor([[prompt_ids[-page]]])
                ids, selected, kept_hidden = [], [], None
                for offset in range(6):
                    ids.append(int(token.item()))
                    _, logits, hidden = model.core(token, length + offset)
                    model.core.store_spec_context(hidden, length + offset)
                    token = logits.argmax(-1).view(1, 1)
                    selected.append(int(token.item()))
                    if offset + 1 == keep:
                        expected[page - 1].copy_(state.storage[page])
                        kept_hidden = hidden.clone()
                if keep < 6:
                    ids[keep] = (ids[keep] + 1) % model.args.vocab_size
                candidates.append(ids)
                outputs.append(selected[:keep])
                hiddens.append(kept_hidden)
            for order in ([0, 1, 2], [2, 0, 1]):
                state.storage[1:4].copy_(initial)
                pages = torch.tensor([i + 1 for i in order])
                starts = torch.tensor([lengths[i] for i in order])
                positions = starts[:, None] + torch.arange(6)
                ids = torch.tensor([candidates[i] for i in order])
                active = torch.ones_like(ids, dtype=torch.bool)
                selected, kept, hidden, finite = runner.graphs.verify_batch(
                    ids, pages, positions, active
                )
                actual_state = state.storage[1:4].cpu()
                expected_hidden = torch.cat([hiddens[i] for i in order])
                actual_counts = kept.tolist()
                row = {
                    "pages": pages.tolist(),
                    "positions": starts.tolist(),
                    "finite": bool(finite.item()),
                    "kept": actual_counts,
                    "kept_equal": actual_counts == [counts[i] for i in order],
                    "ids_equal": [
                        values[:n]
                        for values, n in zip(selected.tolist(), actual_counts)
                    ]
                    == [outputs[i] for i in order],
                    "hidden_equal": torch.equal(hidden, expected_hidden),
                    "hidden_max_abs": float((hidden - expected_hidden).abs().max()),
                    "state_equal": torch.equal(actual_state, expected),
                    "different_fields": [
                        f"{f.module}.{f.name}"
                        for f in state.fields
                        if not torch.equal(
                            actual_state[:, f.offset : f.offset + f.nbytes],
                            expected[:, f.offset : f.offset + f.nbytes],
                        )
                    ],
                }
                # Verify the next draft against the committed serial state.
                bonus = selected.gather(1, (kept - 1).clamp_min(0)[:, None]).flatten()
                before_draft = state.storage[1:4].cpu()
                reference = []
                for j, i in enumerate(order):
                    state.bind(i + 1)
                    reference.append(
                        model.core.forward_spec(
                            bonus[j : j + 1],
                            hidden[j : j + 1],
                            lengths[i] + actual_counts[j] - 1,
                        )
                    )
                expected_draft_state = state.storage[1:4].cpu()
                state.storage[1:4].copy_(before_draft)
                draft = runner.graphs.draft_batch(
                    bonus,
                    hidden,
                    pages,
                    starts + kept - 1,
                    torch.ones(3, dtype=torch.bool),
                )
                row["draft_equal"] = [
                    torch.equal(value, torch.cat([r[i] for r in reference]))
                    for i, value in enumerate(draft)
                ]
                row["draft_state_equal"] = torch.equal(
                    state.storage[1:4].cpu(), expected_draft_state
                )
                row["passed"] = all(
                    row[k]
                    for k in (
                        "finite",
                        "kept_equal",
                        "ids_equal",
                        "hidden_equal",
                        "state_equal",
                        "draft_state_equal",
                    )
                ) and all(row["draft_equal"])
                report["passes"].append(row)
            report["passed"] = all(row["passed"] for row in report["passes"])
            report["graphs"] = runner.graphs.stats()
            report["memory"] = {
                "free_bytes": torch.cuda.mem_get_info(worker.device)[0],
                "allocated_bytes": torch.cuda.memory_allocated(worker.device),
                "reserved_bytes": torch.cuda.memory_reserved(worker.device),
            }
            return global_rank_report(report)
    finally:
        state.storage[1:4].copy_(saved)
        state.bind(0 if previous is None else previous)
