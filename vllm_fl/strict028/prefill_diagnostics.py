# SPDX-License-Identifier: Apache-2.0
"""Read-only real-weight Prefill chunk boundary diagnostics."""

import torch


@torch.inference_mode()
def compare_prefill_layers(worker, ids, chunk_size, layers=2):
    model, state = worker.get_model(), worker.model_runner.state
    events, handles = [], []

    def record(name, kind, value):
        values = value if isinstance(value, (tuple, list)) else (value,)
        events.append(
            (
                name,
                kind,
                tuple(
                    v.detach().clone()
                    for v in values
                    if isinstance(v, torch.Tensor) and v.ndim
                ),
            )
        )

    try:
        for name, module in model.core.named_modules():
            if (
                not name.startswith("layers.")
                or int(name.split(".")[1]) >= layers
                or ".ffn.experts." in name
            ):
                continue
            handles.append(
                module.register_forward_pre_hook(
                    lambda module, args, name=name: record(name, "input", args)
                )
            )
            handles.append(
                module.register_forward_hook(
                    lambda module, args, value, name=name: record(name, "output", value)
                )
            )
        state.bind(0, reset=True)
        model.forward_with_aux(ids, start_pos=0)
        expected = {}
        for name, kind, values in events:
            expected.setdefault((name, kind), []).append(values)
        events.clear()
        state.bind(0, reset=True)
        reports = []
        for start in range(0, len(ids), chunk_size):
            count = min(chunk_size, len(ids) - start)
            model.forward_prefill_chunk(ids[start : start + count], start, len(ids))
            occurrences, differences = {}, []
            for name, kind, values in events:
                key = (name, kind)
                i = occurrences.get(key, 0)
                occurrences[key] = i + 1
                if key not in expected or i >= len(expected[key]):
                    continue
                for part, (actual, full) in enumerate(zip(values, expected[key][i])):
                    if full.ndim >= 2 and full.shape[:2] == (1, len(ids)):
                        golden = full[:, start : start + count]
                    elif full.shape[0] == len(ids):
                        golden = full[start : start + count]
                    else:
                        continue
                    if actual.shape != golden.shape:
                        continue
                    if not torch.equal(actual, golden):
                        differences.append(
                            dict(
                                module=name,
                                kind=kind,
                                part=part,
                                shape=list(actual.shape),
                                different=int((actual != golden).sum()),
                                max_abs=float(
                                    (actual.float() - golden.float()).abs().max()
                                ),
                            )
                        )
                if len(differences) >= 24:
                    break
            reports.append(
                dict(start=start, count=count, first_differences=differences)
            )
            events.clear()
        return reports
    finally:
        for handle in handles:
            handle.remove()
        state.bind(0, reset=True)
