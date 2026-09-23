# SPDX-License-Identifier: Apache-2.0
"""Read-only activation comparison for the optional validation worker RPC."""

import torch

from .batched_decode import DecodeBatch


@torch.inference_mode()
def compare_model_layers(
    runner, tokens, pages, positions, *, layers=None, draft_hidden=None
):
    """Find the first serial-vs-batch difference before attempting capture.

    Hooks only clone tensors in an eager diagnostic run. Request state is
    restored, hooks are removed, and no model or operator binding is replaced.
    """
    model, state = runner.model, runner.state
    layers = len(model.core.layers) if layers is None else layers
    saved = state.storage[1:4].clone()
    previous = state.active_block
    recorded = []
    handles = []

    def record(name, kind, value):
        values = value if isinstance(value, (tuple, list)) else (value,)
        recorded.append(
            (
                name,
                kind,
                tuple(
                    x.detach().clone()
                    for x in values
                    if isinstance(x, torch.Tensor) and x.ndim
                ),
            )
        )

    try:
        for name, module in model.core.named_modules():
            if draft_hidden is None:
                wanted = name.startswith("layers.") and int(name.split(".")[1]) < layers
            else:
                wanted = name.startswith("mtp.") or name in ("embed", "head", "norm")
            if not wanted:
                continue
            handles.append(
                module.register_forward_pre_hook(
                    lambda module, args, name=name: record(name, "input", args)
                )
            )
            handles.append(
                module.register_forward_hook(
                    lambda module, args, output, name=name: record(
                        name, "output", output
                    )
                )
            )
        serial_runs = []
        for i in range(tokens.numel()):
            state.bind(int(pages[i]))
            if draft_hidden is None:
                model.core(tokens[i : i + 1, None], int(positions[i]))
            else:
                model.core.forward_spec(
                    tokens[i : i + 1], draft_hidden[i : i + 1], int(positions[i])
                )
            lookup = {}
            for name, kind, values in recorded:
                lookup.setdefault((name, kind), []).append(values)
            serial_runs.append(lookup)
            recorded = []
        state.storage[1:4].copy_(saved)
        context = DecodeBatch(
            state,
            pages,
            positions[:, None],
            torch.ones_like(positions[:, None], dtype=torch.bool),
        )
        runner.graphs._forward(tokens, context, draft_hidden)
        result = []
        occurrences = {}
        for name, kind, actual in recorded:
            key = (name, kind)
            index = occurrences.get(key, 0)
            occurrences[key] = index + 1
            for request, lookup in enumerate(serial_runs):
                if key not in lookup or index >= len(lookup[key]):
                    # A ratio2 owner publishes a key only on completed groups.
                    # The batched path computes all lanes and masks publication.
                    continue
                expected = lookup[key][index]
                for part, (full, e) in enumerate(zip(actual, expected)):
                    width = e.shape[0]
                    a = full[request * width : (request + 1) * width]
                    if a.shape != e.shape or not torch.equal(a, e):
                        row = {
                            "module": name,
                            "kind": kind,
                            "part": part,
                            "request": request,
                            "actual_shape": list(a.shape),
                            "expected_shape": list(e.shape),
                        }
                        if a.shape == e.shape:
                            row.update(
                                mismatched=int((a != e).sum()),
                                max_abs=float((a.float() - e.float()).abs().max()),
                            )
                        result.append(row)
            if len(result) >= 24:
                break
        statistics = []
        for name, kind, actual in recorded:
            if (
                kind != "input"
                or not name.startswith("layers.")
                or len(name.split(".")) != 2
                or not actual
            ):
                continue
            key = (name, kind)
            if any(key not in lookup for lookup in serial_runs):
                continue
            expected = torch.cat([lookup[key][0][0] for lookup in serial_runs])
            if not torch.equal(actual[0], expected):
                continue
            stream = actual[0].flatten(2).float().square()
            serial_mean = torch.cat(
                [part.mean(-1, keepdim=True) for part in stream.split(1)]
            )
            batch_mean = stream.mean(-1, keepdim=True)
            statistics.append(
                {
                    "module": name,
                    "input_equal": True,
                    "serial_mean": serial_mean.flatten().tolist(),
                    "batch_mean": batch_mean.flatten().tolist(),
                    "mean_equal": torch.equal(serial_mean, batch_mean),
                }
            )
        return {
            "scope": f"{'draft' if draft_hidden is not None else 'target'} layers, all {tokens.numel()} requests, eager serial vs batch",
            "first_differences": result,
            "serial_events": sum(
                sum(len(events) for events in lookup.values()) for lookup in serial_runs
            ),
            "batch_events": len(recorded),
            "hc_statistics": statistics,
        }
    finally:
        for handle in handles:
            handle.remove()
        state.storage[1:4].copy_(saved)
        state.bind(0 if previous is None else previous)
