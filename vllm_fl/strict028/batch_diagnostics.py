# SPDX-License-Identifier: Apache-2.0
"""Read-only activation comparison for the optional validation worker RPC."""

import torch

from .batched_decode import DecodeBatch


@torch.inference_mode()
def compare_model_layers(runner, tokens, pages, positions, *, layers=None):
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
            if not name.startswith("layers.") or int(name.split(".")[1]) >= layers:
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
            model.core(tokens[i : i + 1, None], int(positions[i]))
            serial_runs.append(recorded)
            recorded = []
        serial = []
        for events in zip(*serial_runs):
            name, kind = events[0][:2]
            if any(event[:2] != (name, kind) for event in events):
                raise AssertionError(
                    "serial requests took different diagnostic module paths"
                )
            serial.append(
                (
                    name,
                    kind,
                    tuple(
                        torch.cat(parts)
                        for parts in zip(*(event[2] for event in events))
                    ),
                )
            )
        state.storage[1:4].copy_(saved)
        context = DecodeBatch(
            state,
            pages,
            positions[:, None],
            torch.ones_like(positions[:, None], dtype=torch.bool),
        )
        runner.graphs._forward(tokens, context, None)
        result = []
        lookup = {}
        for name, kind, values in serial:
            lookup.setdefault((name, kind), []).append(values)
        occurrences = {}
        for name, kind, actual in recorded:
            key = (name, kind)
            index = occurrences.get(key, 0)
            occurrences[key] = index + 1
            if key not in lookup or index >= len(lookup[key]):
                continue
            expected = lookup[key][index]
            for part, (a, e) in enumerate(zip(actual, expected)):
                if a.shape != e.shape or not torch.equal(a, e):
                    row = {
                        "module": name,
                        "kind": kind,
                        "part": part,
                        "actual_shape": list(a.shape),
                        "expected_shape": list(e.shape),
                    }
                    if a.shape == e.shape:
                        row.update(
                            mismatched=int((a != e).sum()),
                            max_abs=float((a.float() - e.float()).abs().max()),
                            different_requests=(a != e)
                            .flatten(1)
                            .any(1)
                            .nonzero()
                            .flatten()
                            .tolist(),
                        )
                    result.append(row)
            if len(result) >= 24:
                break
        statistics = []
        for name, kind, actual in recorded:
            if kind != "input" or len(name.split(".")) != 2 or not actual:
                continue
            expected = lookup[(name, kind)][0]
            if not torch.equal(actual[0], expected[0]):
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
            "scope": f"first {layers} layers, all {tokens.numel()} requests, eager serial vs batch",
            "first_differences": result,
            "serial_events": len(serial),
            "batch_events": len(recorded),
            "hc_statistics": statistics,
        }
    finally:
        for handle in handles:
            handle.remove()
        state.storage[1:4].copy_(saved)
        state.bind(0 if previous is None else previous)
