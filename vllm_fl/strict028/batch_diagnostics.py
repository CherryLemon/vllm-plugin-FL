# SPDX-License-Identifier: Apache-2.0
"""Read-only activation comparison for the optional validation worker RPC."""

import torch

from .batched_decode import DecodeBatch


@torch.inference_mode()
def compare_model_layers(runner, tokens, pages, positions, *, layers=8):
    """Find the first serial-vs-batch difference before attempting capture.

    Hooks only clone tensors in an eager diagnostic run. Request state is
    restored, hooks are removed, and no model or operator binding is replaced.
    """
    model, state = runner.model, runner.state
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
                    x[:1].detach().clone()
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
        state.bind(int(pages[0]))
        model.core(tokens[:1, None], int(positions[0]))
        serial = recorded
        recorded = []
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
                        )
                    result.append(row)
            if len(result) >= 24:
                break
        return {
            "scope": f"first {layers} layers, first request, eager serial vs batch",
            "first_differences": result,
            "serial_events": len(serial),
            "batch_events": len(recorded),
        }
    finally:
        for handle in handles:
            handle.remove()
        state.storage[1:4].copy_(saved)
        state.bind(0 if previous is None else previous)
