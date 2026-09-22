# SPDX-License-Identifier: Apache-2.0
"""Optional real-weight reference probe, exposed through Worker extension API."""

from pathlib import Path


def copy_reference_buffers(source, target):
    """Materialize every buffer binding, including aliases hidden by deduplication."""
    source_buffers = dict(source.named_buffers(remove_duplicate=False))
    target_buffers = dict(target.named_buffers(remove_duplicate=False))
    if source_buffers.keys() != target_buffers.keys():
        raise AssertionError("reference buffer structure differs")
    copies = {}
    for name, buffer in source_buffers.items():
        other = target_buffers[name]
        if buffer.shape != other.shape or buffer.dtype != other.dtype:
            raise AssertionError(f"reference buffer contract differs: {name}")
        if id(buffer) not in copies:
            copies[id(buffer)] = buffer.clone()
        module_name, _, field = name.rpartition(".")
        setattr(target.get_submodule(module_name), field, copies[id(buffer)])
    if any(b.is_meta for b in target.buffers()):
        raise AssertionError("reference still contains meta buffers")


def reference_differential(worker, prompt_ids):
    """Compare published prefill and cached decode on shared immutable weights.

    The reference is constructed on meta to avoid duplicating the 500GB weights.
    Every static/state buffer is independently materialized from the reset graph.
    This probe observes the installed implementation; it changes no operator
    bindings, backend settings or vLLM objects. RPC returns scalar diagnostics.
    """
    import dataclasses
    import hashlib
    import importlib.util
    import math
    import sys

    import torch
    from transformers import AutoTokenizer

    model = worker.get_model()
    state = worker.model_runner.state
    source_dir = Path(worker.model_config.model) / "inference"
    sys.path.insert(0, str(source_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "published_v41_model", source_dir / "model.py"
        )
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
    finally:
        sys.path.remove(str(source_dir))
    args = dataclasses.asdict(model.args)
    args["dspark_block_size"] = 0
    tok = AutoTokenizer.from_pretrained(
        worker.model_config.model, local_files_only=True
    )
    with torch.device("meta"), reference.set_dtype(torch.bfloat16):
        golden = reference.Transformer(reference.ModelArgs(**args), tok)
    golden.load_state_dict(model.core.state_dict(), strict=True, assign=True)
    golden.requires_grad_(False)
    for module in golden.modules():
        if isinstance(module, reference.Linear) and module.scale is not None:
            module.weight.scale = module.scale
    state.bind(0, reset=True)
    copy_reference_buffers(model.core, golden)
    captured = {}
    handles = []

    def capture(tag):
        def hook(module, inputs, output):
            captured[tag] = output[0].detach().clone()
            captured[tag + "_mix"] = output[1].detach().clone()

        return hook

    def capture_router(tag):
        def hook(module, inputs, output):
            captured[tag] = output[1].detach().clone()

        return hook

    for number in range(len(model.core.layers)):
        for prefix, core in (("fl", model.core), ("ref", golden)):
            handles.append(
                core.layers[number].register_forward_hook(capture(f"{prefix}_{number}"))
            )
            handles.append(
                core.layers[number].ffn.gate.register_forward_hook(
                    capture_router(f"{prefix}_route_{number}")
                )
            )

    def rrms(a, b):
        return (
            (a.float() - b.float()).square().mean().sqrt()
            / b.float().square().mean().sqrt().clamp_min(1e-30)
        ).item()

    def layer_diagnostics():
        diagnostics = {}
        for number in range(len(model.core.layers)):
            a, e = captured[f"fl_{number}"], captured[f"ref_{number}"]
            routes_a = torch.sort(captured[f"fl_route_{number}"], dim=-1).values
            routes_e = torch.sort(captured[f"ref_route_{number}"], dim=-1).values
            diagnostics[str(number)] = {
                "max_abs": (a.float() - e.float()).abs().max().item(),
                "relative_rms": rrms(a, e),
                "last_token_relative_rms": rrms(a[:, -1], e[:, -1]),
                "mix_relative_rms": rrms(
                    captured[f"fl_{number}_mix"], captured[f"ref_{number}_mix"]
                ),
                "router_different_sets": int(
                    (routes_a != routes_e).any(-1).sum().item()
                ),
            }
        return diagnostics

    def logits_diagnostics(actual, expected):
        relative = rrms(actual, expected)
        top_match = torch.equal(actual.argmax(-1), expected.argmax(-1))
        return {
            "passed": math.isfinite(relative) and relative <= 0.03 and top_match,
            "relative_rms": relative,
            "max_abs": (actual.float() - expected.float()).abs().max().item(),
            "top1_equal": top_match,
            "top1_id": actual.argmax(-1).item(),
            "layers": layer_diagnostics(),
        }

    try:
        ids = torch.tensor(prompt_ids, device=worker.device, dtype=torch.long).view(
            1, -1
        )
        if ids.shape[1] + 4 > model.args.max_seq_len:
            raise ValueError("reference probe requires room for four decode tokens")
        with (
            torch.inference_mode(),
            torch.device(worker.device),
            reference.set_dtype(torch.bfloat16),
        ):
            actual = model(ids, start_pos=0)
            _, expected, _ = golden(ids, 0)
            report = logits_diagnostics(actual, expected)
            state.bind(0, reset=True)
            repeated = model(ids, start_pos=0)
            repeat_equal = torch.equal(actual, repeated)
            # Reinitialize both graphs before walking identical teacher-forced
            # decode tokens. State page 0 is reserved and never a live request.
            state.bind(0, reset=True)
            copy_reference_buffers(model.core, golden)
            model(ids, start_pos=0)
            _, expected_step, _ = golden(ids, 0)
            decode = []
            for step in range(4):
                position = ids.shape[1] + step
                token = expected_step.argmax(-1).view(1, 1)
                state.bind(0, reset=False)
                actual_step = model(token, start_pos=position)
                _, expected_step, _ = golden(token, position)
                decode.append(
                    {
                        "position": position,
                        **logits_diagnostics(actual_step, expected_step),
                    }
                )
        report.update(
            {
                "passed": report["passed"]
                and repeat_equal
                and all(row["passed"] for row in decode),
                "rank": worker.rank,
                "prompt_tokens": len(prompt_ids),
                "repeated_logits_equal": repeat_equal,
                "decode": decode,
                "reference_model_sha256": hashlib.sha256(
                    (source_dir / "model.py").read_bytes()
                ).hexdigest(),
                "scope": "real-weight prefill and four cached decode steps; shared immutable weights and equivalent static buffers",
            }
        )
        return report
    finally:
        for handle in handles:
            handle.remove()
        state.bind(0, reset=True)
        del golden
        captured.clear()
        torch.cuda.empty_cache()


class ReferenceProbeExtension:
    def fl_reference_differential(self, prompt_ids):
        return reference_differential(self, prompt_ids)
