# SPDX-License-Identifier: Apache-2.0
"""Optional real-weight reference probe, exposed through Worker extension API."""

from pathlib import Path


def reference_differential(worker, prompt_ids):
    """Run the published graph on shared immutable weights, independent buffers.

    RPC transports only scalar diagnostics. No vLLM function is replaced and no
    second copy of the 500GB checkpoint is made. Meta construction avoids a
    transient full duplicate of every weight; all derived buffers are copied
    from the equivalent graph after reset and recorded as this probe's scope.
    """
    import dataclasses
    import hashlib
    import importlib.util
    import sys

    import torch
    from transformers import AutoTokenizer

    model = worker.get_model()
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
    args["dspark_block_size"] = (
        0  # The published generation loop also does not call MTP.
    )
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
    worker.model_runner.state.bind(0, reset=True)
    for name, buffer in model.core.named_buffers():
        module_name, _, field = name.rpartition(".")
        setattr(golden.get_submodule(module_name), field, buffer.clone())
    captured = {}
    handles = []

    def capture(tag):
        def hook(module, inputs, output):
            captured[tag] = output[0].detach().clone()

        return hook

    for number in (0, 1, 2, 14, 20, 39):
        handles.append(
            model.core.layers[number].register_forward_hook(capture(f"fl_{number}"))
        )
        handles.append(
            golden.layers[number].register_forward_hook(capture(f"ref_{number}"))
        )
    try:
        ids = torch.tensor(prompt_ids, device=worker.device, dtype=torch.long).view(
            1, -1
        )
        with (
            torch.inference_mode(),
            torch.device(worker.device),
            reference.set_dtype(torch.bfloat16),
        ):
            actual = model(ids, start_pos=0)
            _, expected, _ = golden(ids, 0)
        diagnostics = {}
        for number in (0, 1, 2, 14, 20, 39):
            a, e = captured[f"fl_{number}"].float(), captured[f"ref_{number}"].float()
            diagnostics[str(number)] = {
                "max_abs": (a - e).abs().max().item(),
                "relative_rms": (
                    (a - e).square().mean().sqrt() / e.square().mean().sqrt()
                ).item(),
            }
        error = actual.float() - expected.float()
        relative = (
            error.square().mean().sqrt() / expected.float().square().mean().sqrt()
        ).item()
        top_match = actual.argmax(-1).tolist() == expected.argmax(-1).tolist()
        if relative > 0.03 or not top_match:
            raise AssertionError(
                f"whole-graph differential failed: relative_rms={relative}, top1={top_match}, layers={diagnostics}"
            )
        return {
            "rank": worker.rank,
            "relative_rms": relative,
            "max_abs": error.abs().max().item(),
            "top1_equal": top_match,
            "top1_id": actual.argmax(-1).item(),
            "layers": diagnostics,
            "reference_model_sha256": hashlib.sha256(
                (source_dir / "model.py").read_bytes()
            ).hexdigest(),
            "scope": "real-weight prefill graph; shared immutable weights and equivalent static buffers",
        }
    finally:
        for handle in handles:
            handle.remove()
        del golden
        captured.clear()
        torch.cuda.empty_cache()


class ReferenceProbeExtension:
    def fl_reference_differential(self, prompt_ids):
        return reference_differential(self, prompt_ids)
