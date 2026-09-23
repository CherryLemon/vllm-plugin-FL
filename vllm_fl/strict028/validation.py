# SPDX-License-Identifier: Apache-2.0
"""Optional real-weight reference probe, exposed through Worker extension API."""

from pathlib import Path

import torch


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
    if not len(model.core.mtp):
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
    def fl_pd_stats(self):
        if self.pd_connector is None:
            return {"rank": self.rank, "enabled": False}
        return {"enabled": True, **self.pd_connector.stats()}

    def fl_tp_stats(self):
        from .collectives import tp_collective_stats

        return {"rank": self.rank, **tp_collective_stats()}

    def fl_reference_differential(self, prompt_ids):
        if isinstance(prompt_ids, str):
            import json

            prompt_ids = json.loads(prompt_ids)
        return reference_differential(self, prompt_ids)

    def fl_set_drafting(self, enabled):
        if isinstance(enabled, str):
            import json

            enabled = json.loads(enabled)
        if not isinstance(enabled, bool):
            raise ValueError("drafting mode must be a boolean")
        runner = self.model_runner
        if runner.graph_enabled:
            raise ValueError("drafting mode is fixed while Decode Graph is enabled")
        if enabled and not len(self.get_model().core.mtp):
            raise ValueError("this model was loaded without DSpark weights")
        runner.drafting_enabled = enabled
        runner.draft_token_ids = None
        return {"rank": self.rank, "enabled": enabled}

    def fl_mtp_stats(self):
        return {"rank": self.rank, **self.model_runner.spec_stats}

    def fl_graph_stats(self):
        graphs = self.model_runner.graphs
        return {
            "rank": self.rank,
            **(graphs.stats() if graphs is not None else {"enabled": False}),
        }

    @torch.inference_mode()
    def fl_batched_decode_differential(self, prompt_ids):
        """Compare the real batched graph with serial target/draft and state.

        Run only on an idle validation deployment. This is a correctness RPC,
        not throughput measurement; it temporarily saves three request pages.
        """
        import json

        from .models.deepseek_v41.model import set_dtype

        if isinstance(prompt_ids, str):
            prompt_ids = json.loads(prompt_ids)
        diagnose_layers = False
        if isinstance(prompt_ids, dict):
            diagnose_layers = bool(prompt_ids.get("diagnose_layers"))
            prompt_ids = prompt_ids["prompt_ids"]
        runner, model = self.model_runner, self.get_model()
        if not runner.batched_decode_enabled or runner.requests:
            raise ValueError("an idle batched Decode validation service is required")
        if len(prompt_ids) < 3 or len(prompt_ids) + 8 >= model.args.max_seq_len:
            raise ValueError("prompt must leave room for target/draft replay")
        state = runner.state
        if state.storage.shape[0] < 4:
            raise ValueError("the probe needs three request pages plus the null page")
        saved = state.storage[1:4].clone()
        previous = state.active_block
        report = {"rank": self.rank, "passes": []}
        try:
            lengths = [len(prompt_ids) - 2, len(prompt_ids) - 1, len(prompt_ids)]
            with torch.device(self.device), set_dtype(torch.bfloat16):
                for page, length in zip((1, 2, 3), lengths):
                    state.bind(page, reset=True)
                    _, _, hidden = model.core(torch.tensor([prompt_ids[:length]], device=self.device), 0)
                    model.core.store_spec_context(hidden, 0)
                for order in ((1, 2, 3), (3, 1, 2)):
                    pages = torch.tensor(order, device=self.device)
                    positions = torch.tensor([lengths[p - 1] for p in order], device=self.device)
                    tokens = torch.tensor([prompt_ids[-p] for p in order], device=self.device)
                    active = torch.ones(3, dtype=torch.bool, device=self.device)
                    if diagnose_layers and "layer_diagnostics" not in report:
                        from .batch_diagnostics import compare_model_layers

                        report["layer_diagnostics"] = compare_model_layers(
                            runner, tokens, pages, positions
                        )
                    initial = state.storage[1:4].clone()
                    serial_logits, serial_hidden = [], []
                    for i, page in enumerate(order):
                        state.bind(page)
                        _, logits, hidden = model.core(tokens[i:i + 1, None], lengths[page - 1])
                        model.core.store_spec_context(hidden, lengths[page - 1])
                        serial_logits.append(logits)
                        serial_hidden.append(hidden)
                    expected = state.storage[1:4].clone()
                    state.storage[1:4].copy_(initial)
                    logits, hidden = runner.graphs.target_batch(tokens, pages, positions, active)
                    golden_logits, golden_hidden = torch.cat(serial_logits), torch.cat(serial_hidden)
                    row = {
                        "positions": positions.tolist(),
                        "logits_equal": torch.equal(logits, golden_logits),
                        "hidden_equal": torch.equal(hidden, golden_hidden),
                        "token_ids_equal": torch.equal(logits.argmax(-1), golden_logits.argmax(-1)),
                        "logits_max_abs": float((logits - golden_logits).abs().max()),
                        "state_equal": torch.equal(state.storage[1:4], expected),
                        "different_fields": [
                            f"{f.module}.{f.name}" for f in state.fields
                            if not torch.equal(
                                state.storage[1:4, f.offset:f.offset + f.nbytes],
                                expected[:, f.offset:f.offset + f.nbytes],
                            )
                        ],
                    }
                    bonus = logits.argmax(-1)
                    initial.copy_(state.storage[1:4])
                    serial_draft = []
                    for i, page in enumerate(order):
                        state.bind(page)
                        serial_draft.append(model.core.forward_spec(bonus[i:i + 1], hidden[i:i + 1], lengths[page - 1]))
                    expected.copy_(state.storage[1:4])
                    state.storage[1:4].copy_(initial)
                    draft = runner.graphs.draft_batch(bonus, hidden, pages, positions, active)
                    row["draft_equal"] = [
                        torch.equal(actual, torch.cat([r[c] for r in serial_draft]))
                        for c, actual in enumerate(draft)
                    ]
                    row["draft_state_equal"] = torch.equal(state.storage[1:4], expected)
                    report["passes"].append(row)
                    lengths = [p + 1 for p in lengths]
            report["graphs"] = runner.graphs.stats()
            return report
        finally:
            state.storage[1:4].copy_(saved)
            state.bind(0 if previous is None else previous)

    def fl_mtp_differential(self, prompt_ids):
        if isinstance(prompt_ids, str):
            import json

            prompt_ids = json.loads(prompt_ids)
        return mtp_differential(self, prompt_ids)

    def fl_cuda_graph_probe(self, prompt_ids):
        """Capture one fixed-position target step on the real distributed model.

        This checks whole-model capture compatibility. It is not the serving
        runner's dynamic-position decode graph.
        """
        import json

        import torch

        if isinstance(prompt_ids, str):
            prompt_ids = json.loads(prompt_ids)
        if (
            not isinstance(prompt_ids, list)
            or not 1 <= len(prompt_ids) < self.model_config.max_model_len - 1
        ):
            raise ValueError("graph probe needs a nonempty prompt with decode room")
        model = self.get_model()
        state = self.model_runner.state
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        position = len(prompt_ids)
        original_temperature = model.core.temperature
        model.core.temperature = 0

        def prepare():
            state.bind(0, reset=True)
            logits, hidden = model.forward_with_aux(ids, start_pos=0)
            if self.model_runner.drafting_enabled:
                model.store_draft_context(hidden, 0)
            return logits[0].argmax().view(1)

        try:
            with torch.inference_mode():
                token = prepare()
                logits, hidden = model.forward_with_aux(token, start_pos=position)
                if self.model_runner.drafting_enabled:
                    model.store_draft_context(hidden, position)
                torch.cuda.synchronize(self.device)
                token = prepare()
                free_before, _ = torch.cuda.mem_get_info(self.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    captured_logits, captured_hidden = model.forward_with_aux(
                        token, start_pos=position
                    )
                    if self.model_runner.drafting_enabled:
                        model.store_draft_context(captured_hidden, position)
                free_after, _ = torch.cuda.mem_get_info(self.device)
                matches = []
                for _ in range(2):
                    fresh_token = prepare()
                    token.copy_(fresh_token)
                    graph.replay()
                    graph_logits = captured_logits.clone()
                    fresh_token = prepare()
                    reference_logits, _ = model.forward_with_aux(
                        fresh_token, start_pos=position
                    )
                    matches.append(bool(torch.equal(graph_logits, reference_logits)))
                return {
                    "rank": self.rank,
                    "position": position,
                    "graph_replay_count": len(matches),
                    "logits_exact_match": matches,
                    "graph_reserved_bytes": free_before - free_after,
                    "scope": "one fixed-position target step; not a serving decode graph",
                }
        finally:
            model.core.temperature = original_temperature
            state.bind(0, reset=True)
            torch.cuda.empty_cache()


def mtp_differential(worker, prompt_ids):
    """Compare real-weight DSpark stages, Markov heads and context with the publisher."""
    import dataclasses
    import hashlib
    import importlib.util
    import sys

    import torch
    from transformers import AutoTokenizer

    model = worker.get_model()
    state = worker.model_runner.state
    source = Path(worker.model_config.model) / "inference"
    sys.path.insert(0, str(source))
    try:
        spec = importlib.util.spec_from_file_location(
            "published_v41_mtp", source / "model.py"
        )
        ref = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ref
        spec.loader.exec_module(ref)
    finally:
        sys.path.remove(str(source))
    tok = AutoTokenizer.from_pretrained(
        worker.model_config.model, local_files_only=True
    )
    with torch.device("meta"), ref.set_dtype(torch.bfloat16):
        golden = ref.Transformer(ref.ModelArgs(**dataclasses.asdict(model.args)), tok)
    golden.load_state_dict(model.core.state_dict(), strict=True, assign=True)
    golden.requires_grad_(False)
    for module in golden.modules():
        if isinstance(module, ref.Linear) and module.scale is not None:
            module.weight.scale = module.scale
    state.bind(0, reset=True)
    copy_reference_buffers(model.core, golden)
    captured, handles = {}, []

    def capture(key):
        def hook(module, inputs, output):
            captured[key] = tuple(t.detach().clone() for t in output)

        return hook

    for i in range(len(model.core.mtp)):
        for tag, graph in (("fl", model.core), ("ref", golden)):
            handles.append(graph.mtp[i].register_forward_hook(capture((tag, i))))

    def difference(a, b):
        af, bf = a.float(), b.float()
        return {
            "max_abs": (af - bf).abs().max().item(),
            "relative_rms": (
                (af - bf).square().mean().sqrt()
                / bf.square().mean().sqrt().clamp_min(1e-30)
            ).item(),
            "equal": torch.equal(a, b),
        }

    def cache_equal(other):
        return all(
            torch.equal(a.attn.window_kv_cache, b.attn.window_kv_cache)
            for a, b in zip(model.core.mtp, other.mtp)
        )

    try:
        rows = []
        with (
            torch.inference_mode(),
            torch.device(worker.device),
            ref.set_dtype(torch.bfloat16),
        ):
            ids = torch.tensor(prompt_ids, device=worker.device).view(1, -1)
            logits, hidden = model.forward_with_aux(ids, start_pos=0)
            token, expected, hidden_ref = golden(ids, 0)
            prefill = {
                "logits": difference(logits, expected),
                "hidden": difference(hidden, hidden_ref),
            }
            model.store_draft_context(hidden, 0)
            golden.forward_spec(token, hidden_ref, 0)
            prefill["context_equal"] = cache_equal(golden)
            for offset in range(4):
                pos = len(prompt_ids) + offset
                logits, hidden = model.forward_with_aux(token, start_pos=pos)
                token, expected, hidden_ref = golden(token.view(1, 1), pos)
                # The Runner commits context separately for accepted target
                # tokens; reference forward_spec commits it inside attention.
                model.store_draft_context(hidden, pos)
                actual_draft = model.propose_draft(token, hidden, pos)
                ref_draft = golden.forward_spec(token, hidden_ref, pos)
                rows.append(
                    {
                        "position": pos,
                        "target_logits": difference(logits, expected),
                        "target_hidden": difference(hidden, hidden_ref),
                        "draft_ids_equal": torch.equal(actual_draft[0], ref_draft[0]),
                        "draft_ids": actual_draft[0].tolist(),
                        "draft_logits": difference(actual_draft[1], ref_draft[1]),
                        "confidence": difference(actual_draft[2], ref_draft[2]),
                        "context_equal": cache_equal(golden),
                        "layers": [
                            {
                                "hidden": difference(
                                    captured["fl", i][0], captured["ref", i][0]
                                ),
                                "mix": difference(
                                    captured["fl", i][1], captured["ref", i][1]
                                ),
                            }
                            for i in range(len(model.core.mtp))
                        ],
                    }
                )
        passed = (
            prefill["context_equal"]
            and all(
                r["context_equal"]
                and r["draft_ids_equal"]
                and all(
                    r[k]["equal"]
                    for k in (
                        "target_logits",
                        "target_hidden",
                        "draft_logits",
                        "confidence",
                    )
                )
                and all(x["hidden"]["equal"] and x["mix"]["equal"] for x in r["layers"])
                for r in rows
            )
            and prefill["logits"]["equal"]
            and prefill["hidden"]["equal"]
        )
        return {
            "rank": worker.rank,
            "passed": passed,
            "prompt_tokens": len(prompt_ids),
            "prefill": prefill,
            "steps": rows,
            "loaded_mtp_layers": len(model.core.mtp),
            "reference_sha256": hashlib.sha256(
                (source / "model.py").read_bytes()
            ).hexdigest(),
        }
    finally:
        for handle in handles:
            handle.remove()
        state.bind(0, reset=True)
        del golden
        captured.clear()
        torch.cuda.empty_cache()
