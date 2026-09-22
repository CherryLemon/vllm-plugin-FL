#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint expert differential; this is not a whole-model acceptance."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

from vllm_fl.strict028.bootstrap import register_models, validate_host
from vllm_fl.strict028.components import FLPackedExpert
from vllm_fl.strict028.weights import ExpertShard, load_expert_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    validate_host()
    register_models()
    torch.cuda.set_device(args.device)
    # The published reference allocates GEMM outputs in the default dtype;
    # its Transformer normally establishes BF16 through set_dtype().
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(2841)
    source = args.model / "inference/kernel.py"
    spec = importlib.util.spec_from_file_location("dsv41_reference_kernels", source)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)

    packed = load_expert_shard(
        args.model,
        args.layer,
        ExpertShard((args.expert,), args.tp_rank, args.tp_size),
        device=f"cuda:{args.device}",
    )
    config = json.loads((args.model / "config.json").read_text())
    clamp = config["text_config"]["swiglu_limit"]
    expert = FLPackedExpert(packed, 0, clamp_limit=clamp)

    def ref_linear(x, w, s):
        q, qs = reference.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        return reference.fp4_gemm(
            q,
            qs,
            w.view(torch.float4_e2m1fn_x2),
            s,
            torch.float8_e8m0fnu,
            act_block_size=32,
        )

    records = []
    for rows, amplitude in ((1, 1.0), (7, 1.0), (7, 64.0)):
        x = (
            torch.randn(
                rows,
                config["text_config"]["hidden_size"],
                dtype=torch.bfloat16,
                device="cuda",
            )
            * amplitude
        )
        routing = torch.rand(rows, 1, dtype=torch.float32, device="cuda") * 1.5
        with torch.inference_mode():
            projections = ref_linear(x, packed.gate_up[0], packed.gate_up_scale[0])
            gate, up = projections.float().chunk(2, dim=-1)
            clamped = bool(((gate > clamp) | (up.abs() > clamp)).any().item())
            activated = torch.nn.functional.silu(gate.clamp(max=clamp)) * up.clamp(
                -clamp, clamp
            )
            activated = (activated * routing).to(x.dtype)
            expected = ref_linear(activated, packed.down[0], packed.down_scale[0])
            actual = expert(x, routing)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0.015)
        error = (actual.float() - expected.float()).abs()
        records.append(
            {
                "rows": rows,
                "input_amplitude": amplitude,
                "clamp_exercised": clamped,
                "max_abs_error": error.max().item(),
                "mean_abs_error": error.mean().item(),
                "reference_abs_max": expected.abs().max().item(),
            }
        )
    if not any(record["clamp_exercised"] for record in records):
        raise AssertionError("Real-weight probe did not exercise the clamp")

    from vllm.tokenizers.registry import TokenizerRegistry

    tokenizer = TokenizerRegistry.load_tokenizer("fl_deepseek_v41", str(args.model))
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "你好"}], thinking=False
    )
    report = {
        "status": "passed",
        "scope": "one real expert/shard and tokenizer; not whole-model inference",
        "device": torch.cuda.get_device_name(),
        "dtype": str(torch.bfloat16),
        "expert_storage_bytes": packed.storage_bytes,
        "layout": packed.provenance,
        "reference": {
            "path": "inference/kernel.py",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "dispatch": {
            "linear": "flag_gems.fused.block_scaled_lowp_linear.block_scaled_lowp_linear",
            "activation": "flag_gems.fused.silu_and_mul_with_clamp.silu_and_mul_with_clamp_out",
            "router_multiply": "flag_gems.mul",
        },
        "tolerance": {"atol": 1e-4, "rtol": 0.015},
        "cases": records,
        "tokenizer": {"prompt_ids": prompt_ids, "eos_token_id": tokenizer.eos_token_id},
        "distributed_collectives_tested": False,
        "performance_status": "not_profiled",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
