#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint vLLM generation through the independently registered Worker."""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--reference-probe", action="store_true")
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.tokenizers.registry import TokenizerRegistry

    begin = time.monotonic()
    extra = {}
    if args.reference_probe:
        extra["worker_extension_cls"] = (
            "vllm_fl.strict028.validation.ReferenceProbeExtension"
        )
    llm = LLM(
        **extra,
        model=args.model,
        tokenizer_mode="fl_deepseek_v41",
        hf_overrides={"architectures": ["DeepseekV41FlashFLForCausalLM"]},
        tensor_parallel_size=args.tp,
        load_format="fl_dsv41",
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=2,
        max_num_batched_tokens=256,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        async_scheduling=False,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.95,
    )
    loaded = time.monotonic()
    tok = TokenizerRegistry.load_tokenizer("fl_deepseek_v41", args.model)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": text}], thinking=False)
        for text in ["你好，请用一句话介绍自己。", "1加1等于几？请简短回答。"]
    ]
    samples = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    try:
        llm.generate(
            [{"prompt_token_ids": prompts[0]}],
            SamplingParams(temperature=1, max_tokens=1),
        )
    except ValueError as error:
        if "temperature != 0" not in str(error):
            raise
    else:
        raise AssertionError("unsupported sampling reached the Worker")
    differential = None
    result = llm.generate([{"prompt_token_ids": ids} for ids in prompts], samples)
    repeated = llm.generate([{"prompt_token_ids": prompts[0]}], samples)
    rows = [
        {
            "prompt_token_ids": r.prompt_token_ids,
            "output_ids": r.outputs[0].token_ids,
            "text": r.outputs[0].text,
            "finish_reason": r.outputs[0].finish_reason,
        }
        for r in result
    ]
    repeat_equal = repeated[0].outputs[0].token_ids == result[0].outputs[0].token_ids
    report = {
        "status": "reference_pending"
        if args.reference_probe
        else ("passed" if repeat_equal else "failed"),
        "profile": "fl_dsv41_eager_reference_v1",
        "tp": args.tp,
        "load_seconds": loaded - begin,
        "total_seconds": time.monotonic() - begin,
        "repeat_request_equal": repeat_equal,
        "unsupported_sampling_rejected_before_dispatch": True,
        "repeated_output_ids": repeated[0].outputs[0].token_ids,
        "reference_differential": differential,
        "outputs": rows,
        "scope": "text greedy generation and state reuse; not model-quality or performance acceptance",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.reference_probe:
        differential = llm.collective_rpc(
            "fl_reference_differential", args=(prompts[0],)
        )
        report["reference_differential"] = differential
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        boundary_prompt = tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "请用一句话概括："
                    + "人工智能可以帮助人们分析数据、理解语言和解决问题。" * 12,
                }
            ],
            thinking=False,
        )
        if not 128 < len(boundary_prompt) <= 252:
            raise AssertionError("boundary prompt must cross the 128-token window")
        boundary_output = llm.generate(
            [{"prompt_token_ids": boundary_prompt}],
            SamplingParams(temperature=0, max_tokens=4),
        )[0].outputs[0]
        report["window_boundary_output"] = {
            "prompt_tokens": len(boundary_prompt),
            "output_ids": boundary_output.token_ids,
            "text": boundary_output.text,
            "finish_reason": boundary_output.finish_reason,
        }
        boundary = llm.collective_rpc(
            "fl_reference_differential", args=(boundary_prompt,)
        )
        report["reference_differential"] = differential
        report["window_boundary_differential"] = boundary
        report["total_seconds"] = time.monotonic() - begin
        report["status"] = (
            "passed"
            if repeat_equal and all(row["passed"] for row in differential + boundary)
            else "failed"
        )
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "reference_cases": [
                        {
                            "prompt_tokens": case[0]["prompt_tokens"],
                            "prefill_max_relative_rms": max(
                                r["relative_rms"] for r in case
                            ),
                            "decode_max_relative_rms": max(
                                s["relative_rms"] for r in case for s in r["decode"]
                            ),
                        }
                        for case in (differential, boundary)
                    ],
                },
                indent=2,
            )
        )
        if report["status"] != "passed":
            raise AssertionError(
                f"whole-graph differential failed; diagnostics: {args.output}"
            )
    elif not repeat_equal:
        raise AssertionError(
            f"repeated request changed greedy output; diagnostics: {args.output}"
        )


if __name__ == "__main__":
    main()
