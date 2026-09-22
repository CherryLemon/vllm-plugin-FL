#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real checkpoint DSpark acceptance on the official 0.28 empty host, without PD."""

import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=24)
    a = p.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.tokenizers.registry import TokenizerRegistry

    begin = time.monotonic()
    llm = LLM(
        model=a.model,
        tokenizer_mode="fl_deepseek_v41",
        hf_overrides={"architectures": ["DeepseekV41FlashFLForCausalLM"]},
        tensor_parallel_size=8,
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
        speculative_config={"method": "dspark", "num_speculative_tokens": 5},
        worker_extension_cls="vllm_fl.strict028.validation.ReferenceProbeExtension",
    )
    report = {
        "status": "running",
        "load_seconds": time.monotonic() - begin,
        "tp": 8,
        "method": "dspark",
        "num_speculative_tokens": 5,
        "verification": "serial greedy with accepted-context commits",
        "pd": False,
    }

    def save():
        report["total_seconds"] = time.monotonic() - begin
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    tok = TokenizerRegistry.load_tokenizer("fl_deepseek_v41", a.model)

    def prompt(text):
        return tok.apply_chat_template(
            [{"role": "user", "content": text}], thinking=False
        )

    prompts = [
        prompt("请用中文简短介绍北京。"),
        prompt("请依次写出数字一到十，用逗号分隔。"),
    ]
    params = SamplingParams(temperature=0, max_tokens=a.max_tokens)

    def generate(ids, sampling=params):
        out = llm.generate([{"prompt_token_ids": i} for i in ids], sampling)
        return [
            {
                "prompt_token_ids": r.prompt_token_ids,
                "output_ids": list(r.outputs[0].token_ids),
                "text": r.outputs[0].text,
                "finish_reason": r.outputs[0].finish_reason,
            }
            for r in out
        ]

    llm.collective_rpc("fl_set_drafting", args=(False,))
    report["without_mtp"] = generate(prompts)
    save()
    llm.collective_rpc("fl_set_drafting", args=(True,))
    report["with_mtp"] = generate(prompts)
    report["repeat"] = generate(prompts[:1])
    report["greedy_equal"] = report["without_mtp"] == report["with_mtp"]
    report["repeat_equal"] = report["repeat"] == report["with_mtp"][:1]
    report["stats"] = llm.collective_rpc("fl_mtp_stats")
    save()
    arithmetic = [prompt("1+1等于几？只回答数字。")]
    report["eos"] = generate(arithmetic, SamplingParams(temperature=0, max_tokens=8))
    boundary = prompt(
        "请用一句话概括：" + "人工智能可以帮助人们分析数据、理解语言和解决问题。" * 12
    )
    assert 128 < len(boundary) < 240
    llm.collective_rpc("fl_set_drafting", args=(False,))
    report["boundary_without_mtp"] = generate(
        [boundary], SamplingParams(temperature=0, max_tokens=12)
    )
    llm.collective_rpc("fl_set_drafting", args=(True,))
    report["boundary_with_mtp"] = generate(
        [boundary], SamplingParams(temperature=0, max_tokens=12)
    )
    report["boundary_equal"] = (
        report["boundary_without_mtp"] == report["boundary_with_mtp"]
    )
    save()
    report["reference"] = llm.collective_rpc("fl_mtp_differential", args=(prompts[0],))
    save()
    report["boundary_reference"] = llm.collective_rpc(
        "fl_mtp_differential", args=(boundary,)
    )
    report["stats"] = llm.collective_rpc("fl_mtp_stats")
    report["status"] = (
        "passed"
        if (
            report["greedy_equal"]
            and report["repeat_equal"]
            and report["boundary_equal"]
            and all(
                r["passed"] for r in report["reference"] + report["boundary_reference"]
            )
            and all(r["accepted_tokens"] > 0 for r in report["stats"])
            and report["eos"][0]["finish_reason"] == "stop"
        )
        else "failed"
    )
    save()
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "greedy_equal",
                    "repeat_equal",
                    "boundary_equal",
                    "stats",
                )
            },
            indent=2,
        )
    )
    if report["status"] != "passed":
        raise AssertionError(f"DSpark acceptance failed: {a.output}")


if __name__ == "__main__":
    main()
