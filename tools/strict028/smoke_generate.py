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
    if repeated[0].outputs[0].token_ids != result[0].outputs[0].token_ids:
        raise AssertionError("repeated request changed greedy output after cache reuse")
    report = {
        "status": "passed",
        "profile": "fl_dsv41_eager_reference_v1",
        "tp": args.tp,
        "load_seconds": loaded - begin,
        "total_seconds": time.monotonic() - begin,
        "repeat_request_equal": True,
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
        print(json.dumps({"reference_differential": differential}, indent=2))


if __name__ == "__main__":
    main()
