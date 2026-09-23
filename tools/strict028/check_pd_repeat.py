#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare identical greedy PD completions before measuring Decode throughput."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from benchmark_pd_decode import (
    MODEL,
    calibrate_prompt,
    post,
    prepare_one,
    prompt_encoder,
    request_prompt,
    rpc,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.prompt_tokens < 1 or args.output_tokens < 2 or args.rounds < 2:
        parser.error("the repeatability probe needs positive input and two rounds")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    p_url = args.prefill_url.rstrip("/")
    d_url = args.decode_url.rstrip("/")
    encode, encoder_sha = prompt_encoder(args.model_path)
    prefix, _, _ = calibrate_prompt(encode, args.prompt_tokens)
    prompt_ids = encode(request_prompt(prefix, 0))
    if len(prompt_ids) > args.prompt_tokens:
        raise ValueError("calibrated request exceeds target prompt length")
    prompt_ids += [encode(" note")[-1]] * (args.prompt_tokens - len(prompt_ids))

    report = {
        "status": "running",
        "prompt_tokens": len(prompt_ids),
        "output_tokens": args.output_tokens,
        "rounds": args.rounds,
        "prompt_ids_sha256": hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "encoder_sha256": encoder_sha,
        "before_graph": rpc(opener, d_url, "fl_graph_stats", args.timeout),
        "requests": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for i in range(args.rounds):
            prepared = prepare_one(opener, p_url, prompt_ids, args.timeout)
            body = {
                "model": MODEL,
                "prompt": prompt_ids + [prepared["first_token"]],
                "temperature": 0,
                "max_tokens": args.output_tokens - 1,
                "ignore_eos": True,
                "return_token_ids": True,
                "kv_transfer_params": prepared["params"],
            }
            with post(opener, d_url + "/v1/completions", body, args.timeout) as response:
                result = json.load(response)
            choice = result["choices"][0]
            row = {
                "round": i,
                "prefill_first_token": prepared["first_token"],
                "decode_token_ids": choice["token_ids"],
                "text_sha256": hashlib.sha256(choice["text"].encode()).hexdigest(),
                "finish_reason": choice["finish_reason"],
                "usage": result["usage"],
            }
            report["requests"].append(row)
            print(json.dumps(row), flush=True)
        sequences = {
            (row["prefill_first_token"], tuple(row["decode_token_ids"]))
            for row in report["requests"]
        }
        report["after_graph"] = rpc(opener, d_url, "fl_graph_stats", args.timeout)
        report["status"] = "passed" if len(sequences) == 1 else "diverged"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("status", report["status"], flush=True)
    if report["status"] != "passed":
        raise AssertionError("identical greedy PD requests diverged")


if __name__ == "__main__":
    main()
