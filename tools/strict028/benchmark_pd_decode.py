#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure streamed Decode TPS for the full-state FlagCX PD profile.

The workload and first/last-content metric follow SGLang-FL's
docker/dsv41/benchmark_decode.py. Prefill for every request finishes before a
Decode burst starts, so its time and PD handoff do not enter the decode rate.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "deepseek-v4.1-flash-fl"


def make_prompt(repetitions, padding=0):
    reference = (
        "# Reference: weighted interval scheduling\n"
        "# Jobs have start, end, and value fields. Sort by end. For each job i, "
        "binary-search the last compatible job p(i). The recurrence is "
        "dp[i] = max(dp[i-1], value[i] + dp[p(i)]). Prefer the lexicographically "
        "smaller sequence of original indices when values tie. An empty "
        "schedule has value zero. Include input validation for invalid intervals.\n"
        "def predecessor(ends, start):\n"
        "    lo, hi = 0, len(ends)\n"
        "    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if ends[mid] <= start: lo = mid + 1\n"
        "        else: hi = mid\n"
        "    return lo - 1\n"
    )
    return (
        "Use this reference material when answering the final coding task.\n"
        + reference * repetitions
        + " note" * padding
        + "\nWrite a Python 3 implementation of weighted interval scheduling. "
        "Include the function, type hints, and a brief example.\n"
    )


def request_prompt(prefix, index):
    return prefix + f"\nRequest variant {index}: return the implementation now."


def prompt_encoder(model_path):
    source = Path(__file__).resolve().parents[2] / "vllm_fl/strict028/encoding.py"
    spec = importlib.util.spec_from_file_location("fl_benchmark_encoding", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )

    def encode(text):
        rendered = module.encode_messages(
            [{"role": "user", "content": text}],
            thinking_mode="chat",
            drop_thinking=True,
            reasoning_effort="high",
        )
        return tokenizer.encode(rendered, add_special_tokens=False)

    return encode, hashlib.sha256(source.read_bytes()).hexdigest()


def calibrate_prompt(encode, target):
    lo, hi = 0, 1
    while len(encode(request_prompt(make_prompt(hi), 0))) <= target:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if len(encode(request_prompt(make_prompt(mid), 0))) <= target:
            lo = mid
        else:
            hi = mid
    low, high = 0, target
    while low < high:
        mid = (low + high + 1) // 2
        if len(encode(request_prompt(make_prompt(lo, mid), 0))) <= target:
            low = mid
        else:
            high = mid - 1
    prefix = make_prompt(lo, low)
    return prefix, lo, low


def post(opener, url, body, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return opener.open(request, timeout=timeout)


def rpc(opener, url, method, timeout):
    with post(
        opener,
        url + "/collective_rpc",
        {"method": method, "args": [], "timeout": timeout},
        timeout,
    ) as response:
        return json.load(response)["results"]


def prepare_one(opener, url, prompt_ids, timeout):
    transfer_id = "fl-bench-" + uuid.uuid4().hex
    start = time.perf_counter()
    with post(
        opener,
        url + "/v1/completions",
        {
            "model": MODEL,
            "prompt": prompt_ids,
            "temperature": 0,
            "max_tokens": 1,
            "ignore_eos": True,
            "return_token_ids": True,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "transfer_id": transfer_id,
            },
        },
        timeout,
    ) as response:
        result = json.load(response)
    choice = result["choices"][0]
    tokens = choice["token_ids"]
    params = result.get("kv_transfer_params")
    if (
        choice["finish_reason"] != "length"
        or len(tokens) != 1
        or not params
        or params.get("transfer_id") != transfer_id
    ):
        raise RuntimeError(f"invalid Prefill handoff: {choice=} {params=}")
    params.update(do_remote_prefill=True, do_remote_decode=False)
    return {
        "prompt_ids": prompt_ids,
        "first_token": tokens[0],
        "params": params,
        "prefill_s": time.perf_counter() - start,
    }


def decode_one(opener, url, prepared, index, barrier, timeout, output_tokens):
    barrier.wait(timeout=60)
    start = time.perf_counter()
    body = {
        "model": MODEL,
        "prompt": prepared["prompt_ids"] + [prepared["first_token"]],
        "temperature": 0,
        "max_tokens": output_tokens - 1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "kv_transfer_params": prepared["params"],
    }
    first = last = None
    content = []
    usage = None
    finish_reason = None
    done = False
    with post(opener, url + "/v1/completions", body, timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                chunk = choice.get("text")
                if chunk:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    content.append(chunk)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    end = time.perf_counter()
    if not done or not usage or not content or finish_reason != "length":
        raise RuntimeError(
            f"incomplete Decode request {index}: {done=} {usage=} {finish_reason=}"
        )
    tokens = usage["completion_tokens"]
    if tokens != output_tokens - 1 or last <= first:
        raise RuntimeError(
            f"unexpected Decode output {index}: {tokens=} {first=} {last=}"
        )
    return {
        "index": index,
        "prefill_prompt_tokens": len(prepared["prompt_ids"]),
        "decode_prompt_tokens": usage["prompt_tokens"],
        "prefill_first_token": prepared["first_token"],
        "decode_completion_tokens": tokens,
        "full_completion_tokens": tokens + 1,
        "finish_reason": finish_reason,
        "content_sha256": hashlib.sha256("".join(content).encode()).hexdigest(),
        "prefill_s": prepared["prefill_s"],
        "first_content_from_decode_start_s": first - start,
        "generation_s": last - first,
        "decode_tps": (tokens - 1) / (last - first),
        "start_offset_s": start,
        "first_offset_s": first,
        "last_offset_s": last,
        "end_offset_s": end,
    }


def run_burst(opener, p_url, d_url, prompts, timeout, output_tokens):
    concurrency = len(prompts)
    prefill_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        prepared = list(
            pool.map(
                lambda ids: prepare_one(opener, p_url, ids, timeout),
                prompts,
            )
        )
    prefill_end = time.perf_counter()
    barrier = threading.Barrier(concurrency + 1)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                decode_one,
                opener,
                d_url,
                prepared[i],
                i,
                barrier,
                timeout,
                output_tokens,
            )
            for i in range(concurrency)
        ]
        barrier.wait(timeout=60)
        requests = [future.result() for future in futures]
    first = min(row["first_offset_s"] for row in requests)
    last = max(row["last_offset_s"] for row in requests)
    start = min(row["start_offset_s"] for row in requests)
    end = max(row["end_offset_s"] for row in requests)
    rates = [row["decode_tps"] for row in requests]
    for row in requests:
        for key in (
            "start_offset_s",
            "first_offset_s",
            "last_offset_s",
            "end_offset_s",
        ):
            row[key] -= start
    return {
        "concurrency": concurrency,
        "requests": requests,
        "prefill_phase_s": prefill_end - prefill_start,
        "min_request_decode_tps": min(rates),
        "median_request_decode_tps": statistics.median(rates),
        "aggregate_decode_tps": sum(
            row["decode_completion_tokens"] - 1 for row in requests
        )
        / (last - first),
        "pooled_decode_output_tps": sum(
            row["decode_completion_tokens"] for row in requests
        )
        / (end - start),
        "decode_window_s": last - first,
        "decode_batch_wall_s": end - start,
        "prefill_prompt_token_counts": sorted(
            {row["prefill_prompt_tokens"] for row in requests}
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=32637)
    parser.add_argument("--max-model-len", type=int, default=33792)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if (
        args.target_prompt_tokens < 1
        or args.output_tokens < 3
        or args.target_prompt_tokens + args.output_tokens > args.max_model_len
        or any(c < 1 or c > 16 for c in args.concurrency)
    ):
        parser.error("invalid benchmark shape or concurrency")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    p_url = args.prefill_url.rstrip("/")
    d_url = args.decode_url.rstrip("/")
    encode, encoder_sha = prompt_encoder(args.model_path)
    prefix, repetitions, padding = calibrate_prompt(encode, args.target_prompt_tokens)
    prompts = [encode(request_prompt(prefix, i)) for i in range(max(args.concurrency))]
    if max(map(len, prompts)) + args.output_tokens > args.max_model_len:
        raise ValueError("a request variant exceeds max_model_len")
    metadata = {
        "workload": "shared-prefix weighted-interval coding, fixed output length",
        "reference_method": "SGLang-FL docker/dsv41/benchmark_decode.py",
        "deployment": "flagcx_pd",
        "model": str(args.model_path),
        "prompt_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "encoder_sha256": encoder_sha,
        "reference_repetitions": repetitions,
        "padding_repetitions": padding,
        "prefill_prompt_tokens": [len(ids) for ids in prompts],
        "target_prompt_tokens": args.target_prompt_tokens,
        "output_tokens": args.output_tokens,
        "max_model_len": args.max_model_len,
        "warmups_per_concurrency": args.warmups,
        "measured_rounds_per_concurrency": args.rounds,
        "concurrency": args.concurrency,
        "decode_tps_definition": "(Decode completion_tokens - 1)/(last_content - first_content)",
        "aggregate_decode_tps_definition": "sum(Decode completion_tokens - 1)/(last_content_any - first_content_any)",
        "prefill_phase_excluded": True,
        "pd_handoff_excluded_from_decode_tps": True,
    }
    report = {"status": "running", "metadata": metadata, "rounds": [], "summary": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for url in (p_url, d_url):
            with opener.open(url + "/health", timeout=60):
                pass
        report["before"] = {
            "prefill_pd": rpc(opener, p_url, "fl_pd_stats", args.timeout),
            "decode_pd": rpc(opener, d_url, "fl_pd_stats", args.timeout),
            "decode_mtp": rpc(opener, d_url, "fl_mtp_stats", args.timeout),
        }
        save()
        for concurrency in args.concurrency:
            measured = []
            for round_index in range(-args.warmups, args.rounds):
                result = run_burst(
                    opener,
                    p_url,
                    d_url,
                    prompts[:concurrency],
                    args.timeout,
                    args.output_tokens,
                )
                result.update(round=round_index, warmup=round_index < 0)
                report["rounds"].append(result)
                save()
                print(
                    json.dumps(
                        {
                            "concurrency": concurrency,
                            "round": round_index,
                            "median_request_decode_tps": result[
                                "median_request_decode_tps"
                            ],
                            "aggregate_decode_tps": result["aggregate_decode_tps"],
                            "prefill_phase_s": result["prefill_phase_s"],
                        }
                    ),
                    flush=True,
                )
                if round_index >= 0:
                    measured.append(result)
            rows = [row for result in measured for row in result["requests"]]
            report["summary"][str(concurrency)] = {
                "requests": len(rows),
                "min_request_decode_tps": min(row["decode_tps"] for row in rows),
                "median_request_decode_tps": statistics.median(
                    row["decode_tps"] for row in rows
                ),
                "median_aggregate_decode_tps": statistics.median(
                    result["aggregate_decode_tps"] for result in measured
                ),
                "pooled_decode_output_tps": sum(
                    sum(row["decode_completion_tokens"] for row in result["requests"])
                    for result in measured
                )
                / sum(result["decode_batch_wall_s"] for result in measured),
                "median_prefill_phase_s": statistics.median(
                    result["prefill_phase_s"] for result in measured
                ),
            }
            save()
        report["after"] = {
            "prefill_pd": rpc(opener, p_url, "fl_pd_stats", args.timeout),
            "decode_pd": rpc(opener, d_url, "fl_pd_stats", args.timeout),
            "decode_mtp": rpc(opener, d_url, "fl_mtp_stats", args.timeout),
            "decode_tp": rpc(opener, d_url, "fl_tp_stats", args.timeout),
        }
        expected = sum((args.warmups + args.rounds) * c for c in args.concurrency)
        for role, counter in (
            ("prefill_pd", "sent_pages"),
            ("decode_pd", "received_pages"),
        ):
            before = report["before"][role]
            after = report["after"][role]
            if len(before) != 8 or len(after) != 8:
                raise AssertionError(f"{role} missing TP ranks")
            if any(
                end[counter] - begin[counter] != expected or end["fatal_error"]
                for begin, end in zip(before, after)
            ):
                raise AssertionError(
                    f"{role} did not transfer exactly {expected} pages"
                )
        if any(row["backend"] != "flagcx" for row in report["after"]["decode_tp"]):
            raise AssertionError("Decode did not use FlagCX TP")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "summary": report["summary"],
                    "error": report.get("error"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
