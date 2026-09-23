#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure streamed Decode TPS for the 80-way 128K/8192 FlagCX PD profile.

The target is SGLang-FL's PROFILE_STEADY_DECODE.md, including 80 *active*
Decode requests, DSpark block 5, and CUDA Graph. A small functional run must
be explicitly labeled --smoke; it is not a performance comparison. Prefill
and PD handoff finish before the Decode burst and are excluded from the rate.
"""

import argparse
import hashlib
import importlib.util
import json
import re
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


def post(opener, url, body, timeout, headers=None):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    return opener.open(request, timeout=timeout)


def rpc(opener, url, method, timeout, args=None):
    with post(
        opener,
        url + "/collective_rpc",
        {"method": method, "args": args or [], "timeout": timeout},
        timeout,
    ) as response:
        results = json.load(response)["results"]
        if results and "all_ranks" in results[0]:
            results = results[0]["all_ranks"]
        return sorted(results, key=lambda row: row["rank"])


def admit_deployment(opener, p_url, d_url, prompt_tokens, output_tokens, smoke):
    """Reject an undersized service before preparing the 80 x 128K workload."""
    cards = {}
    for url in (p_url, d_url):
        with opener.open(url + "/health", timeout=60):
            pass
        with opener.open(url + "/v1/models", timeout=60) as response:
            models = json.load(response)["data"]
        if len(models) != 1 or models[0]["max_model_len"] < (
            prompt_tokens + output_tokens
        ):
            raise AssertionError(
                f"{url} does not admit {prompt_tokens + output_tokens} total tokens"
            )
        cards[url] = {"max_model_len": models[0]["max_model_len"]}
    with opener.open(d_url + "/metrics", timeout=60) as response:
        metrics = response.read().decode()
    # Running gauges are created only for engines that have served a request.
    # Startup counters expose every configured DP engine, including idle ones.
    decode_groups = re.findall(
        r"^vllm:num_preemptions_total\{[^}]*\}\s+(\S+)", metrics, re.M
    )
    if not decode_groups:
        raise AssertionError("Decode exposes no DP engine metrics")
    if not smoke and len(decode_groups) != 4:
        raise AssertionError(
            f"Decode exposes {len(decode_groups)} DP metric groups; "
            "the 80-way steady profile requires four"
        )
    return {"models": cards, "decode_dp_metric_groups": len(decode_groups)}


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


def decode_one(
    opener, url, prepared, index, barrier, timeout, output_tokens, dp_rank=0
):
    barrier.wait(timeout=60)
    start = time.perf_counter()
    body = {
        "model": MODEL,
        "prompt": prepared["prompt_ids"] + [prepared["first_token"]],
        "temperature": 0,
        "max_tokens": output_tokens - 1,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
        "kv_transfer_params": prepared["params"],
    }
    first = last = None
    arrivals = []
    generated_ids = []
    content = []
    usage = None
    finish_reason = None
    done = False
    with post(
        opener,
        url + "/v1/completions",
        body,
        timeout,
        {"X-data-parallel-rank": str(dp_rank)},
    ) as response:
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
                if token_ids := choice.get("token_ids"):
                    arrivals.append([time.perf_counter(), len(token_ids)])
                    generated_ids.extend(token_ids)
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
    if sum(n for _, n in arrivals) != tokens:
        raise RuntimeError(
            f"streamed token IDs disagree with completion usage: {index}"
        )
    if tokens != output_tokens - 1 or last <= first:
        raise RuntimeError(
            f"unexpected Decode output {index}: {tokens=} {first=} {last=}"
        )
    return {
        "index": index,
        "data_parallel_rank": dp_rank,
        "token_arrivals": arrivals,
        "prefill_prompt_tokens": len(prepared["prompt_ids"]),
        "decode_prompt_tokens": usage["prompt_tokens"],
        "prefill_first_token": prepared["first_token"],
        "decode_completion_tokens": tokens,
        "full_completion_tokens": tokens + 1,
        "finish_reason": finish_reason,
        "content_sha256": hashlib.sha256("".join(content).encode()).hexdigest(),
        "token_ids_sha256": hashlib.sha256(
            json.dumps(generated_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "prefill_s": prepared["prefill_s"],
        "first_content_from_decode_start_s": first - start,
        "generation_s": last - first,
        "decode_tps": (tokens - arrivals[0][1]) / (arrivals[-1][0] - arrivals[0][0]),
        "first_chunk_tokens": arrivals[0][1],
        "start_offset_s": start,
        "first_offset_s": first,
        "last_offset_s": last,
        "end_offset_s": end,
    }


def longest_steady_window(samples):
    """Keep one uninterrupted 4x20 interval, without waiting or preemption."""
    longest, current = [], []
    for sample in samples:
        valid = (
            sample.get("running") == [20.0] * 4 and sample.get("waiting") == [0.0] * 4
        )
        if not valid:
            current = []
            continue
        if current and (
            sample.get("preemptions") != current[-1].get("preemptions")
            or sample["time_s"] - current[-1]["time_s"] > 2
        ):
            current = []
        current.append(sample)
        if len(current) > len(longest):
            longest = list(current)
    return longest


def measure_steady_window(samples, requests, origin):
    if not samples:
        return None
    # Engine gauges update less often than stream events. Bound them by the
    # interval in which every client has started and none has finished.
    begin = max(samples[0]["time_s"], max(r["token_arrivals"][0][0] for r in requests))
    end = min(samples[-1]["time_s"], min(r["token_arrivals"][-1][0] for r in requests))
    if end <= begin:
        return None
    emitted = [
        sum(n for t, n in row["token_arrivals"] if begin < t <= end)
        for row in requests
    ]
    return dict(
        start_s=begin - origin,
        end_s=end - origin,
        duration_s=end - begin,
        boundary_source="intersection of 4x20 metrics and all request token streams",
        tokens_per_request=emitted,
        aggregate_tps=sum(emitted) / (end - begin),
        median_request_tps=statistics.median(emitted) / (end - begin),
        min_request_tps=min(emitted) / (end - begin),
        max_request_tps=max(emitted) / (end - begin),
    )


def run_burst(
    opener, p_url, d_url, prompts, timeout, output_tokens, profile_label=None, dp_size=1,
    sample_path=None,
):
    concurrency = len(prompts)
    prefill_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        prepared = []
        for index, row in enumerate(pool.map(
            lambda ids: prepare_one(opener, p_url, ids, timeout), prompts
        )):
            prepared.append(row)
            print(json.dumps(dict(
                event="prefill_ready", request=index, total=concurrency,
                prefill_s=row["prefill_s"],
            )), flush=True)
    prefill_end = time.perf_counter()
    print(json.dumps(dict(event="decode_burst_start", concurrency=concurrency)), flush=True)
    barrier = threading.Barrier(concurrency + 1)
    samples = []
    monitor_stop = threading.Event()
    profile_events = []
    if sample_path is not None:
        sample_path.write_text("")

    def monitor():
        while not monitor_stop.is_set():
            stamp = time.perf_counter()
            sample_begin = len(samples)
            try:
                with opener.open(d_url + "/metrics", timeout=5) as response:
                    metrics = response.read().decode()
                values = {}
                for name in ("running", "waiting"):
                    values[name] = [
                        float(v)
                        for v in re.findall(
                            rf"^vllm:num_requests_{name}\{{[^}}]*\}}\s+(\S+)",
                            metrics,
                            re.M,
                        )
                    ]
                preemptions = sum(
                    float(v)
                    for v in re.findall(
                        r"^vllm:num_preemptions_total\{[^}]*\}\s+(\S+)", metrics, re.M
                    )
                )
                samples.append({"time_s": stamp, "preemptions": preemptions, **values})
                steady = len(samples) >= 5 and all(
                    s.get("running") == [20.0] * 4
                    and s.get("waiting") == [0.0] * 4
                    and s.get("preemptions") == preemptions
                    for s in samples[-5:]
                )
                if profile_label and steady and not profile_events:
                    profile_events.append({"armed_at_s": time.perf_counter()})
                    profile_events[-1]["response"] = rpc(
                        opener,
                        d_url,
                        "fl_profile_decode",
                        120,
                        [dict(label=profile_label, steps=10, active=20)],
                    )
            except Exception as error:
                samples.append({"time_s": stamp, "error": repr(error)})
            if sample_path is not None:
                with sample_path.open("a") as journal:
                    for sample in samples[sample_begin:]:
                        journal.write(json.dumps(sample) + "\n")
            monitor_stop.wait(0.5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
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
                    i % dp_size,
                )
                for i in range(concurrency)
            ]
            barrier.wait(timeout=60)
            requests = [future.result() for future in futures]
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=10)
    active = [sum(s.get("running", [])) for s in samples]
    steady_samples = longest_steady_window(samples)
    first = min(row["first_offset_s"] for row in requests)
    last = max(row["last_offset_s"] for row in requests)
    start = min(row["start_offset_s"] for row in requests)
    end = max(row["end_offset_s"] for row in requests)
    rates = [row["decode_tps"] for row in requests]
    steady_rate = measure_steady_window(steady_samples, requests, start)
    for row in requests:
        row["token_arrivals"] = [[t - start, n] for t, n in row["token_arrivals"]]
        for key in (
            "start_offset_s",
            "first_offset_s",
            "last_offset_s",
            "end_offset_s",
        ):
            row[key] -= start
    return {
        "concurrency": concurrency,
        "monotonic_origin_s": start,
        "occupancy_journal": None if sample_path is None else str(sample_path),
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
        "peak_active_decode_requests": max(active, default=0),
        "steady_c80_dp4_samples": len(steady_samples),
        "steady_window": steady_rate,
        "profile_events": profile_events,
        "occupancy_samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=131072)
    parser.add_argument("--max-model-len", type=int, default=139264)
    parser.add_argument("--output-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[80])
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--prefill-cache", action="store_true",
        help="record and require the producer's exact-prefix state cache",
    )
    parser.add_argument(
        "--profile-label",
        help="capture 10 steps per rank after C80 occupancy is steady; run separately from throughput",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="label an explicitly smaller functional run; never report it as steady performance",
    )
    args = parser.parse_args()
    if (
        args.target_prompt_tokens < 1
        or args.output_tokens < 3
        or args.target_prompt_tokens + args.output_tokens > args.max_model_len
        or any(c < 1 or c > 80 for c in args.concurrency)
        or args.rounds < 1
        or args.warmups < 0
    ):
        parser.error("invalid benchmark shape or concurrency")
    if not args.smoke and (
        args.target_prompt_tokens != 131072
        or args.output_tokens != 8192
        or args.concurrency != [80]
    ):
        parser.error(
            "steady Decode performance requires exactly 131072 input, "
            "8192 output, and 80 active requests; use --smoke for a smaller functional run"
        )

    if args.profile_label and (args.rounds != 1 or args.warmups or args.smoke):
        parser.error("profiling requires one target round without warmups")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    p_url = args.prefill_url.rstrip("/")
    d_url = args.decode_url.rstrip("/")
    admission = admit_deployment(
        opener,
        p_url,
        d_url,
        args.target_prompt_tokens,
        args.output_tokens,
        args.smoke,
    )
    encode, encoder_sha = prompt_encoder(args.model_path)
    prefix, repetitions, padding = calibrate_prompt(encode, args.target_prompt_tokens)
    prompts = [encode(request_prompt(prefix, i)) for i in range(max(args.concurrency))]
    if max(map(len, prompts)) > args.target_prompt_tokens:
        raise ValueError("a request variant exceeds target_prompt_tokens")
    pad_id = encode(" note")[-1]
    prompts = [
        ids + [pad_id] * (args.target_prompt_tokens - len(ids)) for ids in prompts
    ]
    if max(map(len, prompts)) + args.output_tokens > args.max_model_len:
        raise ValueError("a request variant exceeds max_model_len")
    metadata = {
        "profiler_enabled": bool(args.profile_label),
        "throughput_comparable": not args.smoke and not args.profile_label,
        "measurement_scope": "functional_smoke"
        if args.smoke
        else "steady_decode_80_active",
        "workload": "shared-prefix weighted-interval coding, fixed output length",
        "reference_method": "SGLang-FL docker/dsv41/PROFILE_STEADY_DECODE.md",
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
        "required_active_decode_requests": None if args.smoke else 80,
        "required_decode_topology": None
        if args.smoke
        else "attention TP2 x DP4; global TP8/EP8",
        "deployment_admission": admission,
        "decode_tps_definition": "(Decode completion_tokens - first_chunk_tokens)/(last_token_chunk - first_token_chunk)",
        "aggregate_decode_tps_definition": "sum(Decode completion_tokens - 1)/(last_content_any - first_content_any)",
        "prefill_phase_excluded": True,
        "prefill_prefix_state_cache": args.prefill_cache,
        "pd_handoff_excluded_from_decode_tps": True,
    }
    report = {"status": "running", "metadata": metadata, "rounds": [], "summary": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        report["before"] = {
            "prefill_pd": rpc(opener, p_url, "fl_pd_stats", args.timeout),
            "decode_pd": rpc(opener, d_url, "fl_pd_stats", args.timeout),
            "decode_mtp": rpc(opener, d_url, "fl_mtp_stats", args.timeout),
            "decode_graph": rpc(opener, d_url, "fl_graph_stats", args.timeout),
        }
        if any(not row["enabled"] for row in report["before"]["decode_graph"]):
            raise AssertionError("Decode Graph is not enabled on every rank")
        if args.prefill_cache:
            cache = rpc(opener, p_url, "fl_prefill_cache_stats", args.timeout)
            if len(cache) != 8 or any(not r["enabled"] for r in cache):
                raise AssertionError("Prefill prefix cache is not enabled on all ranks")
            report["before"]["prefill_cache"] = cache
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
                    args.profile_label,
                    admission["decode_dp_metric_groups"],
                    args.output.with_suffix(f".c{concurrency}.r{round_index}.metrics.jsonl"),
                )
                result.update(round=round_index, warmup=round_index < 0)
                report["rounds"].append(result)
                save()
                if not args.smoke and (
                    result["steady_c80_dp4_samples"] < 5
                    or result["steady_window"] is None
                ):
                    raise AssertionError(
                        "80 active Decode requests across four DP groups with "
                        "zero waiting were not sustained for five metric samples"
                    )
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
            if args.smoke and concurrency == 1 and len(measured) > 1:
                hashes = {row["content_sha256"] for row in rows}
                token_hashes = {row["token_ids_sha256"] for row in rows}
                first_tokens = {row["prefill_first_token"] for row in rows}
                if len(hashes) != 1 or len(token_hashes) != 1 or len(first_tokens) != 1:
                    raise AssertionError(
                        "identical greedy PD smoke requests produced different output"
                    )
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
            steady = [r["steady_window"] for r in measured if r["steady_window"]]
            if steady:
                report["summary"][str(concurrency)].update(
                    median_common_window_aggregate_tps=statistics.median(
                        r["aggregate_tps"] for r in steady
                    ),
                    median_common_window_request_tps=statistics.median(
                        r["median_request_tps"] for r in steady
                    ),
                )
            save()
        report["after"] = {
            "prefill_pd": rpc(opener, p_url, "fl_pd_stats", args.timeout),
            "decode_pd": rpc(opener, d_url, "fl_pd_stats", args.timeout),
            "decode_mtp": rpc(opener, d_url, "fl_mtp_stats", args.timeout),
            "decode_tp": rpc(opener, d_url, "fl_tp_stats", args.timeout),
            "decode_graph": rpc(opener, d_url, "fl_graph_stats", args.timeout),
        }
        if args.prefill_cache:
            report["after"]["prefill_cache"] = rpc(
                opener, p_url, "fl_prefill_cache_stats", args.timeout
            )
        expected = sum((args.warmups + args.rounds) * c for c in args.concurrency)
        p_before = report["before"]["prefill_pd"]
        p_after = report["after"]["prefill_pd"]
        d_before = report["before"]["decode_pd"]
        d_after = report["after"]["decode_pd"]
        for role, before, after in (
            ("prefill", p_before, p_after),
            ("decode", d_before, d_after),
        ):
            if [r["rank"] for r in before] != list(range(8)) or [
                r["rank"] for r in after
            ] != list(range(8)):
                raise AssertionError(f"{role} missing global ranks")
            if any(r["fatal_error"] for r in after):
                raise AssertionError(f"{role} reported a transfer failure")
        for begin, end in zip(p_before, p_after):
            completed = sum(
                end.get(k, 0) - begin.get(k, 0)
                for k in ("sent_pages", "released_pages")
            )
            if completed != expected or end["pending_sends"] != begin["pending_sends"]:
                raise AssertionError(
                    f"Prefill rank {end['rank']} retained or lost requests"
                )
        received = [
            e["received_pages"] - b["received_pages"] for b, e in zip(d_before, d_after)
        ]
        tp_size = d_after[0]["tp_size"]
        if sum(received) != expected * tp_size or any(
            len(set(received[i : i + tp_size])) != 1 for i in range(0, 8, tp_size)
        ):
            raise AssertionError(
                f"Decode TP groups lost or duplicated pages: {received}"
            )
        if any(row["backend"] != "flagcx" for row in report["after"]["decode_tp"]):
            raise AssertionError("Decode did not use FlagCX TP")
        for begin, end in zip(
            report["before"]["decode_graph"], report["after"]["decode_graph"]
        ):
            if (
                end["target_replays"] <= begin["target_replays"]
                or end["draft_replays"] <= begin["draft_replays"]
                or end["target_graphs"] == 0
                or end["draft_graphs"] == 0
            ):
                raise AssertionError("Decode or DSpark did not actually replay a graph")
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
