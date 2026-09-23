#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise real-checkpoint FL Prefill/Decode and verify stitched token IDs."""

import argparse
import json
import urllib.request
import uuid
from pathlib import Path


def call(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        body = response.read()
        return json.loads(body) if body else None


def rpc(url, method):
    return call(
        url,
        "/collective_rpc",
        {"method": method, "args": [], "timeout": 900},
    )["results"]


def completion(url, ids, count, params, ignore_eos=False):
    return call(
        url,
        "/v1/completions",
        {
            "model": "deepseek-v4.1-flash-fl",
            "prompt": ids,
            "temperature": 0,
            "max_tokens": count,
            "ignore_eos": ignore_eos,
            "return_token_ids": True,
            "kv_transfer_params": params,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    baseline = json.loads(args.baseline.read_text())
    candidates = {
        "short_0": (prompts["short"][0], 12, False, baseline["cases"]["short_0"]),
        "short_1": (prompts["short"][1], 12, False, baseline["cases"]["short_1"]),
        "window_prefill": (
            prompts["boundary"],
            12,
            True,
            baseline["cases"]["window_prefill"],
        ),
    }
    chosen = args.case or list(candidates)
    report = {
        "status": "running",
        "prefill_url": args.prefill_url,
        "decode_url": args.decode_url,
        "cases": {},
    }

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    try:
        for url in (args.prefill_url, args.decode_url):
            call(url, "/health")
        report["before"] = {
            "prefill": rpc(args.prefill_url, "fl_pd_stats"),
            "decode": rpc(args.decode_url, "fl_pd_stats"),
        }
        for name in chosen:
            ids, count, ignore_eos, expected = candidates[name]
            transfer_id = "fl-pd-" + uuid.uuid4().hex
            p_response = completion(
                args.prefill_url,
                ids,
                1,
                {"do_remote_decode": True, "transfer_id": transfer_id},
                ignore_eos,
            )
            p_choice = p_response["choices"][0]
            p_tokens = p_choice["token_ids"]
            if len(p_tokens) != 1:
                raise AssertionError(f"Prefill emitted {len(p_tokens)} tokens")
            if p_choice["finish_reason"] == "stop":
                tokens = p_tokens
                d_choice = None
            else:
                params = p_response.get("kv_transfer_params")
                if not params or params.get("transfer_id") != transfer_id:
                    raise AssertionError(f"Prefill omitted connector handoff: {params}")
                params = {
                    **params,
                    "do_remote_prefill": True,
                    "do_remote_decode": False,
                }
                d_response = completion(
                    args.decode_url, ids + p_tokens, count - 1, params, ignore_eos
                )
                d_choice = d_response["choices"][0]
                tokens = p_tokens + d_choice["token_ids"]
            expected_tokens = expected["mtp"]["token_ids"][:count]
            report["cases"][name] = {
                "prompt_tokens": len(ids),
                "prefill_first_token": p_tokens[0],
                "prefill_finish_reason": p_choice["finish_reason"],
                "decode_token_ids": d_choice["token_ids"] if d_choice else [],
                "stitched_token_ids": tokens,
                "expected_token_ids": expected_tokens,
                "matches_single_node": tokens == expected_tokens,
            }
            save()
            if tokens != expected_tokens:
                raise AssertionError(f"PD differs from single-node FlagCX: {name}")
        report["after"] = {
            "prefill": rpc(args.prefill_url, "fl_pd_stats"),
            "decode": rpc(args.decode_url, "fl_pd_stats"),
            "decode_mtp": rpc(args.decode_url, "fl_mtp_stats"),
        }
        for role, counter in (("prefill", "sent_pages"), ("decode", "received_pages")):
            stats = report["after"][role]
            if len(stats) != 8 or {row["rank"] for row in stats} != set(range(8)):
                raise AssertionError(f"{role} missing TP ranks")
            if not all(row["enabled"] and row[counter] >= len(chosen) for row in stats):
                raise AssertionError(f"{role} FlagCX pages did not transfer")
            if any(row["fatal_error"] for row in stats):
                raise AssertionError(f"{role} transfer failed")
        if not all(row["accepted_tokens"] > 0 for row in report["after"]["decode_mtp"]):
            raise AssertionError("Decode did not accept MTP drafts")
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
                    "cases": list(report["cases"]),
                    "error": report.get("error"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
