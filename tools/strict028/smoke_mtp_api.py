#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HTTP and real-weight reference acceptance of a local FL DSpark image.

The temporary validation server enables official vLLM development RPC routes.
Ordinary serve_mtp.sh deployments leave those routes disabled.
"""

import argparse
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prompts = json.loads(a.prompts.read_text())
    report = {"status": "running", "base_url": a.base_url, "pd": False}

    def save():
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def http(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            a.base_url + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            body = response.read()
            return json.loads(body) if body else None

    def rpc(method, *args):
        return http(
            "/collective_rpc",
            {"method": method, "args": [json.dumps(x) for x in args], "timeout": 900},
        )["results"]

    def complete(ids, count=24, ignore_eos=False):
        r = http(
            "/v1/completions",
            {
                "model": "deepseek-v4.1-flash-fl",
                "prompt": ids,
                "max_tokens": count,
                "temperature": 0,
                "ignore_eos": ignore_eos,
                "return_token_ids": True,
            },
        )["choices"][0]
        return {k: r[k] for k in ("text", "token_ids", "finish_reason")}

    try:
        http("/health")
        assert "deepseek-v4.1-flash-fl" in [x["id"] for x in http("/v1/models")["data"]]
        chat = {
            "model": "deepseek-v4.1-flash-fl",
            "temperature": 0,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "1加1等于几？请简短回答。"}],
            "chat_template_kwargs": {"thinking": False},
        }
        try:
            http("/v1/chat/completions", dict(chat, temperature=1))
        except urllib.error.HTTPError as e:
            assert e.code == 400 and "temperature != 0" in e.read().decode()
            report["unsupported_sampling_status"] = 400
        else:
            raise AssertionError("unsupported sampling was admitted")
        rpc("fl_set_drafting", False)
        with ThreadPoolExecutor(2) as pool:
            report["without_mtp"] = list(pool.map(complete, prompts["short"]))
        rpc("fl_set_drafting", True)
        with ThreadPoolExecutor(2) as pool:
            report["with_mtp"] = list(pool.map(complete, prompts["short"]))
        report["greedy_equal"] = report["without_mtp"] == report["with_mtp"]
        report["repeat_equal"] = complete(prompts["short"][0]) == report["with_mtp"][0]
        report["eos"] = [http("/v1/chat/completions", chat) for _ in range(2)]
        assert all(
            r["choices"][0]["message"]["content"].strip() == "2"
            and r["choices"][0]["finish_reason"] == "stop"
            for r in report["eos"]
        )
        save()
        cases = {
            "window_prefill": prompts["boundary"],
            "window_rollover": prompts["boundary"][:120] + prompts["boundary"][-4:],
            "context_limit": (prompts["boundary"] * 2)[:250],
        }
        report["boundary_cases"] = {}
        for name, ids in cases.items():
            count = min(12, 256 - len(ids))
            rpc("fl_set_drafting", False)
            ref = complete(ids, count, True)
            rpc("fl_set_drafting", True)
            actual = complete(ids, count, True)
            report["boundary_cases"][name] = {
                "input_tokens": len(ids),
                "count": count,
                "equal": ref == actual,
                "without_mtp": ref,
                "with_mtp": actual,
            }
            save()
        report["stats"] = rpc("fl_mtp_stats")
        save()
        for key, method in (
            ("target_reference", "fl_reference_differential"),
            ("mtp_reference", "fl_mtp_differential"),
        ):
            report[key] = []
            for ids in (prompts["short"][0], prompts["boundary"]):
                report[key].append(rpc(method, ids))
                save()
        stats = [{k: v for k, v in r.items() if k != "rank"} for r in report["stats"]]
        assert all(s == stats[0] for s in stats)
        assert stats[0]["accepted_tokens"] > 0
        assert report["greedy_equal"] and report["repeat_equal"]
        assert all(r["equal"] for r in report["boundary_cases"].values())
        for case in report["target_reference"] + report["mtp_reference"]:
            assert {r["rank"] for r in case} == set(range(8))
            assert all(r["passed"] for r in case)
        http("/health")
        report["status"] = "passed"
    except Exception as e:
        report.update(status="failed", error=repr(e))
        raise
    finally:
        save()
        print(
            json.dumps(
                {
                    k: report.get(k)
                    for k in (
                        "status",
                        "greedy_equal",
                        "repeat_equal",
                        "stats",
                        "error",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
