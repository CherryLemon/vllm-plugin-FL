#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Request the real-weight, fixed-position distributed CUDA Graph probe."""

import argparse
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    body = {
        "method": "fl_cuda_graph_probe",
        "args": [[0] * args.prompt_length],
        "timeout": args.timeout,
    }
    request = urllib.request.Request(
        args.url.rstrip("/") + "/collective_rpc",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report = {"status": "failed", "endpoint": args.url, "probe": body}
    try:
        with opener.open(request, timeout=args.timeout + 30) as response:
            report["results"] = json.load(response)["results"]
        rows = report["results"]
        if len(rows) != 8 or {r["rank"] for r in rows} != set(range(8)):
            raise AssertionError("graph probe did not return all TP ranks")
        if any(
            r["graph_replay_count"] != 2 or r["logits_exact_match"] != [True, True]
            for r in rows
        ):
            raise AssertionError("fixed-position graph replay differs from eager")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({
            "status": report["status"],
            "endpoint": args.url,
            "prompt_length": args.prompt_length,
            "result_ranks": len(report.get("results", [])),
            "error": report.get("error"),
        }))


if __name__ == "__main__":
    main()
