#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize the eight independent steady Decode traces."""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize_rank(directory, rank):
    receipt = json.loads((directory / f"rank{rank}.receipt.json").read_text())
    rows = receipt["rows"]
    if receipt["rank"] != rank or receipt["steps"] != 10 or len(rows) != 10:
        raise ValueError(f"rank {rank}: expected ten completed steps")
    if receipt["required_active"] != 20 or any(
        r["active"] != 20
        or r["min_position"] < 131072
        or r["max_position"] >= 139264
        or not 1 <= r["target_replays"] <= 6
        or r["draft_replays"] != 1
        for r in rows
    ):
        raise ValueError(f"rank {rank}: profile does not satisfy the target workload")
    with gzip.open(directory / f"rank{rank}.json.gz", "rt") as source:
        events = json.load(source)["traceEvents"]
    expected_launches = sum(r["target_replays"] + r["draft_replays"] for r in rows)
    launches = sum(
        e.get("ph") == "X"
        and e.get("cat") == "cuda_runtime"
        and e.get("name", "").startswith("cudaGraphLaunch")
        for e in events
    )
    markers = [
        e for e in events
        if e.get("ph") == "X" and e.get("name", "").startswith("fl_decode_step_")
    ]
    if launches != expected_launches or len(markers) != 10:
        raise ValueError(
            f"rank {rank}: trace has {launches}/{expected_launches} graph launches "
            f"and {len(markers)}/10 step markers"
        )
    kernels = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    intervals = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        row = kernels[event["name"]]
        row["calls"] += 1
        row["gpu_ms"] += event["dur"] / 1000
        intervals.append((event["ts"], event["ts"] + event["dur"]))
    if not intervals:
        raise ValueError(f"rank {rank}: trace contains no GPU kernels")
    intervals.sort()
    begin, end = intervals[0]
    union = 0
    for lo, hi in intervals[1:]:
        if lo > end:
            union += end - begin
            begin, end = lo, hi
        else:
            end = max(end, hi)
    union += end - begin
    return dict(
        rank=rank,
        steps=10,
        graph_launches=launches,
        target_replays=sum(r["target_replays"] for r in rows),
        draft_replays=sum(r["draft_replays"] for r in rows),
        kernel_calls=len(intervals),
        kernel_sum_ms_per_step=sum(k["gpu_ms"] for k in kernels.values()) / 10,
        gpu_busy_union_ms_per_step=union / 10000,
        gpu_span_ms_per_step=(max(hi for _, hi in intervals) - intervals[0][0]) / 10000,
        kernels=sorted(
            [dict(name=name, **values, gpu_ms_per_step=values["gpu_ms"] / 10)
             for name, values in kernels.items()],
            key=lambda k: k["gpu_ms"], reverse=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks = [summarize_rank(args.directory, rank) for rank in range(8)]
    keys = ("kernel_sum_ms_per_step", "gpu_busy_union_ms_per_step", "gpu_span_ms_per_step")
    result = dict(
        passed=True,
        scope="10 profiled Decode steps per rank; use a separate unprofiled run for TPS",
        note="Kernel sums include overlap; sequential DSpark verification has 1–6 target graphs plus one draft graph per step.",
        directory=str(args.directory.resolve()),
        median_across_ranks={key: statistics.median(r[key] for r in ranks) for key in keys},
        ranks=ranks,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["median_across_ranks"]))


if __name__ == "__main__":
    main()
