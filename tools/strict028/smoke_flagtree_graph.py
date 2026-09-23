#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that the pinned FlagTree compiler executes a FlagGems kernel in CUDA Graph."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import torch
import triton
from flag_gems.fused.act_quant import act_quant_triton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if triton.__version__ != "3.7.1" or not triton.__file__.startswith(
        "/opt/flagtree/"
    ):
        raise RuntimeError(f"FlagTree is not active: {triton.__file__}")
    x = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
    act_quant_triton(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        y, scale = act_quant_triton(x)
    exact_match = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x))
        graph.replay()
        ref_y, ref_scale = act_quant_triton(x)
        exact_match.append(
            bool(torch.equal(y, ref_y) and torch.equal(scale, ref_scale))
        )
    report = {
        "compiler": triton.__file__,
        "compiler_api_version": triton.__version__,
        "flagtree_distribution_version": version("flagtree"),
        "vllm_distribution_version": version("vllm"),
        "graph_replay_count": len(exact_match),
        "exact_match": exact_match,
        "gpu": torch.cuda.get_device_name(),
        "scope": "FlagGems act_quant kernel only; no serving graph",
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not all(exact_match):
        raise AssertionError("FlagTree graph replay differs from eager FlagGems")


if __name__ == "__main__":
    main()
