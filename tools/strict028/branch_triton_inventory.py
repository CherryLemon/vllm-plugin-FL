#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inventory branch-authored Triton against its last merged upstream parent."""

import argparse
import ast
import json
import subprocess
from pathlib import Path

UPSTREAM = "3918f3c5a3"
BRANCH = "a9e3d217cce075c77a8041d14dd822307953735e"
GEMS_BEFORE = "313b4fdd5a2118debcae17089cfb36bee6cc9cea"


def git(repo, *args, optional=False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if not optional:
        r.check_returncode()
    return r.stdout


def functions(source):
    result = {}

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            named = isinstance(child, (ast.FunctionDef, ast.ClassDef))
            name = prefix + child.name if named else prefix
            if isinstance(child, ast.FunctionDef) and any(
                ast.unparse(d).startswith("triton.jit") for d in child.decorator_list
            ):
                result[name] = (ast.dump(child, include_attributes=False), child.lineno)
            visit(child, name + "." if named else prefix)

    visit(ast.parse(source or ""))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--branch", type=Path, required=True)
    p.add_argument("--gems", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    git(a.branch, "merge-base", "--is-ancestor", UPSTREAM, BRANCH)
    paths = git(
        a.branch, "diff", "--name-only", UPSTREAM, BRANCH, "--", "vllm/**/*.py"
    ).splitlines()
    rows = []
    for path in paths:
        old = functions(git(a.branch, "show", f"{UPSTREAM}:{path}", optional=True))
        new = functions(git(a.branch, "show", f"{BRANCH}:{path}"))
        for name, (body, line) in new.items():
            if name in old and body == old[name][0]:
                continue
            if path.endswith("sm90_fp4_indexer.py"):
                fg = "src/flag_gems/fused/DSA/mxfp4_mqa_logits.py"
                fg_name = name.replace("_sm90_fp4", "_mxfp4")
            elif path.endswith("candidate_blocks.py"):
                fg = "src/flag_gems/fused/DSA/finalize_candidate_topk.py"
                fg_name = name.replace("_sm90", "")
            elif path.endswith("sm90_static.py"):
                fg = "src/flag_gems/runtime/backend/_nvidia/hopper/ops/w8a8_block_fp8_matmul_static.py"
                fg_name = name
            elif path.endswith("fused_indexer_q.py"):
                fg = "src/flag_gems/fused/fused_indexer_q_rope_quant.py"
                fg_name = name
            else:
                raise ValueError(f"unmapped branch Triton: {path}:{name}")
            target = functions((a.gems / fg).read_text())
            previous = functions(
                git(a.gems, "show", f"{GEMS_BEFORE}:{fg}", optional=True)
            )
            if fg_name not in target:
                raise AssertionError(f"missing FlagGems implementation: {fg}:{fg_name}")
            entry = name.endswith("_kernel") or name in (
                "_w8a8_block_fp8_matmul_hopper_static",
                "_reduce_block_fp8_split_k",
            )
            rows.append(
                {
                    "source": path,
                    "function": name,
                    "line": line,
                    "kind": "launch_kernel" if entry else "device_helper",
                    "branch_change": "new" if name not in old else "modified",
                    "flaggems": fg,
                    "flaggems_function": fg_name,
                    "flaggems_line": target[fg_name][1],
                    "delivery_change": "new"
                    if fg_name not in previous
                    else (
                        "modified"
                        if target[fg_name][0] != previous[fg_name][0]
                        else "already_present"
                    ),
                    "whole_model_dispatch": False,
                }
            )
    report = {
        "branch_commit": BRANCH,
        "upstream_parent": git(a.branch, "rev-parse", UPSTREAM).strip(),
        "flaggems_commit": git(a.gems, "rev-parse", "HEAD").strip(),
        "flaggems_before": GEMS_BEFORE,
        "functions": rows,
        "launch_kernels": sum(r["kind"] == "launch_kernel" for r in rows),
        "device_helpers": sum(r["kind"] == "device_helper" for r in rows),
        "scope": "branch delta, excluding merged upstream Triton, TileLang, CUDA and Python orchestration",
    }
    assert len(rows) == 14 and report["launch_kernels"] == 6
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("launch_kernels", "device_helpers", "flaggems_commit")
            }
        )
    )


if __name__ == "__main__":
    main()
