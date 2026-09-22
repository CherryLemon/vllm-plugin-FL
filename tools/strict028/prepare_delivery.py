#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package committed sources, separate patches, wheels and validation receipts."""

import argparse
import datetime
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

HOST = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
PLUGIN_BASE = "e88e4db75ece36e482a5fcc288929d4461d53184"
GEMS_BASE = "54b28861639fc5df9367aaf3bc8d062efb4f16f7"
IMAGE_ID = "sha256:249ed60fdd67b96db472e16f945af5aaba565b20159d192ba378035b6d136a1c"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work, out = args.work.resolve(), args.output.resolve()
    sources = {
        "vllm": work / "vllm",
        "plugin": args.plugin.resolve(),
        "FlagGems": work / "FlagGems",
    }
    commits = {}
    for name, root in sources.items():
        if git(root, "status", "--porcelain", "--untracked-files=normal").strip():
            raise RuntimeError(f"commit source changes first: {root}")
        commits[name] = git(root, "rev-parse", "HEAD").decode().strip()
    if commits["vllm"] != HOST:
        raise RuntimeError("host must be the untouched official v0.28.0 tag")
    if out.exists():
        raise FileExistsError(
            "use a new output directory; existing deliveries are immutable"
        )
    out.mkdir(parents=True)
    for folder in (
        "sources",
        "patches",
        "reports",
        "evidence",
        "commands",
        "image/wheels",
        "build-deps",
    ):
        (out / folder).mkdir(parents=True)
    pv = "0.4.0.dev0+g" + commits["plugin"][:7]
    gv = "5.4.0.dev0+g" + commits["FlagGems"][:7]
    for prefix in ("vllm-0.28.0+empty-", f"vllm_plugin_fl-{pv}-", f"flag_gems-{gv}-"):
        wheels = list((work / "wheels").glob(prefix + "*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected exactly one built wheel: {prefix}")
        shutil.copy2(wheels[0], out / "image/wheels" / wheels[0].name)
    for wheel in (work / "build-deps").glob("*.whl"):
        shutil.copy2(wheel, out / "build-deps" / wheel.name)
        if wheel.name.lower().startswith(
            (
                "pyyaml-6.0.1-",
                "packaging-26.3-",
                "sqlalchemy-2.0.54-",
                "greenlet-3.5.6-",
                "typing_extensions-4.16.0-",
            )
        ):
            shutil.copy2(wheel, out / "image/wheels" / wheel.name)
    hashes = {}
    for name, root in sources.items():
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "archive",
                "--format=tar.gz",
                "--output",
                str(out / "sources" / f"{name}.tar.gz"),
                commits[name],
            ],
            check=True,
        )
        for relative in git(root, "ls-files", "-z").decode().split("\0"):
            path = root / relative
            if relative and path.is_file() and not path.is_symlink():
                hashes[f"{name}/{relative}"] = sha(path)
    for name, base in (("plugin", PLUGIN_BASE), ("FlagGems", GEMS_BASE)):
        (out / "patches" / f"{name}.patch").write_bytes(
            git(sources[name], "diff", "--binary", base, commits[name])
        )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "eager_reference_migration",
        "source_commits": commits,
        "source_bases": {"plugin": PLUGIN_BASE, "FlagGems": GEMS_BASE},
        "plugin_merge_inputs": {
            "main": "fd5c727fcdb607bc4354cd11384761bb5d5ecfba",
            "pr544": "27e798a577622d872f26917105132495b229b851",
        },
        "host": {
            "modified": False,
            "version": "0.28.0+empty",
            "build_device": "empty",
            "base_image": "vllm/vllm-openai:v0.28.0-cu129",
            "base_image_id": IMAGE_ID,
            "registry_digest": None,
        },
        "package_versions": {"plugin": pv, "FlagGems": gv},
        "model": {
            "path": "/public-nvme/models/DeepSeek-V4.1-Flash",
            "claimed_revision": "dba1be0a40aa45a94ad051997016db3960a90277",
            "architecture": "DeepseekV41FlashFLForCausalLM",
            "checkpoint_read_only": True,
            "payload_hash_coverage": "config, index and shard headers; not complete payload",
            "quantization_contract": {
                "expert_weights": "E2M1 K-adjacent nibble pairs; per-row K=32 UE8M0 scales",
                "fp8_weights": "E4M3; 32x32 blocks with UE8M0 scales",
                "linear_compute": "preserved per-32 E4M3 activation rounding; BF16 dot and FP32 scaled accumulation",
                "native_fp4_gemm": False,
                "native_fp8_gemm": False,
                "cache": "quantization rounding retained in BF16 request-state storage",
                "packed_indexer_connected": False,
            },
        },
        "validation": {
            "hardware": "8x NVIDIA H100 80GB",
            "profile": "fl_dsv41_eager_reference_v1",
            "configured_context": 256,
            "max_num_seqs": 2,
            "full_model": "evidence/full-model-smoke.json",
            "installed_package": "evidence/installed-audit.json",
        },
        "operator_optimization_handoff": {
            "new_operators_added": True,
            "report": "reports/operator-optimization-handoff.md",
            "performance_status": "not_profiled",
            "target_teams": ["FlagGems", "FlagTree"],
        },
        "not_validated": [
            "non-NVIDIA",
            "PD",
            "MTP",
            "multimodal serving",
            "prefix caching",
            "chunked prefill",
            "distributed graph",
            "long context",
            "formal quality evaluation",
            "performance SLO",
        ],
        "flagtree": {"modified": False},
        "flagrelease": {"requested": False, "status": "not_requested"},
    }
    required = [
        "checkpoint.json",
        "full-model-smoke.json",
        "full-model-smoke.log",
        "installed-audit.json",
        "smoke-install-audit.json",
        "smoke-source-manifest.json",
        "plugin-foundation-regression.log",
        "sampling-contract-tests.log",
        "reference-probe-tests.log",
        "flaggems-new-operators.log",
        "mxfp4-moe-clamp-tests.log",
        "real-expert-tp1.json",
        "real-expert-tp8-rank7.json",
        "published-reference-ops.json",
        "published-reference-ops.log",
        "vllm-empty-build.log",
        "plugin-final-build.log",
        "flaggems-final-build.log",
        "offline-rebuild.log",
    ]
    for name in required:
        shutil.copy2(work / "evidence" / name, out / "evidence" / name)
    smoke = json.loads((out / "evidence/full-model-smoke.json").read_text())
    audit = json.loads((out / "evidence/installed-audit.json").read_text())
    if (
        smoke["status"] != "passed"
        or not smoke.get("reference_differential")
        or audit["status"] != "passed"
    ):
        raise RuntimeError(
            "generation, whole-graph differential and normal-install audit must pass before delivery"
        )
    manifest["validation"]["reference_graph_differential"] = bool(
        smoke.get("reference_differential")
    )
    differential = smoke["reference_differential"]
    if (
        len(differential) != smoke["tp"]
        or {row["rank"] for row in differential} != set(range(smoke["tp"]))
        or any(
            not row["top1_equal"]
            or not math.isfinite(row["relative_rms"])
            or row["relative_rms"] > 0.03
            for row in differential
        )
    ):
        raise RuntimeError("distributed reference comparison is incomplete or failed")
    boundary = smoke.get("window_boundary_differential", [])
    for case in (differential, boundary):
        if {row["rank"] for row in case} != set(range(smoke["tp"])):
            raise RuntimeError("missing reference ranks or boundary case")
        for row in case:
            if (
                not row.get("passed")
                or not row.get("repeated_logits_equal")
                or len(row.get("decode", [])) != 4
            ):
                raise RuntimeError("reference repeat/decode acceptance is incomplete")
            if any(not step["passed"] for step in row["decode"]):
                raise RuntimeError("cached decode differential failed")
    if any(row["prompt_tokens"] <= 128 for row in boundary):
        raise RuntimeError("the boundary comparison did not cross the attention window")
    model_sources = json.loads(
        (out / "evidence/smoke-source-manifest.json").read_text()
    )["source_sha256"]
    model_install = json.loads((out / "evidence/smoke-install-audit.json").read_text())
    runtime_prefixes = ("vllm/vllm/", "plugin/vllm_fl/", "FlagGems/src/flag_gems/")
    current_runtime = {
        key: value
        for key, value in hashes.items()
        if key.startswith(runtime_prefixes) and key.endswith(".py")
    }
    tested_runtime = {
        key: value
        for key, value in model_sources.items()
        if key.startswith(runtime_prefixes) and key.endswith(".py")
    }
    if current_runtime != tested_runtime:
        raise RuntimeError("runtime source changed since the whole-model run; rerun it")
    manifest["validation"]["full_model_package_versions"] = model_install["packages"]
    manifest["validation"]["runtime_source_identical_to_model_test"] = True
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "image/deployment-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (out / "source-manifest.json").write_text(
        json.dumps({"vllm_commit": HOST, "source_sha256": hashes}, indent=2) + "\n"
    )
    for name in (
        "provision_build_env.sh",
        "build_wheels.sh",
        "build_image.sh",
        "smoke_generate.py",
        "audit_install.py",
        "serve.sh",
    ):
        shutil.copy2(args.plugin / "tools/strict028" / name, out / "commands" / name)
    shutil.copy2(args.plugin / "tools/strict028/Dockerfile", out / "image/Dockerfile")
    shutil.copy2(args.plugin / "tools/strict028/README.md", out / "README.md")
    shutil.copy2(
        args.plugin / "docs/strict028-migration.md", out / "reports/migration.md"
    )
    shutil.copy2(
        args.plugin / "docs/strict028-operator-handoff.md",
        out / "reports/operator-optimization-handoff.md",
    )
    (out / "image/build.env").write_text(
        f"FL_PLUGIN_VERSION={pv}\nFL_GEMS_VERSION={gv}\n"
    )
    files = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files)
    )
    print(
        json.dumps(
            {"delivery": str(out), "versions": manifest["package_versions"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
