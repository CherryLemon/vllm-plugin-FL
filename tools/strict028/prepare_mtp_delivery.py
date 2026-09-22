#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare an immutable source/image candidate; final API acceptance seals it."""

import argparse
import json
import shutil
from pathlib import Path

from prepare_delivery import GEMS_BASE, HOST, IMAGE_ID, PLUGIN_BASE, git, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--plugin", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    work, out = a.work.resolve(), a.output.resolve()
    sources = {
        "vllm": work / "vllm",
        "plugin": a.plugin.resolve(),
        "FlagGems": work / "FlagGems",
    }
    commits = {}
    for name, repo in sources.items():
        if git(repo, "status", "--porcelain", "--untracked-files=normal").strip():
            raise ValueError(f"commit source changes first: {repo}")
        commits[name] = git(repo, "rev-parse", "HEAD").decode().strip()
    assert commits["vllm"] == HOST
    if out.exists():
        raise FileExistsError("choose a new delivery path")
    versions = {
        "plugin": "0.4.0.dev0+g" + commits["plugin"][:7],
        "FlagGems": "5.4.0.dev0+g" + commits["FlagGems"][:7],
    }
    for d in (
        "sources",
        "patches",
        "reports",
        "commands",
        "evidence",
        "image/wheels",
        "build-deps",
    ):
        (out / d).mkdir(parents=True)
    hashes = {}
    for name, repo in sources.items():
        (out / "sources" / f"{name}.tar.gz").write_bytes(
            git(repo, "archive", "--format=tar.gz", commits[name])
        )
        for path in git(repo, "ls-files", "-z").decode().split("\0"):
            file = repo / path
            if path and file.is_file() and not file.is_symlink():
                hashes[f"{name}/{path}"] = sha(file)
    for name, base in (("plugin", PLUGIN_BASE), ("FlagGems", GEMS_BASE)):
        (out / "patches" / f"{name}.patch").write_bytes(
            git(sources[name], "diff", "--binary", base, commits[name])
        )
    prefixes = (
        "vllm-0.28.0+empty-",
        f"vllm_plugin_fl-{versions['plugin']}-",
        f"flag_gems-{versions['FlagGems']}-",
    )
    for prefix in prefixes:
        files = list((work / "wheels").glob(prefix + "*.whl"))
        assert len(files) == 1, prefix
        shutil.copy2(files[0], out / "image/wheels" / files[0].name)
    for file in (work / "build-deps").glob("*.whl"):
        shutil.copy2(file, out / "build-deps" / file.name)
        if file.name.lower().startswith(
            (
                "pyyaml-6.0.1-",
                "packaging-26.3-",
                "sqlalchemy-2.0.54-",
                "greenlet-3.5.6-",
                "typing_extensions-4.16.0-",
            )
        ):
            shutil.copy2(file, out / "image/wheels" / file.name)
    source_manifest = {
        "vllm_commit": HOST,
        "source_commits": commits,
        "source_sha256": hashes,
    }
    (out / "source-manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n"
    )
    manifest = {
        "status": "candidate_pending_image_acceptance",
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
        },
        "package_versions": versions,
        "model": {
            "path": "/public-nvme/models/DeepSeek-V4.1-Flash",
            "checkpoint_read_only": True,
            "method": "dspark",
            "num_speculative_tokens": 5,
            "mtp_layers": 3,
            "tp": 8,
            "max_model_len": 256,
            "max_num_seqs": 2,
            "verification": "serial greedy",
            "pd": False,
        },
        "operator_optimization_handoff": {
            "new_operators_added": True,
            "branch_launch_kernels": 6,
            "branch_device_helpers": 8,
            "whole_model_uses_branch_optimized_kernels": False,
            "report": "reports/mtp-and-triton.md",
            "performance_status": "not_profiled",
            "target_teams": ["FlagGems", "FlagTree"],
        },
        "not_validated": [
            "PD",
            "non-NVIDIA",
            "parallel target verification",
            "model CUDA graph",
            "long context",
            "performance SLO",
        ],
    }
    for path in (out / "manifest.json", out / "image/deployment-manifest.json"):
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name in (
        "build_image.sh",
        "build_wheels.sh",
        "provision_build_env.sh",
        "serve.sh",
        "serve_mtp.sh",
        "audit_install.py",
        "smoke_mtp.py",
        "smoke_mtp_api.py",
        "branch_triton_inventory.py",
    ):
        shutil.copy2(a.plugin / "tools/strict028" / name, out / "commands" / name)
    shutil.copy2(a.plugin / "tools/strict028/Dockerfile", out / "image/Dockerfile")
    shutil.copy2(
        a.plugin / "docs/strict028-mtp-and-triton.md", out / "reports/mtp-and-triton.md"
    )
    # Keep the earlier trial labelled as a trial; the exact image is tested next.
    trial = out / "evidence/development-trial"
    shutil.copytree(work / "evidence/mtp", trial)
    shutil.copy2(
        work / "evidence/vllm-empty-build.log", out / "evidence/vllm-empty-build.log"
    )
    (out / "README.md").write_text("""# DeepSeek V4.1 Flash FL / MTP

运行条件：同机构成 TP=8 的 H100 80GB，模型目录只读。
版本、镜像 ID、验收状态见 manifest.json、evidence/image-build.json 和 reports/acceptance.md。
算子清单及接入边界见 reports/mtp-and-triton.md。

```bash
sha256sum -c SHA256SUMS
export FL_MODEL_PATH=/public-nvme/models/DeepSeek-V4.1-Flash
export FL_RUNTIME_IMAGE=<evidence/image-build.json 中的 image_id>
bash commands/serve_mtp.sh
```

服务绑定 127.0.0.1:8000，使用贪心生成、三层 DSpark 和五个草稿 token。
`commands/serve.sh` 可启动关闭 MTP 的基线。停止本任务的容器即可释放 GPU；旧交付和旧镜像仍可回滚。

`sources/` 是冻结源码，`patches/` 含可重建插件/FlagGems tree 的独立补丁；宿主没有补丁。
`image/wheels/` 是镜像正常安装的 wheel，`build-deps/` 为前次固定的离线构建依赖。
`bash commands/build_image.sh` 使用固定的官方基础镜像 ID 重建运行镜像。
重建 wheel 时将三个源码包解包为 /work/{vllm,plugin,FlagGems}，复制 source-manifest.json
及 build-deps，先执行 provision_build_env.sh，再按 manifest.json 的版本设置
FL_PLUGIN_VERSION、FL_GEMS_VERSION 后执行 build_wheels.sh。

正式部署默认不启用开发 RPC。验收临时设置 FL_VALIDATION_RPC=1，使用官方开发端点
完成逐 rank 对比；模型没有 PD 或分支私有 vLLM 补丁。
""")
    print(
        json.dumps(
            {"delivery": str(out), "versions": versions, "status": manifest["status"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
