#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit a normal installed package, with no source-tree PYTHONPATH override."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("remove PYTHONPATH for the installed-package audit")
    import torch

    import vllm

    import vllm_fl
    from vllm_fl.strict028.bootstrap import register_models, validate_host

    validate_host()
    register_models()
    if torch.cuda.is_initialized():
        raise AssertionError("plugin registration initialized CUDA")
    for name in ("vllm._C", "vllm._C_stable_libtorch"):
        if importlib.util.find_spec(name) is not None:
            raise AssertionError(f"empty host unexpectedly contains {name}")
    from vllm import ModelRegistry
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    from vllm_fl.strict028.platform import PlatformFL028

    paths = {
        "vllm": str(Path(vllm.__file__).resolve()),
        "plugin": str(Path(vllm_fl.__file__).resolve()),
        "FlagGems": str(Path(importlib.util.find_spec("flag_gems").origin).resolve()),
    }
    for path in paths.values():
        if "site-packages/" not in path:
            raise AssertionError(f"not a normal package installation: {path}")
    manifest = json.loads(args.source_manifest.read_text())
    if manifest["vllm_commit"] != "2cf0a6915ce544dc493a0990f2ea38d81601128a":
        raise AssertionError("unexpected official host revision")
    checked = {}
    for name, prefix in (
        ("vllm", "vllm/vllm/"),
        ("plugin", "plugin/vllm_fl/"),
        ("FlagGems", "FlagGems/src/flag_gems/"),
    ):
        root = Path(paths[name]).parent
        count = 0
        for relative, expected in manifest["source_sha256"].items():
            if relative.startswith(prefix) and relative.endswith(".py"):
                installed = root / relative.removeprefix(prefix)
                if hashlib.sha256(installed.read_bytes()).hexdigest() != expected:
                    raise AssertionError(f"installed source differs: {relative}")
                count += 1
        if not count:
            raise AssertionError(f"no audited source files for {name}")
        checked[name] = count
    core_hashes = {}
    for cls in (Worker, GPUModelRunner):
        module = sys.modules[cls.__module__]
        if not cls.__module__.startswith("vllm."):
            raise AssertionError("host class origin changed")
        file = Path(module.__file__)
        core_hashes[str(file.relative_to(Path(vllm.__file__).parent))] = hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
    legacy = [name for name in sys.modules if name.startswith("vllm_fl.patches.")]
    if legacy:
        raise AssertionError(f"legacy patch modules loaded: {legacy}")
    if "DeepseekV41FlashFLForCausalLM" not in ModelRegistry.get_supported_archs():
        raise AssertionError("FL model was not registered")
    methods = {
        (cls, name): value
        for cls in (Worker, GPUModelRunner)
        for name, value in vars(cls).items()
        if callable(value)
    }
    import flag_gems

    for (cls, name), value in methods.items():
        if vars(cls).get(name) is not value:
            raise AssertionError(f"FlagGems import replaced {cls.__name__}.{name}")
    report = {
        "status": "passed",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["vllm", "vllm-plugin-fl", "flag-gems", "torch", "triton"]
        },
        "origins": paths,
        "host_source_sha256": core_hashes,
        "installed_source_files_verified": checked,
        "source_manifest_sha256": hashlib.sha256(
            args.source_manifest.read_bytes()
        ).hexdigest(),
        "flag_gems_version": flag_gems.__version__,
        "host_worker_methods_unchanged_on_flaggems_import": True,
        "device_name": PlatformFL028.get_device_name(),
        "cuda_initialized_by_registration": False,
        "legacy_patches_imported": legacy,
        "native_extensions_present": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
