#!/usr/bin/env bash
# Build inside the pinned official vLLM 0.28.0-cu129 image.
# Inputs: /work/{vllm,plugin,FlagGems,build-deps,source-manifest.json}.
# build-deps is the offline dependency wheelhouse recorded in the manifest.
set -euo pipefail
task_root="${FL_BUILD_ROOT:-/work}"
host_python="${FL_HOST_BUILD_PYTHON:-/opt/fl-venv/bin/python}"
gems_python="${FL_GEMS_BUILD_PYTHON:-/opt/gems-build/bin/python}"

"$host_python" - "$task_root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
manifest=json.loads((root/'source-manifest.json').read_text())
assert manifest['vllm_commit']=='2cf0a6915ce544dc493a0990f2ea38d81601128a'
for relative, expected in manifest['source_sha256'].items():
    actual=hashlib.sha256((root/relative).read_bytes()).hexdigest()
    if actual!=expected:
        raise RuntimeError(f'Source integrity mismatch: {relative}')
PY
mkdir -p "$task_root/wheels"
cd "$task_root/vllm"
VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.28.0 \
  uv build --wheel --no-build-isolation --python "$host_python" --out-dir "$task_root/wheels"
cd "$task_root/FlagGems"
SETUPTOOLS_SCM_PRETEND_VERSION="${FL_GEMS_VERSION:?set the pinned FlagGems artifact version}" \
  uv build --wheel --no-build-isolation --python "$gems_python" --out-dir "$task_root/wheels"
cd "$task_root/plugin"
SETUPTOOLS_SCM_PRETEND_VERSION="${FL_PLUGIN_VERSION:?set the pinned plugin artifact version}" \
  uv build --wheel --no-build-isolation --python "$host_python" --out-dir "$task_root/wheels"
