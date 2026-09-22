#!/usr/bin/env bash
# Offline setup in the official 0.28.0-cu129 image. Populate build-deps from
# the delivery wheelhouse first; no network or package upgrade is implicit.
set -euo pipefail
task_root="${FL_BUILD_ROOT:-/work}"
python3 -m venv --system-site-packages /opt/fl-venv
uv pip install --python /opt/fl-venv/bin/python --no-index --no-deps \
  --find-links "$task_root/build-deps" \
  'setuptools-rust==1.13.0' 'semantic_version==2.10.0' \
  'scikit-build-core==0.11.0' 'pybind11==3.1.0' \
  'PyYAML==6.0.1' 'packaging==26.3' 'SQLAlchemy==2.0.54' \
  'greenlet==3.5.6' 'typing_extensions==4.16.0' 'pathspec==1.1.1' \
  'build==1.6.1' 'pyproject_hooks==1.3.3'
# FlagGems' declared build bounds differ from vLLM's. Keep the build tools in
# separate environments, preserving the official image's runtime setuptools.
python3 -m venv --system-site-packages /opt/gems-build
uv pip install --python /opt/gems-build/bin/python --no-index --no-deps \
  --find-links "$task_root/build-deps" \
  'setuptools==76.1.0' 'setuptools-scm==9.2.2' 'wheel==0.46.2' \
  'build==1.6.1' 'pyproject_hooks==1.3.3' 'packaging==26.3'
