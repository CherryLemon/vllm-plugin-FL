# Reproduce the strict vLLM 0.28 migration

This profile uses the untouched official vLLM 0.28.0 source built with
`VLLM_TARGET_DEVICE=empty`, the merged FL plugin and the companion FlagGems
changes. The base image supplies Torch and CUDA; its original compiled vLLM
installation is shadowed by the isolated `/opt/fl-venv` installation.

## Build and audit

The delivery directory contains `manifest.json`, `SHA256SUMS`, source archives,
separate plugin/FlagGems patches, an offline wheelhouse and validation receipts.
Verify `sha256sum -c SHA256SUMS` before use. The manifest pins the base image by
local image ID; a registry digest was not available in this environment.

To rebuild wheels, extract `sources/{vllm,plugin,FlagGems}.tar.gz` into directories
of those names under a build root, alongside `build-deps` and
`source-manifest.json`. Mount that root at `/work` in the pinned official image.
Run `commands/provision_build_env.sh`, then export the two artifact versions from
`image/build.env` and run `commands/build_wheels.sh`. Both scripts work offline.

The runtime build uses the supplied wheels and verifies the official base's
local image ID before building:

```bash
bash commands/build_image.sh
```

Run `commands/audit_install.py --source-manifest /work/source-manifest.json
--output /work/installed-audit.json` using `/opt/fl-venv/bin/python`, from outside
the source trees with `PYTHONPATH` unset. The audit checks every tracked Python
source file in all three installed packages against the frozen sources. It also
checks the empty-build version, absent host extensions, public model registration,
registration without CUDA initialization and absence of legacy FL patch imports.

`commands/smoke_generate.py --model /models/DeepSeek-V4.1-Flash
--output /work/full-model-smoke.json --reference-probe` exercises real weights,
two interleaved requests, request-state reuse and a distributed comparison with
the published inference graph. The checkpoint's `inference/` reference sources
and TileLang are needed for the optional reference probe only.

## Serve and roll back

Set `FL_MODEL_PATH` to the original checkpoint directory and `FL_RUNTIME_IMAGE`
to the built image ID, then run `commands/serve.sh`. It binds the API to localhost
and mounts the checkpoint read-only. The example requires eight homogeneous H100
80GB GPUs, Eager execution and greedy sampling (`temperature: 0`). Send short
text requests within the configured 256-token total context. Cold kernel
compilation runs during startup and may take several minutes.
The image allows 1,800 seconds for engine readiness to cover the 510GB checkpoint
load and cold kernel compilation; this is not a request latency target.

This is a reference integration profile. Consult `reports/migration.md` and
`reports/operator-optimization-handoff.md` for measured coverage and pending
capabilities. The deployment example does not establish an API, performance or
quality acceptance result beyond the receipts actually included in the bundle.

To roll back, stop the named `dsv41-fl-serving` container and restart the previous
deployment using its recorded image ID and launch command. No checkpoint files
or host vLLM installation are modified. Keep both image IDs until the new
deployment has passed the intended workload's acceptance checks.
