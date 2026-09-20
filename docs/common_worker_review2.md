# Common worker follow-up, September 20, 2026

This revision follows the review of PR544 `2d68733`. The shared metadata owner
is also used by PR455, including its packed block-table arena. After PR442
merged into main `71f6148`, this branch retains its fused multi-group kernel
under that owner. The conflict resolution preserves the reviewed explicit
stock/eager/graph policy, receipts, logical-width padding and pointer-table
invalidation. The producer is identical to the one in the measured model
fusion `7e5dc29`; PR442's final head `3035692` changes only CI/Docker files
relative to the previously tested `4e3540a`.

## Changes and failure contracts

`flaggems_runtime` owns process initialization, immutable settings and failed
initialization. The optional MM module owns captured handles and shape routing.
Disabled/excluded MM does not parse the MM threshold. Observing successful
FlagGems `lib=` registration establishes the callable and boxed handle; kernel
repr and source-package paths no longer gate installation. The diagnostic
registration-stack adapter is restricted to the tested Torch 2.11 family.
Other builds have no verified repeated-initialization ownership check.

Metadata padding clears each group's logical width, not its row stride. These
can differ when groups share packed storage. A graph/replay test protects
neighboring groups, real request rows and allocation guards. Existing receipt,
generation, request/token extent, stock fallback and InputBatch invalidation
contracts remain.

The probe copies itself to an isolated directory, supplies an explicit child
working directory and PYTHONPATH, requires installed wheel metadata, and checks
paths and SHA256 hashes returned by actual workers. It records plugin, vLLM,
Torch, FlagGems and Triton versions. Reordered batches are compared by prompt
identity within every policy as well as across policies. Shared drift across
all three policies therefore fails acceptance. `--combination` enables actual
worker FlagGems registration and shape-aware MM alongside metadata, prefix
caching, async scheduling and request reuse.

Mixed FULL has an explicit warning after global graph-policy resolution. The
probe cannot report it accepted. This does not warn for FULL_DECODE_ONLY's FULL
decode runtime, and does not imply that stock metadata fixes mixed FULL.

## Validation

The earlier per-group common wheel at runtime commit `de99118` passed 87
selected checks on H100 without skips, errors or failures. After reconciling
upstream PR442, acceptance rebuilds the fused common wheel and reruns those
checks plus the four adapted upstream policy/capture tests. It also exercises
full workers from the installed wheel, with RPC path/hash verification.

Pending final full-worker probe results. This working draft is not acceptance.

## Reproduction

Build/install a pure Python wheel (`VLLM_VENDOR=''`); generate a JSON mapping
from wheel-relative `vllm_fl/*.py` paths to SHA256. From an isolated directory,
run the committed probe with an explicit wheel installation and manifest:

```bash
python /path/to/tests/e2e_tests/inference/common_metadata_probe.py \
  --output-dir /artifacts/common-decode \
  --installed-root /opt/installed-common \
  --expected-runtime-manifest /artifacts/common_runtime_manifest.json \
  --dependency-path /opt/FlagGems/src \
  --cudagraph-mode FULL_DECODE_ONLY
```

Use a fresh output directory with `--cudagraph-mode PIECEWISE`, then another
with `--cudagraph-mode FULL_DECODE_ONLY --combination`. The fixture checkpoint
is generated locally with a fixed seed; there are no model downloads. Output
artifacts include the checkpoint, per-policy logs, tokens/logprobs and actual
worker identity/counters. Failed output directories must be retained rather
than overwritten.

These are small full-worker regressions, not a quality evaluation or a general
performance claim. PIECEWISE still processes a fixed maximum request extent;
large request/context/group-capacity costs need separate measurement. Other
accelerator vendors, full CI and mixed FULL numerical correctness are not
certified by these results.
