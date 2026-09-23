# DeepSeek V4.1 Flash steady Decode validation

Status at 2026-09-23 13:22 UTC: the target Decode topology and capacity have
passed real-weight CUDA Graph differential checks on all eight ranks.
Chunked Prefill is still under numerical validation. No target-workload
throughput has been measured yet.

## Workload and measurement

The reference is `/root/sglang-plugin-FL/docker/dsv41/PROFILE_STEADY_DECODE.md`
and `FLAGCX_PD.md`: 80 active requests, each with exactly 131072 input tokens
and 8192 generated tokens, greedy sampling and `ignore_eos=True`. The coding
prompt hash matches the reference. Prefill uses TP8/EP8 on `.13`; Decode uses
attention TP2 × DP4 and global EP8 on eight H100 80GB cards on `.68`.
DSpark proposes five tokens. Decode must use actual CUDA Graph replay.

`benchmark_pd_decode.py` defaults to this workload. Smaller functional runs
require `--smoke` and cannot be reported as comparable performance. Requests
are routed evenly across the four Decode groups. Prefill and FlagCX handoff
are excluded from the steady rate. The common measured interval requires
20 running requests per group, zero waiting, stable preemption counts, and
continuous metric samples. The report includes aggregate and per-request
rates for this interval, plus complete streaming token counts.

Run unprofiled throughput and profiling separately. `--profile-label LABEL`
requires one target round and arms CPU+CUDA profiling after five steady
occupancy samples. Every rank records ten Decode steps with stack and shape
information. Validate the eight traces and receipts with:

```bash
python tools/strict028/summarize_decode_profiles.py \
  /public-nvme/yjwu/dsv41-fl-028/campaign-steady/profiles/LABEL \
  --output /public-nvme/yjwu/dsv41-fl-028/campaign-steady/profiles/LABEL-summary.json
```

The summary checks actual launch counts against per-step replay counters.
It reports kernel sums, overlapping-kernel union time and GPU span separately;
these are not interchangeable with unprofiled token throughput.

## Implemented deployment

The base is `vllm/vllm-openai:v0.28.0-cu129`, with official vLLM commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a` installed as `0.28.0+empty`.
The host source is unchanged. Runtime wheels use FlagGems `b459958`, FlagTree
`dbf184230982e2f7cbe6b91fa3ca1ea069833d21`, and FlagCX
`648a6c489d2d54870d173926eda2d0ea0771b39d`. Each image stage contains a
`deployment-manifest.json` with wheel/library hashes.

Decode plugin `18f3d9b` admits max length 139264 and 20 requests per DP group.
It pre-captures one fixed bucket of 20 lanes for target and draft; positions,
page IDs and lane activity are device inputs. FlagGems kernels access each
request's state directly, without whole-page copies during replay.
The official host's `--enforce-eager` disables its own graph manager; the FL
runner explicitly captures and replays its independent graphs.

Verification currently advances the proposed positions sequentially:
one to six target replays plus one draft replay per step. This is the
implementation to measure, and differs from the reference SGLang trace's
two launches per step. Do not equate their per-step kernel counts.

Each rank holds 21 state pages of 452419584 bytes (including the inactive
page). Target-capacity startup and Graph capture succeeded. Physical free
memory after numerical checks was about 212 MiB on rank 0, with additional
unused Torch reserve. Live transfer/request/profiling peaks remain to be
validated. Differential probes use CPU snapshots to avoid cloning several
GiB of live state on the GPU.

Prefill has four GPU slots and 80 independent pinned-host snapshot slots per
rank. A completed prefix is copied to a retained host slot before its GPU slot
is released. FlagCX protocol 2 maps each TP2 consumer to the appropriate TP8
producer lane and releases unused producer copies. This permits all 80
prefixes to be prepared without retaining 80 GPU pages on Prefill.

## Evidence and remaining acceptance

Artifacts are under `/public-nvme/yjwu/dsv41-fl-028/campaign-steady/`.

- `benchmark/real-target-batch-18f3d9b.json`: all eight ranks passed two real
  Graph replay passes, with exact target logits/hidden/IDs and persistent
  state, and exact DSpark IDs/logits/confidence/state.
- `benchmark/dp-b16aa3d-c1-repeat-v2.json` and `dp-b16aa3d-c8-routed.json`:
  small functional PD runs passed repeated greedy requests and explicit
  routing across four DP groups. All ranks replayed Graphs, and producer
  snapshots were released. These are not target performance results.
- `benchmark/reference-prompt-crosscheck.json`: reference prompt hash and
  131072-token lengths verified for multiple request variants.
- `benchmark/real-chunked-18f3d9b-128-32.json` and
  `real-chunked-18f3d9b-8192-4096.json`: chunked Prefill failed the existing
  0.03 relative-RMS threshold, despite equal top-1 tokens. Earliest remaining
  divergence is in FP32 router projection geometry; isolated real-weight
  evidence is in `kernel/gate-geometry-real-weights.json`.
- Prefill candidate `b341282` preserves complete-prefix FP32 projection
  geometry and long-prefix mHC slicing. Five CUDA tests passed. Real-weight
  deployment validation is pending; do not treat this candidate as admitted.

Next acceptance is corrected real-weight Prefill, a full-length PD functional
run, then the exact C80 unprofiled workload and a separate eight-rank profile.
Detailed chronological evidence is in `humanize/model-loop-checkpoint.md`,
`analysis/root-cause.md` and `history/attempts.jsonl` in the campaign directory.
