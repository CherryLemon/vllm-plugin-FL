# DeepSeek V4.1 / unmodified vLLM 0.28 migration

Status: implementation in progress. The current milestone validates the empty-build
boundary, plugin merge, metadata, tokenizer, streaming whole-model loader, independent Worker/Runner,
request-state cache and selected FlagGems operators. Eight-rank whole-model
generation and request-state reuse have passed; whole-graph numerical comparison
is undergoing acceptance. PD, MTP, multimodal serving and non-NVIDIA execution are
not yet validated.

## Pinned sources

- Host: vLLM `v0.28.0`, `2cf0a6915ce544dc493a0990f2ea38d81601128a`.
- Image: `vllm/vllm-openai:v0.28.0-cu129`, local image ID
  `sha256:249ed60fdd67b96db472e16f945af5aaba565b20159d192ba378035b6d136a1c`.
- Plugin base: `0.4.0-dev` at `e88e4db75ece36e482a5fcc288929d4461d53184`.
- Merge main `fd5c727fcdb607bc4354cd11384761bb5d5ecfba`, then PR 544
  `27e798a577622d872f26917105132495b229b851`; merge result `29a0edb`.
- FlagGems base: `54b28861639fc5df9367aaf3bc8d062efb4f16f7`; implementation
  `89d5917dea7bd99565427cca58e93584c5496bd5`.
- Semantic source: CherryLemon/vllm `a9e3d217cce075c77a8041d14dd822307953735e`
  and checkpoint inference code at claimed revision `dba1be0a40aa45a94ad051997016db3960a90277`.
- Official image supplies Torch `2.13.0+cu129`, Triton `3.7.1`.

## Empty-build boundary

`VLLM_FL_STRICT028=1 VLLM_PLUGINS=fl` selects a separate bootstrap/platform.
Root package import does not import Torch, vLLM or FlagGems in this mode.
Registration uses public AutoConfig, ModelRegistry, TokenizerRegistry, renderer,
loader and cache-spec APIs. The host must
report `0.28.0+empty`; the active package has no vLLM CUDA extensions.
Legacy FL patches and FlagGems patch_empty_vllm are not invoked.

The exact tag is built with `VLLM_TARGET_DEVICE=empty`. Because the official
image lacks git, wheel construction uses `SETUPTOOLS_SCM_PRETEND_VERSION=0.28.0`
against that verified source revision. This preserves the `+empty` build suffix.
No host source patch, module stub or registry dictionary write is used.

## Completed component checks

- PR 544 metadata is updated to 0.28 DCP-local slot mapping, hybrid block
  splitting, recurrent-state groups and graph ownership/lifetime.
- FlagGems LibEntry includes Triton 3.7 cached-launch argument conventions;
  process-local optional MM dispatch detects competing registrations on Torch 2.13.
- Packed MXFP4 paged/workspace Indexer: 21 H100 cases, including group-6 fallback.
- Marlin MXFP4 clamp: 8 BF16/FP16 cases. This is a separate W4A16 contract;
  it is not silently selected for the reference V4.1 expert path.
- New block-scaled low-precision Linear: 26 cases, including zero tokens,
  E2M1 nibble decoding, E4M3 activation rounding and CUDA graph replay.
- Real layer-0 expert-0 against the published TileLang reference: TP=1 and
  TP=8/rank=7 shard, 3 cases each, including active clamp; max absolute error 0.
  These are component/shard checks, not an eight-rank distributed model run.
- Tokenizer: 32 reference cases; actual local tokenizer prompt/EOS checked.
- New FP4 cache rounding and sparse attention/sink: 10 independent/graph cases;
  9 additional published TileLang cases. Quantized bytes/scales agree exactly;
  tested sparse-attention maximum absolute error is 0.0009765625.
- Plugin foundation and state regression: 123 passing tests; greedy sampling
  admission: 10 tests. FlagGems new-operator suite: 57 tests.
- Real TP=8 Eager generation: two interleaved prompts and an identical repeated
  request passed. The arithmetic answer was `2` followed by the real EOS token.
  This is a generation/state-reuse check, not a model-quality assessment.
- All eight ranks loaded 13,004 tensors each (66,153,777,416 parameter bytes
  per rank). All required checkpoint tensors are consumed; 2,401 MTP tensors
  are explicitly skipped with speculative decoding disabled.
- Loader preserves packed codes/scales and pads a 288-wide TP=8 partition to
  the FlagGems Marlin alignment of 384 without changing logical model dimensions.
- Checkpoint audit: 48 shards / 96,085 tensors / 510,286,023,000 payload bytes.
  Full 16-bit expansion would be 1,526,495,246,152 bytes before sharding and
  runtime memory. Hash coverage is config/index/shard headers, not all payloads.

## Working environment and evidence

The isolated worktree is `/root/wt/dsv41-strict028` (`feat/dsv41-strict028`).
The development container is `dsv41-fl-028`; runtime Python is
`/opt/fl-venv/bin/python`. Artifacts and logs are under
`/public-nvme/yjwu/dsv41-fl-028/evidence`. FlagGems changes are in
`/public-nvme/yjwu/dsv41-fl-028/FlagGems` (`feat/dsv41-mxfp4-indexer`).

Tests currently use source snapshots for development. Rebuilt wheels, a clean
installed-package audit, final image recipe and delivery manifest remain required
before this can be called a reproducible deployment.

## Eager reference profile

The independent `DeepseekV41FlashFLForCausalLM` architecture is selected by an
`hf_overrides` architecture overlay; the original checkpoint and its quantization
configuration remain read-only and unchanged. Use the explicit `fl_dsv41`
loader, `fl_deepseek_v41` tokenizer and strict Worker.

The graph derives structural arguments from original HF configuration. It
includes Engram and loads vision weights; the serving interface currently
accepts text only. Its principal low-precision GEMMs, cache rounding, sparse
attention and Sinkhorn call FlagGems explicitly. Other reference compositions
are enumerated by `EXECUTION_PROFILE` in the model's `ops.py`.

Routed experts use contiguous EP ownership within the homogeneous TP group;
heads/projections/embedding use TP. Huge Engram tables stream in bounded CPU
chunks. This whole-graph loader is separate from the standalone Marlin-aligned
TP expert loader above: its experts preserve the original complete width.

One opaque scheduler-owned page contains each request's windows, compressed KV,
index keys, partial compressor state and Engram history. The spec charges every
byte. Rebinding a request also republishes the current layer's index cache even
when a compression group is incomplete. This avoids references to another
request or another source layer. Prefix caching and chunked prefill are rejected.

The profile requires BF16, Eager, a single homogeneous node with TP dividing
8, and greedy sampling. DP/PP/CP, MTP, PD, LoRA, structured output, sampling
penalties and logprobs are rejected explicitly. The admission ceiling is 4096
context tokens; the current smoke configuration allocates 256 and allows two
requests. This does not establish long-context support or a performance SLO.

## Remaining integration work

Finish whole-model numerical acceptance, rebuild normal wheel
installs and audit the final runtime. Replace declared Torch compositions and
connect the packed Indexer to the graph. Bounded long-context top-k, paged
layouts, prefix/chunked-prefill, graph, PD, MTP, multimodal serving and non-NVIDIA
execution each require separate implementation and acceptance.

New operators require a FlagGems/FlagTree optimization handoff; see
[strict028-operator-handoff.md](strict028-operator-handoff.md). Current operator
performance status is **not_profiled**; correctness results do not establish
throughput, bounded long-context workspace or cross-chip portability.
