# GLM5-Next vLLM 0.24 review iteration

This iteration addresses the review of `af0e829e034ddab5f647ab6ffbbd649812ed95ad`.
The supported loading contract is an unquantized checkpoint, PP1, no speculative
decoding, and no EPLB. TP and ordinary EP remain available. Quantized checkpoint
storage is distinct from the FP8 indexer KV cache; the latter remains supported.

| Review finding | Result |
| --- | --- |
| Portable MLA logger lacks `info_once` | Use the vLLM logger; exercise active-plan selection in fresh processes. |
| GLM provider influences generic MLA dispatch | Generic FlagGems candidate declines MLA; only the GLM plan selects the portable implementation. Tests cross the real dispatcher and vendor/FlagGems methods. |
| Incomplete mHC PP state | Reject PP > 1 before hybrid config adaptation and again at final validation/model construction. |
| KDA speculative state and flattened KPool request grouping | Reject every speculative configuration at the same boundaries; no speculative support is claimed. |
| FP8 projection loading and KDA quantization exclusion | Reject quantization configs and undeclared non-floating/FP8 checkpoint tensors before ordinary loading. FP8 checkpoint conversion is deferred. |
| Incomplete EPLB outer-model interface | Reject EPLB; ordinary EP is a separate mode. |
| Missing packed quantization mapping | Declare GLM gate/up mapping on both public model wrappers; this does not enable quantized loading. |
| Parameter-name-only weight audit | Additionally record successful destination/shard/local-expert loader calls; reject missing/duplicate packed slices and restore loaders on failure. Scope the audit to the complete checkpoint, including interleaved text/head/vision prefix groups. |
| MQA per-token scale interpreted as groups | Normalize exact per-vector shapes; reject incompatible scale shapes. Force missing/rejecting FlagGems fallback with N=128/512/2051. |
| Import-time public vision FlashAttention replacement | Remove all global assignments; private GLM custom op consumes FA2 output and leaves public tuple/LSE contracts untouched. |
| Whole-pool KV repacking | Read page bytes and scale offsets directly, respecting padded physical strides. |
| Existing processor tests | Correct dense-layer defaults for small configs; retain legacy last frame 295. Keep subsecond clips nonempty. |
| Integrated clamped MoE NameError | Resolve the runtime platform before both clamp and fused paths; retain clamp behavior and cover ROCm/unknown priority selection. |

Validation uses vLLM 0.24.0, Torch 2.11.0+cu129, Transformers 5.12.1 and
FlagGems 5.3.3.dev15+gf471641e5.pkgfix1. The CPU suite passed 718 tests (7 GPU
cases deselected). Installed-wheel validation passed 52 tests, including six H100 operator cases,
including updated block-table/context replay, padded page strides, and
FP16/BF16 variable-length vision attention. These checks do not by themselves
establish end-to-end multimodal quality or full-model graph support.

A wheel built from `d83b7f978338e495a475c441c510593c9377c697` also passed
BF16 TP16/EP startup with all 47 checkpoint shards and 23 targeted serving
requests: short arithmetic, mixed 12,122/14,322/18,722/23,122-token retrieval,
16 concurrent independent codes, one image and a 0.4-second video. Every
response finished with `stop` and matched its expected content; isolation
responses contained only their own request code. Both service installations
matched all 212 wheel source hashes, and neither node logged an inference
error. These are smoke checks, not a full quality or maximum-capacity test;
startup used `--skip-mm-profiling`, and full GPQA was not rerun.

The sparse MLA graph opt-in remains off. Historical GPQA and throughput
measurements belong to their original source snapshots; they are not reruns of
this iteration. Cross-vendor tests cover dispatch contracts, not accelerator
numerical parity.

## Operator handoff

| Item | Owner for follow-up | Contract and evidence | Remaining scope |
| --- | --- | --- | --- |
| `kernels/glm5_next/paged_mqa.py` | FlagGems | FP8 page-stride read, per-token FP32 scales, unchanged dot/ReLU/head reduction; `test_paged_mqa_stride.py` covers page boundaries, padding and metadata replay | Upstream page-stride/scale-offset interface; vendor numerical/performance qualification |
| `kernels/glm5_next/vision_attention.py` | FlagTree + FlagGems | Model-private composition of existing FA2, fake implementation and static sequence bound; FP16/BF16 replay tests | Verify complete vision encoder compile/graph path and dynamic image/video generation workloads |

Other architectural follow-ups remain: make the resolved provider/capability
decision explicit across plans, replace temporary MLA-constructor capability
patches, and move compressed-page layout policy out of the generic runner.
This iteration does not claim those broader restructurings or all merge gates
are complete. Existing non-GLM model serving and full multimodal quality remain
separate release checks.

## Paged MQA measurement

H100, one active request at 2,049 tokens, 32-token pages, 25 warmups and 100
CUDA-event samples of the complete eager wrapper. Baseline is the full-pool
repack at `af0e829`; candidate and baseline both pass the FP32 PyTorch reference
at `atol=rtol=2e-5`.

| Physical pages | Baseline median (us) | Candidate median (us) | Speedup | Baseline extra peak bytes | Candidate extra peak bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 79.17 | 57.68 | 1.37x | 549,888 | 9,216 |
| 4,096 | 80.32 | 58.32 | 1.38x | 17,310,720 | 9,216 |
| 16,384 | 185.22 | 57.82 | 3.20x | 69,215,232 | 9,216 |

These are operator-wrapper measurements, not model-serving speedups. The
PyTorch reference medians were 102.11/101.07/107.79 us. Compute Sanitizer was
unavailable in the validation image, so no sanitizer pass is claimed.
