# FlagOS new-operator optimization handoff

New operators added: **true**. Components: vllm-plugin-FL and FlagGems.
Follow-up teams: FlagGems and FlagTree. All performance status below is
**not_profiled**. Source revisions and patches are recorded by the deployment
manifest. No optimization owner or external tracking issue has been assigned.

| Operator | Source in FlagGems | Correctness evidence | Runtime use | Requested work |
|---|---|---|---|---|
| `block_scaled_lowp_linear` | `fused/block_scaled_lowp_linear.py` | 26 operator cases; real expert TP1 and TP8/rank7 against published TileLang, 3 cases each, zero error | Explicit call in FL reference dense/MoE projections | Grouped expert execution, tiling, weight reuse and native FP8 dispatch where supported |
| `mxfp4_paged_index_logits`, `mxfp4_workspace_index_logits` | `fused/DSA/mxfp4_mqa_logits.py` | 21 mask/layout/group-6 cases | Standalone operator; not yet selected by the Eager graph | Bounded workspace and fused top-k; backend-specific launches |
| `fp4_quantize_reference` | `fused/dsv41_reference_ops.py` | Independent packing/rounding cases and published TileLang byte equality for both scale formats | Explicit indexer and compressed-KV rounding | Fused RoPE, packing and cache writes; non-NVIDIA conversion lowering |
| `sparse_attention_with_sink` | `fused/dsv41_reference_ops.py` | Independent Torch and published TileLang, including empty rows and multiple sparse blocks | Explicit sparse attention in the Eager graph | Split reduction/parallel decode, paged storage and vendor tuning |
| `hc_split_sinkhorn_reference` | `fused/dsv41_reference_ops.py` | 12 independent/repeat cases and 3 published TileLang exact cases | Explicit four-stream mHC | Retain precise exp/div, fused affine arithmetic and butterfly reduction when optimizing |
| Marlin MXFP4 clamp extension | `fused/fused_marlin_moe.py` | 8 BF16/FP16 clamp/empty-token cases | Separate W4A16 component, not the V4.1 reference expert contract | Fuse clamp with preserved rounding; add activation-quantization/expert mapping contracts before using for V4.1 |

## Contracts and acceptance

`block_scaled_lowp_linear` preserves the original per-32 E4M3 activation
quantization and E8M0 scales. FP8 weights use 32×32 blocks; E2M1 weights use
per-row groups of 32, two adjacent K codes per byte. Each unscaled 32-wide
product uses BF16 matrix arithmetic and FP32 accumulation, followed by its
activation and weight scales. The wrapper returns BF16, FP16 or FP32 as
requested. No full-weight dequantization cache is created. Empty M and partial
N tiles are tested. This is a compatibility compute path even on H100; it does
not claim native FP8/FP4 throughput. Correctness tests include graph replay;
the whole-model profile remains Eager. Priority P0: preserve quantization and
router weighting before the second projection's activation quantization.

The packed indexer has logical head dimension 128. Q uses 64 payload bytes
plus four E8M0 exponent bytes; K pages place all row payloads before all row
scales. Masked/unreachable candidates must remain unreachable, including short
sequences and partially filled newest blocks. Group-6 reuse must satisfy the
validated per-request conditions or use the ordinary path. Output logits are
currently dense FP32 `[rows,width]`: long-context memory is **not bounded** by
top-k. Priority P0: stream/fuse top-k and validate candidate ordering and
precision before connecting it to the serving graph.

FP4 cache rounding uses E2M1 ties-to-even. The indexer uses groups of 32 with
E8M0 scales; compressed KV uses groups of 16 with E4M3 scales. Zero groups retain
a nonzero scale. Packed output is uint8; the current model uses in-place
quantize/dequantize into BF16. Packed bytes, scale bytes, ties, signed zero,
empty tensors and graph replay are covered. Priority P1: retain these exact
bytes when fusing with cache insertion.

Sparse attention takes BF16 Q `[B,S,H,D]`, shared BF16 KV `[B,N,D]`, FP32 sink
`[H]`, and int32 indices `[B,S,K]`. `-1` means an empty entry. Visibility is
already encoded in indices; there is no implicit `index <= query_row` mask.
The sink contributes to the denominator only. QK/accumulation are FP32;
unnormalized probabilities round to BF16 before PV, matching the published
reference. Workspace does not allocate a dense Q×KV matrix. D=64/128/256/512
and at most 64 local heads are admitted; tested shapes are in the evidence.
Priority P1: validate reduction/sink equivalence after split or paged rewrites.
The 16-head path preserves adjacent-pair, eight-group, then four-pair summation.
PV accumulates directly into the rescaled FP32 accumulator across sparse blocks.
Precise exponentials/division and the published sink normalization are required:
even a single BF16 rounding difference can affect subsequent quantization.

The original Hopper atomic FP32 Split-K path caused repeat-request differences.
Worker initialization enables deterministic algorithms and a cuBLAS workspace.
Unquantized mHC/compressor/head projections explicitly retain Torch `F.linear`:
both FP32 reduction changes and BF16 compressor rounding were observed to amplify
through the graph. These projections are pending FlagGems integration. The
quantized dense/expert projections remain on `block_scaled_lowp_linear`.

The Marlin extension preserves its existing W4A16 contract. It does not acquire
FP8 activation semantics merely because clamp is now available. Its Hopper
inline-PTX implementation has not become portable through this change. Priority
P2: select it only after a separately validated semantic mode is defined.

## Coverage and integration boundaries

Only H100/CUDA 12.9/Torch 2.13/Triton 3.7 execution is measured. Other chips,
FP16-only backends, graph capture of the serving graph and multimodal serving
require separate acceptance. MTP and two-host FlagCX PD correctness are
documented in `strict028-mtp-and-triton.md` and `strict028-flagcx-pd.md`;
their acceptance does not establish distributed graph correctness. Throughput
and long-context SLOs are not measured.

The FL Eager graph explicitly records Torch reference compositions for
unquantized projections, Indexer/top-k, routing, Engram, RoPE, norms/residuals, compressor softmax,
FP8-cache dequantization and vision. These are open migration items, not silent
fallbacks. The production tuning path must replace them deliberately and retain
the real-checkpoint layer/logit differential.

Non-operator work includes public model/tokenizer/renderer/loader registration,
NVML capability discovery before device binding, streaming checkpoint loading,
the custom Worker/Runner, and scheduler-owned complete request-state pages.
FlagGems LibEntry also needs the Triton 3.7 cached-launch compatibility fix.
