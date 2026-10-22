# GLM-5.3 Flash indexer precision

The GLM indexer selects its activation and cache format from
`current_platform.supports_fp8()` before cache allocation and compiler tracing.
NVIDIA SM89+ supports native FP8; A100/SM80 and unknown capabilities select BF16.
Moore Threads MP31+ and GCU capability 4+ retain their native FP8 path.
The native DeepGEMM candidate additionally requires NVIDIA SM90+ and its ABI.

| FP8-dependent path | Platform without FP8 | Existing implementation reused |
|---|---|---|
| Query Hadamard + FP8 quantization | Hadamard BF16 output, unit scale | FlagGems-vllm `hadamard128` |
| Per-token key quantization/cache | Store BF16 keys | `qsa_store_cache_rows` |
| Prefill K-pool compression/cache | FP32 softmax/reduction, BF16 materialization and rotation/cache | PyTorch softmax/sum; `hadamard128`, `qsa_store_cache_rows` |
| Decode K-pool tail update/cache | Ordered BF16 tail writes and compression | Same existing primitives; paged tail state retained |
| Gather quantized keys/scales | Gather BF16 pages, unit scale | PyTorch gather/indexing/copy |
| Prefill FP8 MQA logits | BF16 Q/K GEMM, FP32 weighted ReLU reduction | FlagGems-vllm `router_gemm` (BF16 x BF16 -> FP32), PyTorch ReLU/sum |
| Decode FP8 paged MQA logits | Read BF16 pages, then the same BF16 GEMM/reduction | Same existing primitives, respecting page padding and valid lengths |

No device kernels are added. INT8 quantization/GEMM primitives exist, but the
pinned operator library has no complete matching INT8 K-pool/cache/MQA chain.
BF16 therefore avoids introducing an INT8 quantization, scale or cache ABI.
The pinned library also lacks the newer public BF16 paged-MQA API; the adapter
uses its existing BF16 GEMM rather than copying or adding a paged kernel.

The main MLA cache, KDA state, vision attention and unquantized MoE already use
BF16/FP32. FP8 checkpoint conversion is outside this change: the loader continues
to reject quantized checkpoints. The BF16 indexer covers the supported zero-RoPE,
K-pool-compressed GLM configuration; other layouts fail during validation.

The indexer cache uses 256 bytes per 128-wide BF16 key instead of 132 bytes for
FP8 plus its FP32 scale (about 1.94x before page padding). The cache spec and
profiling workspace use the selected dtype. BF16 removes quantization error but
can change near-tied TopK choices; whole-model accuracy/performance equivalence
has not been established. Composite execution may be slower than fused FP8.

CI and the GLM extra use upstream's common FlagGems-vllm pin
`6f8a48649fafd8ec811b36acf7ce65915c84a6da`, including its portable math shim.
