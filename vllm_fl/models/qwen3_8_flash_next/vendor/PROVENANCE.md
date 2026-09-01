# Qwen3.8-Flash-Next / Qwen4 vendor provenance

This directory records the kernel ownership used by the Day0 model commit.
The pinned source is the filesystem extraction of
`vllm/vllm-openai:qwen38-flash-next`:

* image digest: `sha256:bd995759b5b8ac51062e04c9e4d7c91c382d1ba377bb787e24dca2ccb39925e9`
* source root: `/root/qwen4/day0_work/upstream_qwen38_image_source_20260826`
* official runtime: vLLM `0.1.dev20073+g8e685d198`, PyTorch `2.13.0+cu130`
* target runtime: plugin main's vLLM `0.24.0` ABI

All official Python sources carry the upstream SPDX notices:
`Apache-2.0` and `Copyright contributors to the vLLM project`.  The local
files retain those notices.  The source hashes below are SHA-256 hashes of the
unmodified extraction; the vendored file hashes are intentionally different
when the v0.24 namespace/header adapter is applied.

| capability | official source | source SHA-256 | plugin owner | status / local change |
|---|---|---|---|---|
| HC math | `nvidia/ops/hc.py` | `4917215beeadb3265096f29c8080897c7a1af683b92ff403fdd3d0bb969f51d5` | `vendor/official/hc.py` + `common/hyperconnection.py` | kernels are vendored; `vendor/vllm024/dispatch.py` selects them only for explicit NVIDIA CUDA/Triton `QWEN4_HC_BACKEND=official`; pure-torch/current H100 path remains the explicit default fallback |
| QSA math | `nvidia/ops/qsa.py` | `c4ffe3674cafa0ce2dabc39a39f0ddbb4b594bc358ad210ffce9d04383350c7f` | `vendor/official/qsa.py` + `gpu/ops/qsa.py` | official MQA/split-K/store/compression kernels are vendored; v0.24 local QSA owner remains selected because cache layout and metadata differ; local H100/native TopK dispatch is explicit |
| QSA pre-indexer | `nvidia/ops/qsa_pre_indexer.py` | `e93fd5f12a101ffd68b4959eccb3e127f737413a5b422819bee3f49e595e6477` | `vendor/official/qsa_pre_indexer.py` + `gpu/indexer_qsa.py` | math is vendored for later A/B, but execution is **disabled**: the official kernel consumes the newer fused metadata/work graph; current plugin uses eager v0.24 metadata |
| QSA cache metadata | `common/qsa_cache.py` | `e3460b06cd7ed309e47ad5dfd3d4250890539b912385503133bd98a003f73ba8` | `common/qsa_cache.py` | the official metadata Triton graph is not copied/enabled; local builder has one `token_to_req` producer and eager consumers, with explicit invalid-slot handling |
| PLE math/helper | `common/ple.py` | `ab0d4075367c4a8526eed2bdbfe1c5ea3e8accfda84d5f50b8263b771f5613aa` | `common/ple.py` | exact Apache-2.0 helper vendor; `gpu/ple_layer.py` is a v0.24 adapter that keeps native eager state I/O and graph-safe local state kernels |
| PLE layer | `nvidia/ple_layer.py` | `a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7` | `gpu/ple_layer.py` | adapted to v0.24 Mamba state/Conv ABI; newer FP8/offload-only APIs are not imported |
| GDN | `qwen_gdn_linear_attn.py` | `81b4dcd0952492375c93bffc2cdf45f10b45ab5e117f2e1d949a147d144e64f0` | `patches/gdn_packed_decode.py` | the snapshot has a Python selector but no actual C++/CUDA source/object for fused decode; it is **not claimed as vendor**. v0.24 packed recurrent Triton remains, with the existing FP32-beta semantic patch and explicit risk |

The principal adapted-file hashes in this commit are:

| local file | upstream source/hash | local SHA-256 | boundary |
|---|---|---|---|
| `common/hyperconnection.py` | `common/hyperconnection.py` / `cf2028bbd7dceb33f78393be62e3c936c29aff7255dc185f211d1e57cb007393` | `cd7cef3d899b7c185a6c91d670bfbc9562a14de7a88222334fb1abedd86d6280` | renamed/portable HC math plus explicit official-HC opt-in |
| `common/qsa_cache.py` | `common/qsa_cache.py` / `e3460b06cd7ed309e47ad5dfd3d4250890539b912385503133bd98a003f73ba8` | `9179d5be33553b1a2165966cbd8f3e859a15f4d6ad45cfbfe1830c7b6b3a6472` | removed newer metadata graph; eager v0.24 builder and single map producer |
| `gpu/indexer_qsa.py` | `nvidia/indexer_qsa.py` / `e2f398a2fe29466c9681627651ccb5b8eb2b5980445c9e44b270ff45bcb61066` | `cec0a90d0c91c8311b50f9949950e05a4f087f499e5e6f50149d9c1894c568ea` | v0.24 cache/rope adapter; fused pre-indexer deliberately not selected |
| `gpu/ple_layer.py` | `nvidia/ple_layer.py` / `a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7` | `05df0511d8bc071a1319dbebcb1444a5ab9cd0b825dfcc96cded92d646f5da36` | v0.24 Mamba/Conv state adapter, explicit invalid-index masks |
| `gpu/ops/qsa.py` | `nvidia/ops/qsa.py` / `c4ffe3674cafa0ce2dabc39a39f0ddbb4b594bc358ad210ffce9d04383350c7f` | `d4dec3d4d7551982f1e37859a4a71c75735f72415a3bb4f04bcd3082a7c6aa9a` | current validated H100 fallback and native TopK dispatch |
| `gpu/ops/ple_state.py` | `nvidia/ple_layer.py` state path / `a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7` | `5ff4cff6791b74d3bb33c60d83404d022bc345b23991a6d1d7ba8254faa40d23` | graph-safe local stride-aware state I/O; eager native path retained |

The three files in `vendor/official/` have local hashes
`f558029d79d38ea927b7f3e7750f7ffa72566ec60f1f4b68752e5c93c274d262`,
`e3f3c5e206fd54d26de0e0bd7b541e184695d71f98137bf3cff605b37e704b89`, and
`4f1682022900db5344bc51b6a07b62c6a88da749a6637ad56e4b504aff6dc27c` for
`hc.py`, `qsa.py`, and `qsa_pre_indexer.py`, respectively.  Their only
source-level additions are the provenance/HOLD headers; they remain dormant
unless a vLLM 0.24-compatible caller explicitly selects them.

## Layer boundaries

`vendor/official/` is source/math provenance.  `vendor/vllm024/` owns only
the ABI gate and dispatch decision.  The model `common/` and `gpu/` files own
v0.24 cache/metadata/layout adaptation and semantic correctness fixes.  No
module imports the official image's installed newer `vllm` package.

The common attention graph, slot-metadata graph, and Qwen QSA metadata
subgraph are all **HOLD** in this commit.  The QSA builder uses the v0.24
eager path and a single `token_to_req` producer; no second graph producer is
introduced.  Invalid request/block/position indices are explicitly marked
invalid and gathered through a zero sentinel solely for a masked access; they
are never clamped into a real request or cache row.
