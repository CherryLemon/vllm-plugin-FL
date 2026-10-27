# Hy4-preview on native vLLM 0.28

This draft forward-ports the Hy4-preview model foundation to `0.5.0-dev`, whose
test dependency is native vLLM 0.28.0. Hy4-preview support already merged in
[#453](https://github.com/flagos-ai/vllm-plugin-FL/pull/453) targets the separate
vLLM 0.24 main line. Its model/configuration concepts are prerequisites being
ported here, not newly invented model support or a second performance result.

## Cross-branch text prerequisites

The PR target is still `0.5.0-dev` at
`0fc46f4086cae74fb19337ad31e6d7f69d8d8df6`. The separately inspected
`0.4.0-dev` head is `9ad9a78930ced6e6b48505de46a72a72cdd35c97`;
these two heads currently have the same source tree,
`7f84954dde18b36ea841589356215d6857569549`. This PR has not been retargeted.
Neither target tree contains the Hy4-preview package from main's #453.

The necessary text prerequisites are configuration, causal model/weight mapping,
MLA/indexer/sink integration, routed/shared experts and plugin registration.
They are a version-scoped port of the existing adaptation, not achievements
newly counted as this round's optimizations. The code keeps identifiers such as
`hy_v4`, `HYV4ForCausalLM` and `HYV4MTP` unchanged.

There is no processor, vision encoder, image/video preprocessing or multimodal
registration in this PR. A small `text_compat.py` entry retains the existing
`HYV4VLForConditionalGeneration` checkpoint architecture name for ordinary text
inference. It uses the same causal text model, preserves token IDs/embeddings,
ignores encoder weights without allocating them, and explicitly rejects media
inputs. Its outer configuration keeps only the nested text configuration.
Repeated text positions retain the checkpoint's three-axis rotary shape; no
media grid or placeholder logic is implemented.

## This round's optimization integration

The optional increments are IHC post/pre and MLA provider consumers, W8A8
O-proj selection, MTP configuration/quantization metadata, ordinary draft-body /
full3 / tail graphs, mixed-prefill wrappers, input staging and common request
metadata. They do not rewrite the core scheduler. Each numerical library API
and each native compatibility requirement is listed below; none is counted as
already delivered just because its consumer is present here.

## Ownership and dependencies

The plugin owns model data flow, metadata, checkpoint mapping, graph input
staging and provider selection. Reusable numerical kernels remain in FlagGems
and FlagGems-vllm. iHC pre/post use the existing public library implementations;
no reusable numerical kernel body is copied into this framework port.

The optional production paths extracted from the historical E23 source are
default-off. API presence is a dependency, not proof that the path executed.

| Option | Framework behavior | Required library or native support |
|---|---|---|
| `HY4_IHC_POST_PRE=1` | Fuse the intra-layer post/pre boundary after capability checks | FlagGems-vllm `ihc_post_pre` / `can_ihc_post_pre`, draft [#1002](https://github.com/flagos-ai/FlagGems-vllm/pull/1002) |
| `HY4_MLA_COMPOSITE_STAGING=1` | Prepare Q/sink tensors before native sparse MLA | FlagGems-vllm `concat_pad_mla_q_and_sink`, draft #1002 |
| `HY4_MLA_GATE_QUANT=1` | Produce the stock `QuantizedActivation` for the selected W8A8 O-proj | FlagGems-vllm `mul_scaled_int8_quant`, draft #1002, and the compatible INT8 O-proj provider |
| `HY4_INT8_OPROJ_DECODE=1` | Select the decode projection provider while retaining native weight loading and TP reduction | FlagGems-vllm `int8_scaled_mm_decode`; separate provider delivery required |
| `HY4_INDEXER_DTYPE=int8` | Select INT8 query/cache/logits with the native indexer metadata ABI | FlagGems INT8 sparse-indexer APIs; optional prefix TopK / empty-tile APIs require their library delivery |
| `HY4_MTP_DRAFT_FULL_GRAPH=1` | Stage native proposer results into ordinary draft-body graphs | Companion native vLLM 0.28 runner/proposer hooks |
| `HY4_MTP_FULL_DRAFT3_GRAPH=1`, `HY4_MTP_TAIL2_GRAPH=1` | Capture three draft forwards or the two-forward tail; retain native heads, sampling and rejection | Body-graph option, companion native hooks and FlagGems-vllm `mtp_tail_step` |
| `HY4_PREFILL_BREAKABLE_GRAPH=1` | Dispatch native target FULL versus mixed/prefill PIECEWISE wrappers | Companion native hooks, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, the exact validated configuration accepted by `validate_mixed_config` |
| `FLAGOS_ASYNC_INPUT_STAGING=1`, `HY4_MTP_QSL_GPU_STAGING=1`, `HY4_SPEC_COMMON_REQUEST_IDS=1` | Native launch/input/metadata preparation and request-ID transport | Companion native hooks plus FlagGems-vllm integer metadata/staging APIs |

The graph modules are connected through native vLLM extension hooks. Those
native changes belong to the native vLLM repository and are not copied into this
plugin PR. They affect `gpu_model_runner.py`, `llm_base_proposer.py`, the MLA
indexer metadata builder and sparse MLA attention. Enabling graph/staging flags
on a stock wheel raises a clear compatibility error at registration rather than
silently ignoring the options. A compatibility port must retain only ordinary
production graph/metadata hooks; historical diagnostic bootstrap code is not a
dependency of this draft.

The already merged common-attention-metadata work in
[#442](https://github.com/flagos-ai/vllm-plugin-FL/pull/442) is a prerequisite
concept being ported to this branch, not new upstream coverage. The separate
INT8 fallback proposal [#622](https://github.com/flagos-ai/vllm-plugin-FL/pull/622)
targets the main line; this version-scoped package does not amend that PR.
The RMS public API work in FlagGems-vllm
[#1001](https://github.com/flagos-ai/FlagGems-vllm/pull/1001) is separate.

Library delivery was checked separately from these consumers:

| Status | Contracts |
|---|---|
| Published in FlagGems-vllm #1002 at `d5179b3bd7b4e23d37b09a16ad44bd6cbd8782e2` | iHC post/pre, Q/sink concat/pad, gate-times-INT8 quantization, SM90 decode INT8 GEMM, grouped routing TopK, pure SiLU/quant fusion |
| Also published in #1002 at the same head; no current-model GPU qualification implied | MRoPE strided cache, MTP tail state, async correct/gather and request-ID metadata |
| Held / not delivered by the performance PR | prefix TopK, decode-indexer metadata, empty-tile logits; core FlagGems INT8 indexer also retains a graph-validation gap |

The eleven published library contracts and their framework consumers remain
separate from current-model integration validation. The optional gate-quant and
decode-O-proj providers load only when selected paths run. Held APIs do not
become delivered merely because their consumers exist here. Full MTP/metadata
integration still requires the separate native work and GPU checks above.

## Validation boundaries

The submitted forward-port has CPU registration/metadata/launch-selection
contracts, Python syntax/import-closure checks and formatting checks. It has not
been integrated or run on GPUs against the current branch and library PR heads.
CPU tests do not establish tensor numerics, graph replay, MTP acceptance,
distributed safety or serving performance. Media support is outside this PR.

Historical E23 measured one fixed prompt, C1, 1024 input / 256 output tokens,
three repeats, with combined mean TPOT 10.256724 ms. That result belongs to its
original frozen source/provider/native integration. This adapted draft omits
historical diagnostic source and uses library-owned iHC operators, so it does
not reproduce or inherit that performance qualification. There was no matched
P0N3 OFF256 per-feature ablation; no latency contribution is assigned to a
single optimization. C64 weighted performance and independent broad accuracy
remain unvalidated here.

The retained profile evidence distinguished ordinary FULL/MTP replays from
PIECEWISE prefill and eager input staging. The entire step is not claimed to be
FULL. Later IHC pair fusion and MoE GEMM geometry candidates are not part of
this forward-port. Old owned/bootstrap managers, lifetime probes, load/step
receipts, absolute deployment paths, service logs and frozen evidence bundles
are intentionally absent.

CPU checks can be run without importing accelerator model modules:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  --confcutdir=tests/unit_tests/models \
  tests/unit_tests/models/test_hy4_preview_028_contracts.py
```
