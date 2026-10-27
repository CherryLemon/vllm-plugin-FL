# Hy4-preview on native vLLM 0.28

This draft forward-ports the Hy4-preview model foundation to `0.5.0-dev`, whose
test dependency is native vLLM 0.28.0. Hy4-preview support already merged in
[#453](https://github.com/flagos-ai/vllm-plugin-FL/pull/453) targets the separate
vLLM 0.24 main line. Its model/configuration concepts are prerequisites being
ported here, not newly invented model support or a second performance result.

The package keeps checkpoint and registry identifiers such as `hy_v4`,
`HYV4ForCausalLM`, `HYV4VLForConditionalGeneration` and `HYV4MTP` unchanged.
It includes configuration, text/VL model and weight mapping, sparse MLA sink and
indexer integration, compressed-tensors W8A8 selection and MTP metadata mapping.
Registration uses the plugin's existing `register_model` extension point.

## Ownership and dependencies

The plugin owns model data flow, metadata, checkpoint mapping, graph input
staging and provider selection. Reusable numerical kernels remain in FlagGems
and FlagGems-vllm. iHC pre/post use the existing public library implementations;
image normalization uses its Torch reference composition in this forward-port.

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

Additional optional E23 APIs are explicit pending library dependencies until
their delivery is verified: `indexer_prefix_topk` /
`supports_indexer_prefix_topk`, `int8_indexer_paged_logits_empty_tile`,
`mrope_strided_cache` / `can_mrope_strided_cache`, `decode_indexer_metadata`,
`mtp_tail_step`, `async_input_correct`, `async_input_gather` and
`populate_request_metadata`. This draft does not claim that #1002 already
contains every one of them. Keep their corresponding options disabled unless
the required provider is installed. The optional gate-quant and decode-O-proj
providers are imported only when their selected paths run.

## Validation boundaries

The submitted forward-port has CPU registration/metadata/launch-selection
contracts, Python syntax/import-closure checks and formatting checks. It has not
been integrated or run on GPUs against the current branch and library PR heads.
CPU tests do not establish tensor numerics, graph replay, MTP acceptance,
multimodal correctness, distributed safety or serving performance.

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
