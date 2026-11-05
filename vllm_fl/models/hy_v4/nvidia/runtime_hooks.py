# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 model-scoped runtime policy and delayed official vLLM method patches."""

from __future__ import annotations


def _runtime_capabilities():
    from vllm.config import CUDAGraphMode
    from vllm.platforms import current_platform

    return current_platform, CUDAGraphMode


def _int8_indexer_selected():
    from ..indexer_policy import use_int8_indexer

    return use_int8_indexer()


def _supports_mixed_config(config):
    from .prefill_graph import supports_mixed_config

    return supports_mixed_config(config)


def create_hy4_runtime(config):
    """Decline unsupported configurations before creating a graph or buffer."""
    hf = getattr(getattr(config, "model_config", None), "hf_config", None)
    if getattr(hf, "model_type", None) not in ("hy_v4", "hy_v4_vl"):
        return None
    current_platform, CUDAGraphMode = _runtime_capabilities()

    if not current_platform.is_cuda():
        return None
    if current_platform.get_device_capability() != (9, 0):
        return None
    parallel = config.parallel_config
    spec = config.speculative_config
    compilation = config.compilation_config
    if (
        spec is None
        or spec.method != "mtp"
        or spec.num_speculative_tokens != 3
        or spec.enforce_eager
        or spec.parallel_drafting
        or spec.disable_padded_drafter_batch
        or parallel.tensor_parallel_size != 16
        or not parallel.enable_expert_parallel
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.use_ubatching
        or config.lora_config is not None
        or compilation.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
    ):
        return None
    if not _int8_indexer_selected():
        return None
    if not _supports_mixed_config(config):
        return None
    return Hy4RuntimeHooks()


class Hy4RuntimeHooks:
    """Framework resource factories; numerical providers retain their policy."""

    mixed_graphs = True

    @staticmethod
    def make_common_metadata_graph():
        from .runtime_metadata import CommonAttentionMetadataGraphRunner

        return CommonAttentionMetadataGraphRunner()

    @staticmethod
    def make_input_staging(runner):
        from .runtime_staging import make_async_input_staging

        return make_async_input_staging(runner)

    @staticmethod
    def make_draft_graph(proposer, runner):
        from .mtp_draft_graph import make_hyv4_draft_graph

        return make_hyv4_draft_graph(proposer, runner)

    @staticmethod
    def select_decode_deltas(*args, **kwargs):
        from .runtime_metadata import select_decode_deltas

        return select_decode_deltas(*args, **kwargs)

    @staticmethod
    def compute_step_common_attention_metadata(*args, **kwargs):
        from .runtime_metadata import compute_step_common_attention_metadata

        return compute_step_common_attention_metadata(*args, **kwargs)

    @staticmethod
    def decode_lens_from_common(*args, **kwargs):
        from .runtime_metadata import decode_lens_from_common

        return decode_lens_from_common(*args, **kwargs)

    @staticmethod
    def wrap_breakable(*args, **kwargs):
        from .prefill_graph import wrap_native_breakable

        return wrap_native_breakable(*args, **kwargs)

    @staticmethod
    def warm_full_decode_metadata(*args, **kwargs):
        from .decode_metadata import warm_full_decode_metadata_before_capture

        return warm_full_decode_metadata_before_capture(*args, **kwargs)


def bind_common_runtime_metadata(common):
    """Keep immutable metadata facts within their owning worker's active step."""
    from .runtime_patch import current_runner
    from .runtime_proof import get_current_cpu_seq_lens_proof

    runner = current_runner()
    if runner is None or getattr(common, "_model_runtime", None) is not getattr(
        runner, "_model_runtime", None
    ):
        return common
    proof = getattr(common, "cpu_seq_lens_proof", None)
    if proof is not None and proof is not get_current_cpu_seq_lens_proof():
        common.cpu_seq_lens_proof = None
    return common


def prepare_runtime_for_worker(config):
    """Select and install before constructing this NVIDIA worker's V1 runner."""
    if config is None:
        return False
    hf = getattr(getattr(config, "model_config", None), "hf_config", None)
    if getattr(hf, "model_type", None) not in ("hy_v4", "hy_v4_vl"):
        return False
    if create_hy4_runtime(config) is None:
        return False
    if getattr(config, "use_v2_model_runner", False):
        raise ValueError(
            "Hy4 MTP3 graph optimization requires VLLM_USE_V2_MODEL_RUNNER=0"
        )
    from .runtime_patch import install_runtime_patches

    install_runtime_patches()
    return True
