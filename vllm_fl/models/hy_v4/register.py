# SPDX-License-Identifier: Apache-2.0
"""Register Hy4-preview configuration and models with the vLLM 0.28 host."""

import os


def validate_native_graph_hooks(proposer_cls=None):
    """Reject graph flags when the companion native integration is absent."""
    flags = (
        "HY4_MTP_DRAFT_FULL_GRAPH",
        "HY4_MTP_FULL_DRAFT3_GRAPH",
        "HY4_MTP_TAIL2_GRAPH",
        "HY4_PREFILL_BREAKABLE_GRAPH",
        "HY4_MTP_QSL_GPU_STAGING",
        "FLAGOS_ASYNC_INPUT_STAGING",
        "HY4_SPEC_COMMON_REQUEST_IDS",
    )
    if not any(os.environ.get(flag, "0") == "1" for flag in flags):
        return
    if proposer_cls is None:
        from vllm.v1.spec_decode.llm_base_proposer import LLMBaseProposer

        proposer_cls = LLMBaseProposer
    if not hasattr(proposer_cls, "_hy4_call_model"):
        raise RuntimeError(
            "Hy4-preview graph/staging options require the companion "
            "native vLLM 0.28 runner/proposer integration; keep these "
            "options disabled with an unpatched vLLM wheel."
        )


def register_hy_v4():
    validate_native_graph_hooks()
    from transformers import AutoConfig
    from vllm.model_executor.models import ModelRegistry

    from .config import HYV4Config, HYV4VLConfig

    AutoConfig.register("hy_v4", HYV4Config, exist_ok=True)
    AutoConfig.register("hy_v4_vl", HYV4VLConfig, exist_ok=True)
    ModelRegistry.register_model(
        "HYV4ForCausalLM", "vllm_fl.models.hy_v4.nvidia.model:HYV4ForCausalLM"
    )
    ModelRegistry.register_model(
        "HYV4VLForConditionalGeneration",
        "vllm_fl.models.hy_v4.nvidia.text_compat:HYV4VLForConditionalGeneration",
    )
    ModelRegistry.register_model("HYV4MTP", "vllm_fl.models.hy_v4.nvidia.mtp:HYV4MTP")

    from .mtp_config import (
        install_hyv4_mtp_config_override,
        install_hyv4_mtp_parallel_config,
    )

    install_hyv4_mtp_config_override()
    install_hyv4_mtp_parallel_config()
