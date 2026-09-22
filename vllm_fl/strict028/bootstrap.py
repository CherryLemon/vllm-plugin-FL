# SPDX-License-Identifier: Apache-2.0
"""Entry points must remain safe before device binding and in spawned workers."""

from importlib.metadata import version

from packaging.version import Version

_registered = False


def validate_host() -> None:
    installed = Version(version("vllm"))
    if installed.base_version != "0.28.0" or installed.is_prerelease:
        raise RuntimeError(f"strict028 requires vLLM 0.28.0, found {installed}")
    if installed.local != "empty":
        raise RuntimeError(
            "strict028 requires an unmodified VLLM_TARGET_DEVICE=empty build; "
            f"found {installed}"
        )


def register_platform() -> str:
    validate_host()
    return "vllm_fl.strict028.platform.PlatformFL028"


def register_models() -> None:
    global _registered
    validate_host()
    if _registered:
        return
    from transformers import AutoConfig
    from vllm.tokenizers.registry import TokenizerRegistry

    from .config import DeepseekV41FLConfig

    AutoConfig.register("deepseek_v41", DeepseekV41FLConfig)
    TokenizerRegistry.register(
        "fl_deepseek_v41", "vllm_fl.strict028.tokenizer", "DeepseekV41Tokenizer"
    )
    from vllm.renderers.registry import RENDERER_REGISTRY

    # This renderer delegates all encoding to our tokenizer; its transport and
    # async plumbing are shared with the host's V4 renderer, not its model.
    RENDERER_REGISTRY.register(
        "fl_deepseek_v41", "vllm.renderers.deepseek_v4", "DeepseekV4Renderer"
    )
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV41FlashFLForCausalLM",
        "vllm_fl.strict028.models.deepseek_v41.entry:DeepseekV41FlashFLForCausalLM",
    )
    # Public registration APIs, also required in engine/scheduler processes.
    from . import model_loader  # noqa: F401

    _registered = True
