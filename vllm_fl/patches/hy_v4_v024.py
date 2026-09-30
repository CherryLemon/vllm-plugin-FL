# Copyright (c) 2026 BAAI. All rights reserved.
"""Install HY4 support into a pristine vLLM 0.24 runtime.

All changes stay inside vllm-plugin-FL. The hook registers the checkpoint
config, compressed-MLA architecture conversion, lazy model implementation,
and expert-sliced safetensors loader without modifying the vLLM installation.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version as package_version

from vllm_fl.configs.hy_v4_quantization import (
    _patch_mxfp8_override_order as _patch_mxfp8_override_order,
)

logger = logging.getLogger(__name__)

_ARCHITECTURE = "HYV4ForCausalLM"
_LOAD_FORMAT = "hy4_safetensors"


def is_vllm_024() -> bool:
    """Return whether the active vLLM belongs to the 0.24 ABI line.

    HY4 is intentionally implemented against vLLM 0.24.  Keep this probe
    local to the model adapter so the model commit does not import the
    unrelated compatibility module from another branch.
    """
    try:
        release = package_version("vllm")
    except PackageNotFoundError:
        try:
            import vllm

            release = getattr(vllm, "__version__", "")
        except Exception:
            return False
    parts = release.split("+", 1)[0].split(".")
    return len(parts) >= 2 and parts[:2] == ["0", "24"]


def apply_hy_v4_v024_patches() -> bool:
    """Register the HY4 runtime components required by vLLM 0.24.x."""
    if not is_vllm_024():
        return False

    from vllm.model_executor import model_loader
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import (
        config as transformers_config,
        model_arch_config_convertor,
    )

    from vllm_fl.configs.hy_v4 import HYV4Config
    from vllm_fl.configs.hy_v4_convertor import HYV4ModelArchConfigConvertor
    from vllm_fl.model_loader.hy_v4_loader import HYV4SafetensorsLoader

    transformers_config._CONFIG_REGISTRY.setdefault("hy_v4", HYV4Config)
    model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS["hy_v4"] = (
        HYV4ModelArchConfigConvertor
    )
    model_registry.ModelRegistry.register_model(
        _ARCHITECTURE,
        "vllm_fl.models.hy_v4:HYV4ForCausalLM",
    )

    registered_loaders = model_loader._LOAD_FORMAT_TO_MODEL_LOADER
    if registered_loaders.get(_LOAD_FORMAT) is not HYV4SafetensorsLoader:
        model_loader.register_model_loader(_LOAD_FORMAT)(HYV4SafetensorsLoader)

    logger.info("Installed HY4 runtime compatibility for vLLM 0.24")
    return True


__all__ = [
    "apply_hy_v4_v024_patches",
]


def __getattr__(name):
    if name == "HYV4ModelArchConfigConvertor":
        from vllm_fl.configs.hy_v4_convertor import HYV4ModelArchConfigConvertor

        return HYV4ModelArchConfigConvertor
    if name == "HYV4Config":
        from vllm_fl.configs.hy_v4 import HYV4Config

        return HYV4Config
    if name == "HYV4SafetensorsLoader":
        from vllm_fl.model_loader.hy_v4_loader import HYV4SafetensorsLoader

        return HYV4SafetensorsLoader
    raise AttributeError(name)
