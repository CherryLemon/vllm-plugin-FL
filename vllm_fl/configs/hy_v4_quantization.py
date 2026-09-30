# SPDX-License-Identifier: Apache-2.0
"""HY4 MXFP8 override ordering without runtime registration dependencies."""

from functools import wraps
from typing import Any


def _patch_mxfp8_override_order(me_quant: Any) -> None:
    """Make vLLM 0.24 probe the canonical ModelOpt MXFP8 entry first.

    vLLM 0.24 maps both ``modelopt_mxfp8`` and the online shorthand
    ``mxfp8`` to ``ModelOptMxFp8Config``, but only the former is present in
    ``ModelConfig._verify_quantization``'s ordered override list.  Therefore
    the shorthand reports an override before the canonical entry is reached
    and ModelConfig rejects a serialized MXFP8 checkpoint.  Returning a
    no-override view for the shorthand is equivalent to placing ``mxfp8``
    after ``modelopt_mxfp8`` in that list, without replacing ModelConfig or
    modifying the vLLM installation.
    """
    current_getter = me_quant.get_quantization_config
    if getattr(current_getter, "_hy4_v024_mxfp8_order", False):
        return

    alias = None

    def make_alias():
        class MXFP8AliasAfterCanonical(current_getter("mxfp8")):
            @classmethod
            def override_quantization_method(
                cls,
                hf_quant_cfg: dict[str, Any],
                user_quant: str | None,
                hf_config: Any = None,
            ) -> None:
                if getattr(hf_config, "model_type", None) == "hy_v4":
                    return None
                return current_getter("mxfp8").override_quantization_method(
                    hf_quant_cfg, user_quant, hf_config=hf_config
                )

        return MXFP8AliasAfterCanonical

    @wraps(current_getter)
    def get_quantization_config(name: str):
        nonlocal alias
        if name == "mxfp8":
            if alias is None:
                alias = make_alias()
            return alias
        return current_getter(name)

    get_quantization_config._hy4_v024_mxfp8_order = True
    me_quant.get_quantization_config = get_quantization_config

