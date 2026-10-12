# SPDX-License-Identifier: Apache-2.0
"""Device-kernel availability checks shared by operator providers."""

import torch


def has_device_kernel(name: str, device_type: str) -> bool:
    """A schema or a Meta kernel alone does not implement runtime execution."""
    key = {"cuda": "CUDA", "cpu": "CPU", "xpu": "XPU"}.get(device_type, "PrivateUse1")
    try:
        return any(
            torch._C._dispatch_has_kernel_for_dispatch_key(name, k)
            for k in (key, "CompositeImplicitAutograd", "CompositeExplicitAutograd")
        )
    except RuntimeError:
        return False
