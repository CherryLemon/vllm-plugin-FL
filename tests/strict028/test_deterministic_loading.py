# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_fl.strict028.models.deepseek_v41.model import Linear


@pytest.mark.gpu
def test_fp4_parameter_allocation_with_deterministic_algorithms():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.device("cuda"):
            layer = Linear(64, 32, dtype=torch.float4_e2m1fn_x2)
        assert layer.weight.dtype == torch.float4_e2m1fn_x2
        assert layer.weight.shape == (32, 32)
        assert layer.scale.shape == (32, 2)
        assert layer.weight.scale is layer.scale
        # Loading bytes must preserve the checkpoint's nibble codes verbatim.
        packed = (
            torch.arange(256, dtype=torch.uint8, device="cuda").repeat(4).view(32, 32)
        )
        with torch.no_grad():
            layer.weight.view(torch.uint8).copy_(packed)
        assert torch.equal(layer.weight.view(torch.uint8), packed)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
