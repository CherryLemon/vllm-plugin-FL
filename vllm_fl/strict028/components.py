# SPDX-License-Identifier: Apache-2.0
"""Explicit FlagGems components for the V4.1 numerical compatibility path."""

import math

import torch
from torch import nn

from .weights import PackedExperts


class FLQuantLinear(nn.Module):
    """Keep checkpoint encoding; retain E4M3 activation quantization per K=32."""

    def __init__(self, weight, scale, *, weight_format):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.weight_format = weight_format

    def forward(self, x):
        from flag_gems.fused.block_scaled_lowp_linear import block_scaled_lowp_linear

        return block_scaled_lowp_linear(
            x,
            self.weight,
            self.scale,
            weight_format=self.weight_format,
            output_dtype=x.dtype,
        )


class FLPackedExpert(nn.Module):
    """One expert/shard with clamp and router weighting before GEMM2 quantization.

    This component follows the released reference Expert. The fast Marlin
    composition has a different activation/weighting contract and is therefore
    not silently selected here. TP partial outputs require an external sum.
    """

    def __init__(
        self, experts: PackedExperts, local_expert: int, *, clamp_limit: float
    ):
        super().__init__()
        if not 0 <= local_expert < len(experts.shard.expert_ids):
            raise ValueError("Expert is not local to this shard")
        if not math.isfinite(clamp_limit) or clamp_limit <= 0:
            raise ValueError("V4.1 requires a positive SwiGLU clamp")
        self.clamp_limit = clamp_limit
        self.intermediate_size = experts.gate_up.shape[1] // 2
        self.gate_up = FLQuantLinear(
            experts.gate_up[local_expert],
            experts.gate_up_scale[local_expert],
            weight_format="mxfp4",
        )
        self.down = FLQuantLinear(
            experts.down[local_expert],
            experts.down_scale[local_expert],
            weight_format="mxfp4",
        )

    def forward(self, x: torch.Tensor, router_weight: torch.Tensor | None = None):
        import flag_gems
        from flag_gems.fused.silu_and_mul_with_clamp import silu_and_mul_with_clamp_out

        if x.ndim != 2:
            raise ValueError("Expert input must be [tokens, hidden]")
        if router_weight is not None and (
            router_weight.shape != (x.shape[0], 1)
            or router_weight.dtype != torch.float32
            or router_weight.device != x.device
        ):
            raise ValueError(
                "Router weights must be FP32 [tokens, 1] on the input device"
            )
        if x.shape[0] == 0:
            return torch.empty_like(x)
        projected = self.gate_up(x)
        gate, up = projected.split(self.intermediate_size, dim=-1)
        # The reference rounds GEMM1 to the activation dtype, then performs
        # clamp, SiLU, multiplication and routing in FP32 before one final cast.
        activated = torch.empty(gate.shape, dtype=torch.float32, device=x.device)
        silu_and_mul_with_clamp_out(gate, up, activated, self.clamp_limit)
        if router_weight is not None:
            activated = flag_gems.mul(activated, router_weight)
        return self.down(activated.to(x.dtype))
