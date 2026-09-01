# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606).

The two concrete variants are:
  - ``HyperConnectionBase``  - simple average pooling across hc_count parallel
    streams (equivalent to hyperconnection_average).
  - ``GatedResidualSimple``  - learnable low-rank gated mixing and injection
    (gated_residual_simple).

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout). The local torch
implementation consumes the hyper input viewed as ``[..., HC, HS]``.

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidualSimple(hc_config, role="attn")
    self.mlp_hc = GatedResidualSimple(hc_config, role="mlp")

    hidden_states, residual = self.attn_hc.mix(hidden_states)
    hidden_states = attention(hidden_states)
    hidden_states = self.attn_hc.combine(hidden_states, residual)

    hidden_states, residual = self.mlp_hc.mix(hidden_states)
    hidden_states = mlp(hidden_states)
    hidden_states = self.mlp_hc.combine(hidden_states, residual)
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..vendor.vllm024.dispatch import dispatch_hc_math, use_official_hc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class HyperConnectionConfig:
    """Configuration shared by all HyperConnection variants."""

    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: torch.dtype = torch.bfloat16
    mtp_hc: bool = False
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        # The official image's grouped norm is a useful CUDA/Triton math
        # vendor, but its wrapper only accepts a contiguous 2-D matrix.  Keep
        # the pure-torch implementation for the generic/cross-vendor shapes;
        # the explicit backend switch prevents a second implicit dispatch.
        if (
            self.group_size is not None
            and hidden_states.ndim == 2
            and hidden_states.stride(-1) == 1
            and use_official_hc()
        ):
            from ..vendor.official.hc import grouped_gemma_rmsnorm

            num_groups = hidden_states.shape[-1] // self.group_size

            def fallback_grouped_norm(
                x: torch.Tensor,
                w: torch.Tensor,
                eps: float,
                groups: int,
            ) -> torch.Tensor:
                del w, eps, groups
                grouped = x.float().unflatten(
                    -1, (x.shape[-1] // self.group_size, self.group_size)
                )
                variance = grouped.square().mean(dim=-1, keepdim=True)
                return (
                    grouped * torch.rsqrt(variance + self.variance_epsilon)
                ).flatten(-2).to(input_dtype)

            return dispatch_hc_math(
                grouped_gemma_rmsnorm,
                fallback_grouped_norm,
                hidden_states,
                self.weight,
                self.variance_epsilon,
                num_groups,
            )

        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (
                grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


# ---------------------------------------------------------------------------
# Average-pooling variant
# ---------------------------------------------------------------------------
class HyperConnectionBase(nn.Module):
    """Average-pooling HyperConnection (``hyperconnection_average``).

    Splits the incoming ``[..., HC*HS]`` tensor (HC outer, HS inner) into
    ``HC`` parallel streams, averages them for the block input, and
    broadcasts the block output back to every stream.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        layer_idx: int | None = None,
        role: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.role = role

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    def mix(self, hyper_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Average the HC streams into a single block input."""
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        # [*, HC, HS] — mean over HC (dim=-2).
        unflat = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = unflat.mean(dim=-2)
        return mixed_input, hyper_input

    def combine(
        self, block_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast the block output back to every stream."""
        assert residual.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        residual_reshaped = residual.unflatten(-1, (self.hc_count, self.hidden_size))
        combined = residual_reshaped + block_output.unsqueeze(-2)
        return combined.flatten(-2)


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidualSimple(HyperConnectionBase):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``mix()`` applies GemmaRMSNorm per HC stream and projects through a
    low-rank sigmoid gate to produce a single block input. ``combine()``
    injects the block output back into each stream through a learned
    per-stream injection weight.

    This implementation uses only PyTorch operators. Tensor-parallel
    collectives are supplied by its caller.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        layer_idx: int | None = None,
        role: str | None = None,
        use_mix: bool = True,
        use_combine: bool = True,
    ) -> None:
        super().__init__(config, layer_idx, role)
        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- raw Linear weights (checkpoint-compatible) ----------------------
        if use_mix:
            self.input_mix_weight_down = nn.Linear(
                self.hyper_hidden_size,
                config.hc_lowrank,
                bias=False,
                dtype=config.params_dtype,
            )
            self.input_mix_weight_up = nn.Linear(
                config.hc_lowrank,
                self.hyper_hidden_size,
                bias=False,
                dtype=config.params_dtype,
            )
        if use_combine:
            self.block_inject_weight = nn.Linear(
                self.hyper_hidden_size,
                self.hc_count,
                bias=False,
                dtype=config.params_dtype,
            )

    def _normalize(self, hyper_input: torch.Tensor) -> torch.Tensor:
        if self.config.hc_per_branch_norm:
            return self.hc_norm(hyper_input)
        return self.hc_norm(
            hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        ).flatten(-2)

    def mix(
        self, hyper_input: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Mix: RMSNorm -> low-rank gate -> gated mean."""
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if not hasattr(self, "input_mix_weight_down"):
            raise RuntimeError("mix was disabled for this hyper-connection")
        hyper_input_normed = self._normalize(hyper_input)
        # Gate — original mix order: linear+silu then linear+sigmoid.
        gate = F.silu(
            F.linear(hyper_input_normed, self.input_mix_weight_down.weight)
            / self.hc_count
        )
        gate_logits = F.linear(gate, self.input_mix_weight_up.weight)
        gate = torch.sigmoid(gate_logits).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        if (
            hyper_input_normed.ndim == 2
            and hyper_input_normed.stride(-1) == 1
            and use_official_hc()
        ):
            from ..vendor.official.hc import hc_gate_mix

            def fallback_gate_mix(
                x: torch.Tensor, g: torch.Tensor, count: int
            ) -> torch.Tensor:
                del count
                return (
                    torch.sigmoid(g).unflatten(
                        -1, (self.hc_count, self.hidden_size)
                    )
                    * x.unflatten(-1, (self.hc_count, self.hidden_size))
                ).mean(dim=-2)

            mixed_input = dispatch_hc_math(
                hc_gate_mix,
                fallback_gate_mix,
                hyper_input_normed,
                gate_logits,
                self.hc_count,
            )
        else:
            mixed_input = (
                gate
                * hyper_input_normed.unflatten(
                    -1, (self.hc_count, self.hidden_size)
                )
            ).mean(dim=-2)
        return mixed_input.to(hyper_input.dtype), (hyper_input, hyper_input_normed)

    def combine(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if not hasattr(self, "block_inject_weight"):
            raise RuntimeError("combine was disabled for this hyper-connection")
        hyper_input, hyper_input_normed = residuals
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        # The paired mix keeps its normalized hyper input so combine uses the
        # same HC module's injection weight.
        injection_logits = F.linear(
            hyper_input_normed, self.block_inject_weight.weight
        )

        if (
            hyper_input.ndim == 2
            and hyper_input.stride(-1) == 1
            and use_official_hc()
        ):
            from ..vendor.official.hc import hc_combine

            def fallback_combine(
                residual_input: torch.Tensor,
                block: torch.Tensor,
                injection: torch.Tensor,
                count: int,
            ) -> torch.Tensor:
                residual_streams = residual_input.unflatten(
                    -1, (self.hc_count, self.hidden_size)
                )
                injection_weight = 2.0 * torch.sigmoid(injection / count)
                return (
                    residual_streams
                    + block.unsqueeze(-2) * injection_weight.unsqueeze(-1)
                ).flatten(-2)

            return dispatch_hc_math(
                hc_combine,
                fallback_combine,
                hyper_input,
                block_output,
                injection_logits,
                self.hc_count,
            ).to(hyper_input.dtype)

        residual = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        injection_weight = 2.0 * torch.sigmoid(injection_logits / self.hc_count)
        output = residual + block_output.unsqueeze(-2) * injection_weight.unsqueeze(-1)
        return output.flatten(-2).to(hyper_input.dtype)


__all__ = [
    "GatedResidualSimple",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
]
