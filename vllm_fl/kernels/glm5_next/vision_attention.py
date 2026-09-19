# SPDX-License-Identifier: Apache-2.0
"""Model-local FlagGems FA2 adapter; never replaces public vLLM functions."""

import torch
from torch import nn

from vllm.utils.torch_utils import direct_register_custom_op


def _vision_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    from flag_gems import flash_attn_varlen_func

    batch, seq_len, heads, head_dim = query.shape
    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0,
            (batch + 1) * seq_len,
            seq_len,
            dtype=torch.int32,
            device=query.device,
        )
    # The packed token count is a static upper bound for every sequence.
    # Do not read max_seqlen from a CUDA tensor inside graph capture.
    result = flash_attn_varlen_func(
        query.reshape(-1, heads, head_dim),
        key.reshape(-1, heads, head_dim),
        value.reshape(-1, heads, head_dim),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=batch * seq_len,
        max_seqlen_k=batch * seq_len,
        dropout_p=0.0,
        causal=False,
        softmax_scale=scale,
        fa_version=2,
    )
    # Only this vision caller consumes the output alone. Public FA callers
    # retain their original tuple/LSE contract.
    if isinstance(result, tuple):
        result = result[0]
    return result.reshape_as(query)


def _vision_attention_fake(query, key, value, cu_seqlens, scale):
    return torch.empty_like(query)


direct_register_custom_op(
    op_name="glm5_vision_attention",
    op_func=_vision_attention,
    fake_impl=_vision_attention_fake,
)


class Glm5VisionAttention(nn.Module):
    def __init__(self, num_heads, head_size, scale, prefix=""):
        super().__init__()
        self.scale = scale

    def forward(self, query, key, value, cu_seqlens=None, max_seqlen=None):
        return torch.ops.vllm.glm5_vision_attention(
            query, key, value, cu_seqlens, self.scale
        )
