# SPDX-License-Identifier: Apache-2.0
"""Explicit FL calls for the Eager compatibility profile.

The composition in model.py still uses PyTorch for routing, index selection,
RoPE, Engram lookup/gating and residual arithmetic. This is a declared reference
profile, not a claim that all operations have migrated to FlagGems.
"""

import torch


def dense_linear(x, weight):
    import flag_gems

    out = flag_gems.mm(x.reshape(-1, x.shape[-1]), weight.T)
    return out.reshape(*x.shape[:-1], weight.shape[0])


def lowp_linear(x, weight, scale):
    from flag_gems.fused.block_scaled_lowp_linear import block_scaled_lowp_linear

    fp4 = weight.dtype == torch.float4_e2m1fn_x2
    out = block_scaled_lowp_linear(
        x.reshape(-1, x.shape[-1]).contiguous(),
        weight.view(torch.uint8) if fp4 else weight,
        scale,
        weight_format="mxfp4" if fp4 else "fp8",
        output_dtype=x.dtype,
    )
    return out.reshape(*x.shape[:-1], weight.shape[0])


def act_quant(
    x, block_size=32, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu, inplace=False
):
    from flag_gems.fused.act_quant import act_quant_triton

    if block_size != 32 or scale_fmt != "ue8m0" or scale_dtype != torch.float8_e8m0fnu:
        raise ValueError(
            "V4.1 compatibility profile requires per-32 E8M0 activation scales"
        )
    q, scale = act_quant_triton(x.contiguous(), block_size, scale_fmt)
    if inplace:
        # Explicit temporary reference composition, recorded in execution manifest.
        decoded = (
            q.float().reshape(*scale.shape, block_size) * scale[..., None]
        ).reshape(x.shape)
        x.copy_(decoded)
        return x
    return q, scale.to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    from flag_gems.fused.dsv41_reference_ops import fp4_quantize_reference

    if scale_dtype not in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        raise ValueError("unsupported FP4 scale dtype")
    return fp4_quantize_reference(
        x,
        block_size,
        scale_format="e8m0" if scale_dtype == torch.float8_e8m0fnu else "e4m3",
        inplace=inplace,
    )


def sparse_attn(*args):
    from flag_gems.fused.dsv41_reference_ops import sparse_attention_with_sink

    return sparse_attention_with_sink(*args)


def hc_split_sinkhorn(*args):
    from flag_gems.fused.mhc.hc_split_sinkhorn import hc_split_sinkhorn as op

    return op(*args)


EXECUTION_PROFILE = {
    "name": "fl_dsv41_eager_reference_v1",
    "flaggems": [
        "block_scaled_lowp_linear",
        "mm",
        "act_quant_triton",
        "fp4_quantize_reference",
        "sparse_attention_with_sink",
        "hc_split_sinkhorn",
    ],
    "torch_reference": [
        "routing/topk",
        "indexer einsum and score rounding",
        "RoPE",
        "Engram hash/lookup/gate",
        "norm and residual arithmetic",
        "compressor softmax",
        "FP8 cache dequantize",
        "vision",
    ],
    "communication": "torch.distributed NCCL; homogeneous TP with local routed experts",
    "performance": "not_profiled",
}
