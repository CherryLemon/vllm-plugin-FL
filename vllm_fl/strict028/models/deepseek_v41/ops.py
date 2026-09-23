# SPDX-License-Identifier: Apache-2.0
"""Explicit FL calls for the reference Prefill and Graph Decode profiles.

The composition in model.py still uses PyTorch for routing, index selection,
RoPE, Engram lookup/gating and residual arithmetic. This is a declared reference
profile, not a claim that all operations have migrated to FlagGems.
"""

import os
from contextlib import contextmanager

import torch

_decode_batch_size = 1
_prefill_prompt_length = 0


@contextmanager
def prefill_geometry(prompt_length):
    """Retain the complete prefix's unquantized reduction geometry."""
    global _prefill_prompt_length
    previous, _prefill_prompt_length = _prefill_prompt_length, prompt_length
    try:
        yield
    finally:
        _prefill_prompt_length = previous


def prefill_uses_tiled_hc():
    return _prefill_prompt_length > 4096


def prefill_linear(x, weight, reference_rows=None):
    """Pad independent rows to preserve the reference cuBLAS algorithm.

    Changing M changes FP32 accumulation in routers and compressors. Those
    ULP differences can cross later low-precision quantization boundaries.
    Padding is confined to Prefill and never changes Decode Graph execution.
    """
    rows = _prefill_prompt_length if reference_rows is None else reference_rows
    count = x.numel() // x.shape[-1]
    if rows > count:
        padded = x.new_zeros((rows, x.shape[-1]))
        padded[:count].copy_(x.reshape(count, -1))
        out = torch.nn.functional.linear(padded, weight)[:count]
        return out.reshape(*x.shape[:-1], weight.shape[0]).clone()
    return dense_linear(x, weight)


@contextmanager
def decode_dense_batch(batch_size):
    """Preserve each request's reference GEMV/GEMM reduction in a decode batch.

    Changing cuBLAS M from one request to a batch can alter FP32 compressor
    state by one ULP, which then crosses low-precision quantization boundaries.
    This compatibility composition keeps the request dimension separate for
    the unquantized projections. Low-precision FlagGems GEMMs remain batched.
    """
    global _decode_batch_size
    previous, _decode_batch_size = _decode_batch_size, batch_size
    try:
        yield
    finally:
        _decode_batch_size = previous


@contextmanager
def gathered_decode_batch(data_size):
    """Preserve request grouping after gathering equal batches across DP."""
    with decode_dense_batch(_decode_batch_size * data_size):
        yield


def dense_linear(x, weight):
    # Preserve the published reduction for unquantized mHC, compressor and
    # head projections. Small changes amplify at subsequent quantization ties.
    # This is an explicit reference composition, not a silent dispatch fallback.
    if _decode_batch_size > 1:
        if x.shape[0] % _decode_batch_size:
            raise ValueError("dense decode projection must retain its request grouping")
        return torch.cat(
            [
                torch.nn.functional.linear(part, weight)
                for part in x.chunk(_decode_batch_size, dim=0)
            ],
            dim=0,
        )
    return torch.nn.functional.linear(x, weight)


def grouped_output_projection(x, weight):
    """Preserve each request's cuBLAS M for the BF16 grouped wo_a product."""
    if _decode_batch_size > 1:
        if x.shape[0] != _decode_batch_size:
            raise ValueError("grouped decode output must retain its request axis")
        return torch.cat(
            [torch.einsum("bsgd,grd->bsgr", part, weight) for part in x.split(1)], dim=0
        )
    return torch.einsum("bsgd,grd->bsgr", x, weight)


def decode_mean(x, dim, keepdim=False):
    """Preserve the reference reduction geometry for each decode request.

    ATen can change the number of reduction CTAs when the output row count
    grows. Sub-ULP FP32 changes in mHC statistics can become BF16 residual
    differences, then alter routing. Keep the scalar request's geometry while
    the surrounding low-precision GEMMs and state access remain batched.
    """
    if _decode_batch_size > 1:
        if x.shape[0] % _decode_batch_size:
            raise ValueError("decode reduction must retain its request grouping")
        return torch.cat(
            [
                part.mean(dim=dim, keepdim=keepdim)
                for part in x.chunk(_decode_batch_size, dim=0)
            ],
            dim=0,
        )
    return x.mean(dim=dim, keepdim=keepdim)


def decode_sum(x, dim, keepdim=False):
    """Keep per-request ATen sum association, as for decode_mean."""
    if _decode_batch_size > 1:
        if x.shape[0] % _decode_batch_size:
            raise ValueError("decode sum must retain its request grouping")
        return torch.cat(
            [
                part.sum(dim=dim, keepdim=keepdim)
                for part in x.chunk(_decode_batch_size, dim=0)
            ],
            dim=0,
        )
    return x.sum(dim=dim, keepdim=keepdim)


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
    from flag_gems.fused.dsv41_reference_ops import hc_split_sinkhorn_reference as op

    return op(*args)


EXECUTION_PROFILE = {
    "name": (
        "fl_dsv41_batched_decode_graph_v1"
        if os.environ.get("VLLM_FL_BATCHED_DECODE") == "1"
        else "fl_dsv41_decode_graph_v1"
        if os.environ.get("VLLM_FL_DECODE_GRAPH") == "1"
        else "fl_dsv41_eager_reference_v1"
    ),
    "flaggems": [
        "block_scaled_lowp_linear",
        "block_scaled_mxfp4_moe",
        "act_quant_triton",
        "fp4_quantize_reference",
        "sparse_attention_with_sink",
        "hc_split_sinkhorn_reference",
    ],
    "torch_reference": [
        "unquantized projections (mHC, compressor and head)",
        "routing/topk",
        "indexer einsum and score rounding",
        "RoPE",
        "Engram hash/lookup/gate",
        "norm and residual arithmetic",
        "compressor softmax",
        "FP8 cache dequantize",
        "vision",
    ],
    "communication": (
        "FlagCX; homogeneous TP with local routed experts"
        if os.environ.get("VLLM_FL_TP_BACKEND", "nccl").lower() == "flagcx"
        else "torch.distributed NCCL; homogeneous TP with local routed experts"
    ),
    "performance": "not_profiled",
}
