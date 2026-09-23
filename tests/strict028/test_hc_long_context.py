# SPDX-License-Identifier: Apache-2.0
"""Long-prefill Hyper-Connections keep the short-path arithmetic."""

from types import SimpleNamespace

import torch

from vllm_fl.strict028.models.deepseek_v41 import model as model_module
from vllm_fl.strict028.models.deepseek_v41.model import Block


def test_long_context_hc_slices_match_dense_reference():
    torch.manual_seed(20260923)
    x = torch.randn(1, 4097, 4, 3, dtype=torch.bfloat16)
    pre = torch.randn(1, 4097, 4)
    post = torch.randn(1, 4097, 4)
    comb = torch.randn(1, 4097, 4, 4)
    flattened = torch.randn(1, 4097, 3, dtype=torch.bfloat16)

    expected_pre = torch.sum(pre.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)
    actual_pre = Block.hc_pre(None, x, pre)
    assert torch.equal(actual_pre, expected_pre)

    expected_post = (
        post.unsqueeze(-1) * flattened.unsqueeze(-2)
        + torch.sum(comb.unsqueeze(-1) * x.unsqueeze(-2), dim=2)
    ).to(flattened.dtype)
    actual_post = Block.hc_post(None, flattened, x, post, comb)
    assert torch.equal(actual_post, expected_post)


def test_long_context_hc_mixes_preserve_per_token_projection(monkeypatch):
    torch.manual_seed(20260923)
    x = torch.randn(1, 4097, 4, 3, dtype=torch.bfloat16)
    hc_fn = torch.randn(24, 12)
    scale = torch.randn(3)
    base = torch.randn(24)
    block = SimpleNamespace(norm_eps=1e-6, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)
    monkeypatch.setattr(model_module, "hc_split_sinkhorn", lambda mixes, *_: mixes)

    flattened = x.flatten(2).float()
    expected = model_module.dense_linear(flattened, hc_fn) * torch.rsqrt(
        flattened.square().mean(-1, keepdim=True) + block.norm_eps
    )
    actual = Block.hc_mixes(block, x, hc_fn, scale, base)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
