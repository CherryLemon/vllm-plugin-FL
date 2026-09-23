# SPDX-License-Identifier: Apache-2.0
"""Long-prefill Hyper-Connections keep the short-path arithmetic."""

import torch

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
