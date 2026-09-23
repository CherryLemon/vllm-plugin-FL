# SPDX-License-Identifier: Apache-2.0
"""Long Engram prefill keeps token-local lookup and gating semantics."""

import torch

from vllm_fl.strict028.models.deepseek_v41.model import Engram


class TinyEngram:
    forward = Engram.forward
    _forward_chunk = Engram._forward_chunk

    def __init__(self):
        self.dim = 3
        self.hc_mult = 4
        self.clamp_value = 1e-6
        self.eps = 1e-6
        self.embed = torch.nn.Embedding(8, 2)
        self.wkv = torch.nn.Linear(4, 15, bias=False)
        self.q_weight = torch.randn(4, 3)
        self.k_weight = torch.randn(4, 3)


def test_long_engram_matches_full_sequence_reference():
    torch.manual_seed(20260923)
    module = TinyEngram()
    x = torch.randn(1, 4097, 4, 3, dtype=torch.bfloat16)
    hashes = torch.randint(0, 8, (1, 4097, 2))
    mask = torch.rand(1, 4097) > 0.2

    expected = module._forward_chunk(x, hashes, mask)
    actual = module.forward(x, hashes, mask)
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    assert torch.equal(actual[:, ~mask[0]], x[:, ~mask[0]])
