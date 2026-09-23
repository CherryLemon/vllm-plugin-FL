# SPDX-License-Identifier: Apache-2.0
"""The bounded long-prefill candidate representation preserves visibility."""

import torch

from vllm_fl.strict028.models.deepseek_v41.model import (
    expand_candidate_blocks,
    select_candidate_blocks,
)


def test_packed_candidate_blocks_match_dense_masks_for_partial_visibility():
    torch.manual_seed(20260923)
    logits = torch.randn(1, 17, 37)
    visible = torch.tensor(
        [0, 1, 7, 8, 9, 13, 16, 17, 20, 24, 25, 29, 32, 33, 35, 36, 37]
    ).view(1, -1, 1)
    logits.masked_fill_(torch.arange(37) >= visible, -torch.inf)

    dense = select_candidate_blocks(logits, visible, 3, 8)
    packed = select_candidate_blocks(logits, visible, 3, 8, packed=True)

    assert packed.dtype == torch.int32
    assert packed.shape == (1, 17, 3)
    assert torch.equal(expand_candidate_blocks(packed, 37, 8), dense)
    assert not dense[0, 0].any()
