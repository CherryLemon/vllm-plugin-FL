# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

from vllm_fl.strict028.validation import copy_reference_buffers


def test_materializes_every_shared_rope_binding_without_aliasing_live_state():
    source, target = nn.ModuleList(), nn.ModuleList()
    shared = torch.arange(8, dtype=torch.float32)
    placeholder = torch.empty(8, device="meta")
    for _ in range(3):
        src, dst = nn.Module(), nn.Module()
        src.register_buffer("rope", shared, persistent=False)
        dst.register_buffer("rope", placeholder, persistent=False)
        source.append(src)
        target.append(dst)
    # Default named_buffers() only returns the first of these three bindings.
    assert len(list(source.named_buffers())) == 1
    copy_reference_buffers(source, target)
    assert target[0].rope is target[1].rope is target[2].rope
    assert target[0].rope is not source[0].rope
    for module in target:
        torch.testing.assert_close(module.rope, shared)
    target[0].rope.zero_()
    assert source[0].rope.sum().item() == 28
