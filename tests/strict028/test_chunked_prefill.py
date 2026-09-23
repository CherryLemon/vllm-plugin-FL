# SPDX-License-Identifier: Apache-2.0
"""Chunk boundaries must preserve causal window, compressor and DSpark state."""

import pytest
import torch
from test_batched_decode import make_model

from vllm_fl.strict028.chunked_prefill import ChunkPrefill
from vllm_fl.strict028.models.deepseek_v41.model import set_dtype

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.inference_mode()
@pytest.mark.parametrize(
    "sizes", [(3, 2, 17, 1, 45, 69), (64, 64, 9), (1, 1, 1, 5, 129)]
)
def test_chunked_prefix_matches_full_prefix_and_next_decode(sizes):
    model, state = make_model()
    ids = (torch.arange(sum(sizes), device="cuda") * 7).remainder(64)[None]
    with torch.device("cuda"), set_dtype(torch.bfloat16):
        state.bind(1, reset=True)
        _, expected_logits, expected_hidden = model.core(ids, 0)
        model.core.store_spec_context(expected_hidden, 0)
        state.bind(2, reset=True)
        hiddens = []
        start = 0
        for size in sizes:
            context = ChunkPrefill(start, size, ids.shape[1], ids.device)
            _, logits, hidden = model.core(ids[:, start : start + size], context)
            model.core.store_spec_context(hidden, context)
            hiddens.append(hidden)
            start += size
        torch.testing.assert_close(logits, expected_logits, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            torch.cat(hiddens, 1), expected_hidden, rtol=1e-4, atol=1e-4
        )
        for field in state.fields:
            actual = state.storage[2, field.offset : field.offset + field.nbytes].view(
                field.dtype
            )
            expected = state.storage[
                1, field.offset : field.offset + field.nbytes
            ].view(field.dtype)
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1e-5 if field.dtype == torch.float32 else 0,
                atol=1e-6 if field.dtype == torch.float32 else 0,
                msg=lambda msg, field=field: f"{field.module}.{field.name}: {msg}",
            )
        for position in range(start, start + 3):
            token = expected_logits.argmax(-1)[:, None]
            state.bind(1)
            _, expected_logits, _ = model.core(token, position)
            state.bind(2)
            _, logits, _ = model.core(token, position)
            torch.testing.assert_close(logits, expected_logits, rtol=1e-4, atol=1e-4)
