"""QSA KV-cache compatibility tests for the supported vLLM ABIs."""

from __future__ import annotations

import pytest
import torch

# The reference package intentionally has model -> qsa -> model ownership.  Keep
# the model import here so the test mirrors package registration instead of
# importing qsa as a standalone leaf module.
from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import QSAStateBackend
from vllm_fl.models.qwen3_8_flash_next.gpu import model as _model  # noqa: F401
from vllm_fl.models.qwen3_8_flash_next.gpu.qsa import (
    Qwen3_8FlashNextQSAAttentionBackend,
    _unpack_qsa_kv_cache,
)


@pytest.mark.parametrize(
    "backend,expected_indexes,expected_order,expected_layered_order",
    [
        (
            Qwen3_8FlashNextQSAAttentionBackend,
            False,
            (0, 1, 2, 3, 4),
            (0, 1, 2, 3, 4, 5),
        ),
        # The official QSA raw-state metadata uses a block-strided page so the
        # vLLM 0.24 planner can pad its physical page to the main KV page.
        (QSAStateBackend, True, (0, 1, 2, 3), (0, 1, 2, 3, 4)),
    ],
)
def test_qsa_backends_expose_layout_contract(
    backend, expected_indexes, expected_order, expected_layered_order
):
    assert backend.indexes_kv_by_block_stride() is expected_indexes
    assert backend.get_kv_cache_stride_order() == expected_order
    assert backend.get_kv_cache_stride_order(True) == expected_layered_order


def test_qsa_backend_owns_vendor_neutral_legacy_layout():
    assert Qwen3_8FlashNextQSAAttentionBackend.get_kv_cache_shape(
        3, 16, 2, 8
    ) == (3, 2, 16, 2, 8)
    assert Qwen3_8FlashNextQSAAttentionBackend.get_kv_cache_stride_order() == (
        0,
        1,
        2,
        3,
        4,
    )
    assert Qwen3_8FlashNextQSAAttentionBackend.get_kv_cache_stride_order(True) == (
        0,
        1,
        2,
        3,
        4,
        5,
    )


def test_unpack_legacy_vllm_024_cache_layout():
    cache = torch.zeros(3, 2, 16, 2, 8)
    key, value = _unpack_qsa_kv_cache(cache, 8)
    assert key.shape == value.shape == (3, 16, 2, 8)
    assert key.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert value.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert key.storage_offset() == cache.storage_offset()
    assert value.storage_offset() == cache.storage_offset() + cache[0, 0].numel()

    # The QSA cache-update Triton kernel flattens only the contiguous head/dim
    # tail.  This must remain a view and writes must reach the allocator-owned
    # backing storage so no vendor performs a hidden full-cache copy during
    # decode.
    flat_key = key.reshape(3, 16, 1, 16)
    flat_value = value.reshape(3, 16, 1, 16)
    assert (
        flat_key.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    )
    assert (
        flat_value.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    )
    assert flat_key.storage_offset() == key.storage_offset()
    assert flat_value.storage_offset() == value.storage_offset()

    key_update = torch.arange(flat_key.numel(), dtype=cache.dtype).reshape_as(flat_key)
    value_update = -torch.arange(
        flat_value.numel(), dtype=cache.dtype
    ).reshape_as(flat_value)
    flat_key.copy_(key_update)
    flat_value.copy_(value_update)
    torch.testing.assert_close(cache[:, 0], key_update.reshape_as(cache[:, 0]))
    torch.testing.assert_close(cache[:, 1], value_update.reshape_as(cache[:, 1]))


def test_unpack_rejects_packed_vllm_cache_layout():
    cache = torch.arange(3 * 2 * 16 * 16).reshape(3, 2, 16, 16)
    with pytest.raises(ValueError, match="packed 4-D"):
        _unpack_qsa_kv_cache(cache, 8)


@pytest.mark.parametrize(
    "shape,head_size",
    [((3, 3, 16, 2, 8), 8), ((3, 16, 8), 8)],
)
def test_unpack_rejects_unknown_cache_layout(shape, head_size):
    with pytest.raises(ValueError, match="QSA KV cache"):
        _unpack_qsa_kv_cache(torch.empty(shape), head_size)
