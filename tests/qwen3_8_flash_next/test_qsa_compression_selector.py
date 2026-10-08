# SPDX-License-Identifier: Apache-2.0
"""Compression selection follows the layout supported by the fused kernel."""

from types import SimpleNamespace

import pytest

from vllm_fl.models.qwen3_8_flash_next.gpu import indexer_qsa
from vllm_fl.models.qwen3_8_flash_next.gpu.ops import qsa


@pytest.mark.parametrize(
    "enabled,dim,neox,fused",
    [
        (True, 128, True, True),
        (False, 128, True, False),
        (True, 64, True, False),
        (True, 128, False, False),
    ],
)
def test_compression_selector_matches_supported_layout(
    monkeypatch, enabled, dim, neox, fused
):
    monkeypatch.setattr(indexer_qsa, "_QSA_FUSED_COMPRESS_ENABLED", enabled)
    instance = indexer_qsa.QSAIndexer.__new__(indexer_qsa.QSAIndexer)
    instance.index_head_dim = dim
    instance.select_all_tokens = False
    instance.rotary_emb = SimpleNamespace(rotary_dim=64, is_neox_style=neox)
    selected = instance._compression_impl()
    assert selected is (
        qsa.qsa_compress_norm_mrope_store_groups
        if fused
        else qsa.qsa_compress_groups_with_ratio
    )
