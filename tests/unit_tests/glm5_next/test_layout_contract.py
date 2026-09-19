# SPDX-License-Identifier: Apache-2.0
"""The real runner honors explicit GLM layouts and preserves other backends."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_fl.models.glm5_next_kpool import Glm5NextIndexerAttentionBackend
from vllm_fl.runtime.kv_layout import get_physical_cache_layout


class OtherCompressedBackend:
    @staticmethod
    def get_kv_cache_shape(blocks, page, heads, width, **kwargs):
        return blocks, page, heads, width


def run_reshape(backend, spec, kernel_size):
    from vllm_fl.worker.model_runner import ModelRunnerFL

    group = SimpleNamespace(
        kv_cache_spec=spec,
        backend=backend,
        kv_cache_group_id=0,
        layer_names=["indexer"],
    )
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_tensors=[], kv_cache_groups=[group]),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        runner_only_attn_layers=set(),
        _kv_cache_spec_attn_group_iterator=lambda: iter([group]),
    )
    raw = torch.zeros(spec.page_size_bytes * 3, dtype=torch.uint8)
    output = ModelRunnerFL._reshape_kv_cache_tensors(
        runner, {"indexer": raw}, [kernel_size]
    )
    return raw, output["indexer"]


@pytest.mark.parametrize("storage", [48, 64, 96])
def test_other_compressed_backend_keeps_own_page_size(storage):
    spec = MLAAttentionSpec(
        block_size=storage * 4,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
    )
    assert get_physical_cache_layout(OtherCompressedBackend, spec) is None
    raw, cache = run_reshape(OtherCompressedBackend, spec, spec.block_size)
    assert cache.shape == (3, storage, 1, 132)
    assert cache.data_ptr() == raw.data_ptr()


@pytest.mark.parametrize("logical,padding", [(256, 0), (512, 0), (512, 256)])
def test_glm_padded_pages_are_views_with_shared_descriptor(
    logical, padding, monkeypatch
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: SimpleNamespace(major=9)
    )
    spec = MLAAttentionSpec(
        block_size=logical,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        page_size_padded=logical // 4 * 132 + padding,
        indexes_kv_by_block_stride=True,
    )
    layout = get_physical_cache_layout(Glm5NextIndexerAttentionBackend, spec)
    raw, cache = run_reshape(Glm5NextIndexerAttentionBackend, spec, 64)
    assert cache.shape[:2] == (3 * layout.pages_per_block, layout.kernel_block_size)
    assert cache.stride(0) == spec.page_size_bytes // layout.pages_per_block
    assert cache.data_ptr() == raw.data_ptr()
    cache[-1, -1, ...] = 17
    assert raw.eq(17).any()


def test_metadata_builder_delegates_non_glm_and_uses_same_glm_layout(monkeypatch):
    from vllm.platforms import current_platform
    from vllm.v1.worker.utils import AttentionGroup

    from vllm_fl.patches import glm5_next_kpool_v024 as hooks

    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: SimpleNamespace(major=9)
    )
    delegated = []
    monkeypatch.setitem(
        hooks._RUNTIME_BASELINES,
        hooks._kpool_target(AttentionGroup, "create_metadata_builders"),
        lambda *args: delegated.append(args[0]),
    )
    patched = hooks._create_metadata_builders_patch("test-layout").replacement
    spec = MLAAttentionSpec(
        block_size=512,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
    )
    group = SimpleNamespace(
        backend=OtherCompressedBackend, kv_cache_spec=spec, layer_names=["indexer"]
    )
    patched(group, None, torch.device("cpu"), 64)
    assert delegated == [group]
    monkeypatch.setattr(
        Glm5NextIndexerAttentionBackend,
        "get_builder_cls",
        lambda: lambda spec, *args: SimpleNamespace(kv_cache_spec=spec),
    )
    group.backend = Glm5NextIndexerAttentionBackend
    patched(group, None, torch.device("cpu"), 64)
    builder = group.metadata_builders[0]
    assert builder.kv_cache_spec.storage_block_size == 64
    assert builder._glm5_physical_layout.pages_per_block == 2
