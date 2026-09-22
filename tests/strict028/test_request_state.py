# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_fl.strict028.cache import RequestState


def make_state():
    model = nn.Module()
    model.attn = nn.Module()
    model.attn.register_buffer(
        "window_kv_cache", torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    )
    model.attn.register_buffer(
        "compress_kv_cache", torch.zeros(1, 8, 8, dtype=torch.bfloat16)
    )
    model.attn.register_buffer("k_cache", torch.zeros(1, 8, 4, dtype=torch.bfloat16))
    model.attn.register_buffer("kv_state", torch.zeros(1, 2, 8))
    model.attn.register_buffer("score_state", torch.full((1, 2, 8), -torch.inf))
    model.attn.register_buffer("freqs_cis", torch.ones(16, 4, dtype=torch.complex64))
    model.engram_hash = nn.Module()
    model.engram_hash.register_buffer("cache", torch.empty(1, 16, dtype=torch.int64))
    state = RequestState(model, 16)
    config = KVCacheConfig(
        3,
        [KVCacheTensor(3 * state.page_bytes, ["fl_request_state"])],
        [KVCacheGroupSpec(["fl_request_state"], state.spec)],
    )
    state.allocate(config, "cpu")
    return model, state, config


def test_interleaved_requests_preserve_all_persistent_state_and_reset_reused_page():
    model, state, _ = make_state()
    assert len(state.fields) == 6
    static = model.attn.freqs_cis
    state.bind(1, reset=True)
    assert model.attn.score_state.isneginf().all()
    for field in state.fields:
        getattr(model.get_submodule(field.module), field.name).fill_(11)
    state.bind(2, reset=True)
    assert model.engram_hash.cache.count_nonzero() == 0
    assert model.attn.k_cache.count_nonzero() == 0
    for field in state.fields:
        getattr(model.get_submodule(field.module), field.name).fill_(22)
    state.bind(1)
    for field in state.fields:
        assert torch.all(getattr(model.get_submodule(field.module), field.name) == 11)
    state.bind(1, reset=True)
    assert model.attn.score_state.isneginf().all()
    assert model.engram_hash.cache.count_nonzero() == 0
    state.bind(2)
    assert torch.all(model.engram_hash.cache == 22)
    assert model.attn.freqs_cis is static


def test_spec_accounts_for_whole_state_and_preserves_builtin_registration():
    _, state, _ = make_state()
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from vllm_fl.strict028.platform import PlatformFL028

    builtin = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
    )
    assert KVCacheSpecRegistry.get_manager_class(builtin) is FullAttentionManager
    PlatformFL028.register_custom_kv_cache_specs(None)
    assert KVCacheSpecRegistry.get_manager_class(state.spec) is FullAttentionManager
    assert state.page_bytes >= sum(field.nbytes for field in state.fields)
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=16),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    assert state.spec.max_memory_usage_bytes(config) == state.page_bytes
    assert state.spec.max_num_blocks_per_req(config, 16) == 1


def test_mismatched_cache_byte_count_is_rejected():
    _, state, config = make_state()
    config.kv_cache_tensors[0].size -= 1
    with pytest.raises(ValueError, match="allocation mismatch"):
        state.allocate(config, "cpu")


def test_original_parameter_names_do_not_misclassify_root_image_embeddings():
    from vllm_fl.strict028.models.deepseek_v41.loader import canonical_name, shard_axis

    assert shard_axis("image_start") is None
    assert shard_axis("layers.0.hc_attn_fn") is None
    assert shard_axis("layers.0.attn.attn_sink") == 0
    assert shard_axis("layers.0.attn.wo_b.scale") == 1
    assert (
        canonical_name("model.layers.1.mlp.gate.e_score_correction_bias_vl")
        == "layers.1.ffn.gate.bias_vl"
    )
