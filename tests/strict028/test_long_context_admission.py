# SPDX-License-Identifier: Apache-2.0
"""The 32K experiment must be explicit and bounded."""

from types import SimpleNamespace

import pytest
import torch

import vllm  # noqa: F401 - initialize the host before importing its plugin platform

from vllm_fl.strict028.platform import PlatformFL028


def config(max_model_len):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV41FlashFLForCausalLM"]),
            dtype=torch.bfloat16,
            enforce_eager=True,
            max_model_len=max_model_len,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=8,
            nnodes_within_dp=1,
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False,
            async_scheduling=False,
            max_num_seqs=16,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            cache_dtype="auto",
            num_gpu_blocks_override=None,
        ),
        lora_config=None,
        kv_transfer_config=None,
        speculative_config=None,
        load_config=SimpleNamespace(load_format="fl_dsv41"),
    )


def test_long_context_needs_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_FL_EXPERIMENTAL_LONG_CONTEXT", raising=False)
    with pytest.raises(ValueError, match="VLLM_FL_EXPERIMENTAL_LONG_CONTEXT"):
        PlatformFL028.check_and_update_config(config(33792))

    monkeypatch.setenv("VLLM_FL_EXPERIMENTAL_LONG_CONTEXT", "1")
    admitted = config(33792)
    PlatformFL028.check_and_update_config(admitted)
    assert admitted.cache_config.block_size == 33792
    assert admitted.cache_config.num_gpu_blocks_override == 17

    with pytest.raises(ValueError, match="max_model_len<=33792"):
        PlatformFL028.check_and_update_config(config(33793))


def test_mixed_axis_decode_requires_explicit_graph_pd_contract(monkeypatch):
    cfg = config(256)
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.parallel_config.data_parallel_size = 4
    cfg.parallel_config.enable_expert_parallel = True
    cfg.scheduler_config.max_num_seqs = 20
    cfg.kv_transfer_config = SimpleNamespace(
        kv_connector="DeepseekV41FLConnector",
        kv_connector_module_path="vllm_fl.strict028.pd_connector",
        kv_role="kv_consumer",
    )
    monkeypatch.setenv("VLLM_FL_TP_BACKEND", "flagcx")
    for name in (
        "VLLM_FL_EXPERIMENTAL_DP",
        "VLLM_FL_BATCHED_DECODE",
        "VLLM_FL_DECODE_GRAPH",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="experimental DP"):
        PlatformFL028.check_and_update_config(cfg)

    for name in (
        "VLLM_FL_EXPERIMENTAL_DP",
        "VLLM_FL_BATCHED_DECODE",
        "VLLM_FL_DECODE_GRAPH",
    ):
        monkeypatch.setenv(name, "1")
    PlatformFL028.check_and_update_config(cfg)
    assert cfg.cache_config.num_gpu_blocks_override == 21
    cfg.kv_transfer_config.kv_role = "kv_producer"
    with pytest.raises(ValueError, match="experimental DP"):
        PlatformFL028.check_and_update_config(cfg)


def test_128k_prefill_requires_chunked_execution(monkeypatch):
    monkeypatch.setenv("VLLM_FL_EXPERIMENTAL_LONG_CONTEXT", "1")
    monkeypatch.delenv("VLLM_FL_CHUNKED_PREFILL", raising=False)
    cfg = config(139264)
    with pytest.raises(ValueError, match="max_model_len<=33792"):
        PlatformFL028.check_and_update_config(cfg)
    monkeypatch.setenv("VLLM_FL_CHUNKED_PREFILL", "1")
    with pytest.raises(ValueError, match="matching"):
        PlatformFL028.check_and_update_config(cfg)
    cfg.scheduler_config.enable_chunked_prefill = True
    PlatformFL028.check_and_update_config(cfg)
    assert cfg.cache_config.block_size == 139264
    cfg.model_config.max_model_len += 1
    with pytest.raises(ValueError, match="max_model_len<=139264"):
        PlatformFL028.check_and_update_config(cfg)


def test_128k_consumer_requires_batched_graph(monkeypatch):
    monkeypatch.setenv("VLLM_FL_EXPERIMENTAL_LONG_CONTEXT", "1")
    monkeypatch.setenv("VLLM_FL_TP_BACKEND", "flagcx")
    cfg = config(139264)
    cfg.kv_transfer_config = SimpleNamespace(
        kv_connector="DeepseekV41FLConnector",
        kv_connector_module_path="vllm_fl.strict028.pd_connector",
        kv_role="kv_consumer",
    )
    for name in ("VLLM_FL_BATCHED_DECODE", "VLLM_FL_DECODE_GRAPH"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="max_model_len<=33792"):
        PlatformFL028.check_and_update_config(cfg)
    for name in ("VLLM_FL_BATCHED_DECODE", "VLLM_FL_DECODE_GRAPH"):
        monkeypatch.setenv(name, "1")
    PlatformFL028.check_and_update_config(cfg)
