# SPDX-License-Identifier: Apache-2.0
"""The 32K experiment must be explicit and bounded."""

from types import SimpleNamespace

import pytest
import torch

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
