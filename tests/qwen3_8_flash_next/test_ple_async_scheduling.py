"""CPU checks for the PLE asynchronous-scheduling contract.

vLLM resolves ``async_scheduling=None`` later than the model config hook, so
the hook must pin the default when the runner only has a CPU token history.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from vllm_fl.patches.qwen3_8_flash_next import (
    Qwen3_8FlashNextForCausalLMConfig,
    Qwen3_8FlashNextForConditionalGenerationConfig,
    ple_ngram_context_uses_cpu_history,
)


def _vllm_config(*, ple_layer_ids, async_scheduling):
    text_config = SimpleNamespace(
        hc_count=4,
        ple_layer_ids=ple_layer_ids,
        mamba_ssm_dtype="float32",
        indexer_n_heads=None,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=text_config,
            hf_config=text_config,
            multimodal_config=None,
        ),
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            ubatch_size=1,
            pipeline_parallel_size=1,
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )


def test_runner_declares_cpu_token_history():
    assert ple_ngram_context_uses_cpu_history() is True


def test_unset_async_scheduling_is_pinned_to_synchronous(caplog):
    config = _vllm_config(ple_layer_ids=[1], async_scheduling=None)
    with caplog.at_level(logging.INFO):
        Qwen3_8FlashNextForCausalLMConfig.verify_and_update_config(config)
    assert config.scheduler_config.async_scheduling is False
    assert any(
        "n-gram token history on the CPU" in record.getMessage()
        for record in caplog.records
    )


def test_explicit_synchronous_scheduling_is_untouched(caplog):
    config = _vllm_config(ple_layer_ids=[1], async_scheduling=False)
    with caplog.at_level(logging.INFO):
        Qwen3_8FlashNextForCausalLMConfig.verify_and_update_config(config)
    assert config.scheduler_config.async_scheduling is False
    assert not any(
        "n-gram token history on the CPU" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "config_cls",
    [
        Qwen3_8FlashNextForCausalLMConfig,
        Qwen3_8FlashNextForConditionalGenerationConfig,
    ],
)
def test_explicit_asynchronous_scheduling_is_rejected(config_cls):
    config = _vllm_config(ple_layer_ids=[1], async_scheduling=True)
    with pytest.raises(NotImplementedError, match="--no-async-scheduling"):
        config_cls.verify_and_update_config(config)


def test_without_ple_async_scheduling_is_not_restricted():
    config = _vllm_config(ple_layer_ids=[], async_scheduling=None)
    Qwen3_8FlashNextForCausalLMConfig.verify_and_update_config(config)
    assert config.scheduler_config.async_scheduling is None
