"""Core Qwen3.8 framework contracts; numerical tests live in FlagGems-vllm."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_fl.patches.qwen3_8_flash_next import (
    Qwen3_8FlashNextForCausalLMConfig,
    apply_native_index_select_policy,
    needs_native_index_select,
    should_skip_generic_flaggems_aten,
)


def _config(model_type: str):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        )
    )


def test_nvidia_qwen4_skips_generic_flaggems_aten_by_default():
    config = _config("qwen4_exp_text")
    assert should_skip_generic_flaggems_aten(
        config, vendor_name="nvidia", whitelist=None
    )


def test_explicit_whitelist_overrides_nvidia_generic_aten_policy():
    config = _config("qwen4_exp_text")
    assert not should_skip_generic_flaggems_aten(
        config, vendor_name="nvidia", whitelist=["silu"]
    )


def test_policy_is_idempotent_and_preserves_explicit_whitelist():
    config = _config("qwen3_8_flash_next_text")
    _, blacklist = apply_native_index_select_policy(config, None, ["index_select"])
    assert blacklist == [
        "index_select",
        "conv1d",
        "_conv_depthwise2d",
        "conv2d",
        "pad",
        "constant_pad_nd",
    ]
    whitelist, blacklist = apply_native_index_select_policy(config, ["add"], ["copy_"])
    assert whitelist == ["add"]
    assert blacklist == ["copy_"]


def test_policy_does_not_change_other_models():
    config = _config("qwen3_text")
    assert not needs_native_index_select(config)
    whitelist, blacklist = apply_native_index_select_policy(config, None, ["copy_"])
    assert whitelist is None
    assert blacklist == ["copy_"]


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


@pytest.mark.parametrize("async_scheduling", [None, True])
@pytest.mark.parametrize(
    "config_cls",
    [Qwen3_8FlashNextForCausalLMConfig],
)
def test_gpu_history_preserves_async_choice(config_cls, async_scheduling):
    config = _vllm_config(ple_layer_ids=[1], async_scheduling=async_scheduling)
    config_cls.verify_and_update_config(config)
    assert config.scheduler_config.async_scheduling is async_scheduling


@pytest.fixture
def cpu_history(monkeypatch):
    from vllm_fl.worker.model_runner import ModelRunnerFL

    monkeypatch.setattr(ModelRunnerFL, "ple_ngram_context_source", "cpu")


@pytest.mark.parametrize(
    "config_cls",
    [
        Qwen3_8FlashNextForCausalLMConfig,
    ],
)
def test_explicit_asynchronous_scheduling_is_rejected(config_cls, cpu_history):
    config = _vllm_config(ple_layer_ids=[1], async_scheduling=True)
    with pytest.raises(NotImplementedError, match="--no-async-scheduling"):
        config_cls.verify_and_update_config(config)
