# SPDX-License-Identifier: Apache-2.0
"""Actual GLM worker boundary validates policy before patches and ATen takeover."""

from types import SimpleNamespace

import pytest

from vllm_fl import activation, flaggems_runtime
from vllm_fl.dispatch.binding import OperatorBinding
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import PolicyManager, SelectionPolicy
from vllm_fl.dispatch.types import BackendImplKind, OpImpl
from vllm_fl.patches import (
    glm5_next_runtime as glm_patch,
    glm5_next_worker as worker_patch,
)
from vllm_fl.utils import use_flaggems_op

_REQUIRED_OPS = ("moe_align_block_size", "invoke_fused_moe_triton_kernel")


def _config(glm=True, arch="Glm5NextForCausalLM"):
    model_type = "glm5_next_text" if glm else "llama"
    architecture = arch if glm else "LlamaForCausalLM"
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type=model_type, architectures=[architecture]
            ),
            hf_text_config=SimpleNamespace(model_type=model_type),
            architectures=[architecture],
            quantization=None,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, enable_eplb=False),
        speculative_config=None,
        quant_config=None,
    )


@pytest.fixture
def startup(monkeypatch, tmp_path):
    import flag_gems

    activation.reset_activation_for_tests()
    monkeypatch.setattr(flaggems_runtime, "_STATE", None)
    monkeypatch.setattr(flaggems_runtime, "_FAILED", False)
    monkeypatch.setattr(worker_patch.fl_envs, "USE_FLAGGEMS", True)
    oplist_path = str(tmp_path / "oplist.txt")
    monkeypatch.setattr(
        worker_patch.fl_envs, "FLAGGEMS_ENABLE_OPLIST_PATH", oplist_path
    )
    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "1")
    monkeypatch.setenv("VLLM_FL_PREFER", "flagos")
    monkeypatch.setenv("USE_FLAGGEMS", "1")
    for name in ("VLLM_FL_FLAGOS_WHITELIST", "VLLM_FL_FLAGOS_BLACKLIST_APPEND"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "mm,linear")
    monkeypatch.setattr(glm_patch, "use_nvidia_reference", lambda: False)

    events = []
    enable_calls = []
    numeric_calls = []
    applied = []
    available = {op: True for op in _REQUIRED_OPS}
    manager = OpManager(register_builtins=False)
    policy_manager = PolicyManager()
    monkeypatch.setattr(PolicyManager, "_instance", policy_manager)
    policy_manager.set_global_policy(SelectionPolicy.from_dict(prefer="vendor"))
    monkeypatch.setattr(worker_patch, "get_default_manager", lambda: manager)
    monkeypatch.setattr(
        worker_patch, "register_oot_ops", lambda: events.append("register")
    )

    def apply():
        events.append("activate")
        applied.append(True)

    monkeypatch.setattr(glm_patch, "_apply_glm5_activation", apply)
    activation.register_plan_provider(glm_patch._glm5_plan_provider)

    def implementation(name, kind, vendor=None):
        def run(*args, **kwargs):
            numeric_calls.append((name, kind))
            return kind.value

        if kind == BackendImplKind.DEFAULT:

            def is_available():
                events.append(f"check:{name}")
                return available[name] and use_flaggems_op(name)

            run._is_available = is_available
        return OpImpl(name, f"{name}.{kind.value}", kind, run, vendor=vendor)

    manager.registry.register_many(
        [
            implementation(op, kind, vendor)
            for op in _REQUIRED_OPS
            for kind, vendor in (
                (BackendImplKind.DEFAULT, None),
                (BackendImplKind.VENDOR, "cuda"),
            )
        ]
    )

    def enable(kind, **kwargs):
        events.append("enable")
        enable_calls.append((kind, kwargs))

    monkeypatch.setattr(flag_gems, "enable", lambda **kwargs: enable("all", **kwargs))
    monkeypatch.setattr(
        flag_gems, "only_enable", lambda **kwargs: enable("only", **kwargs)
    )
    yield SimpleNamespace(
        manager=manager,
        policy_manager=policy_manager,
        events=events,
        enable_calls=enable_calls,
        numeric_calls=numeric_calls,
        applied=applied,
        available=available,
        flag_gems=flag_gems,
        oplist_path=oplist_path,
    )
    activation.reset_activation_for_tests()


def test_whitelist_conflict_precedes_registration_and_activation(startup, monkeypatch):
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST")
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "grouped_topk")
    with pytest.raises(activation.ActivationConflict, match="excludes operators"):
        worker_patch.initialize_glm5_worker(_config(), rank=0)
    assert startup.events == []
    assert startup.applied == []
    assert startup.enable_calls == []
    assert activation.get_active_plan() is None


def test_invalid_model_mode_precedes_registration_and_activation(startup):
    config = _config()
    config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(glm_patch.Glm5ConfigError, match="pipeline_parallel_size"):
        worker_patch.initialize_glm5_worker(config, rank=0)
    assert startup.events == []
    assert startup.applied == []
    assert startup.enable_calls == []
    assert activation.get_active_plan() is None


@pytest.mark.parametrize("failure", ["unavailable", "blacklist", "per_op"])
def test_actual_resolver_rejects_policy_before_patches_or_enable(
    startup, monkeypatch, failure
):
    if failure == "unavailable":
        startup.available[_REQUIRED_OPS[0]] = False
    elif failure == "blacklist":
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", _REQUIRED_OPS[0])
    else:
        startup.policy_manager.set_global_policy(
            SelectionPolicy.from_dict(per_op_order={_REQUIRED_OPS[0]: ["vendor:cuda"]})
        )
    with pytest.raises(activation.ActivationConflict):
        worker_patch.initialize_glm5_worker(_config(), rank=0)
    assert startup.events[0] == "register"
    assert startup.applied == []
    assert startup.enable_calls == []
    assert startup.numeric_calls == []
    assert activation.get_active_plan() is None


@pytest.mark.parametrize(
    "arch", ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"]
)
@pytest.mark.parametrize("rank", [0, 1])
def test_text_and_vision_startup_preserve_phase_order_and_native_aten_blacklist(
    startup, arch, rank
):
    assert worker_patch.initialize_glm5_worker(_config(arch=arch), rank=rank) is True
    assert startup.events[0] == "register"
    assert startup.events[-2:] == ["activate", "enable"]
    for op in _REQUIRED_OPS:
        assert startup.events.index(f"check:{op}") < startup.events.index("activate")
    assert startup.numeric_calls == []
    assert startup.enable_calls == [
        (
            "all",
            dict(
                unused=["mm", "linear"],
                record=rank == 0,
                once=True,
                path=startup.oplist_path,
            ),
        )
    ]
    assert activation.get_active_plan().name == "glm5_next"
    policy = startup.policy_manager.get_policy()
    assert policy.get_per_op_order(_REQUIRED_OPS[0]) == ["flagos", "reference"]
    assert policy.get_per_op_order("fp8_fp4_mqa_logits") is None


def test_explicit_flaggems_whitelist_is_forwarded_without_extension(
    startup, monkeypatch
):
    whitelist = [*_REQUIRED_OPS, "grouped_topk"]
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST")
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", ",".join(whitelist))
    assert worker_patch.initialize_glm5_worker(_config(), rank=0) is True
    assert startup.enable_calls[0] == (
        "only",
        dict(include=whitelist, record=True, once=True, path=startup.oplist_path),
    )


def test_disabled_generic_aten_does_not_enable_flaggems(startup, monkeypatch):
    monkeypatch.setattr(worker_patch.fl_envs, "USE_FLAGGEMS", False)
    assert worker_patch.initialize_glm5_worker(_config(), rank=0) is True
    assert startup.applied == [True]
    assert startup.enable_calls == []
    assert startup.numeric_calls == []


def test_repeated_initialization_preserves_process_owned_patches_and_aten(startup):
    assert worker_patch.initialize_glm5_worker(_config(), rank=0) is True
    assert worker_patch.initialize_glm5_worker(_config(), rank=0) is True
    assert startup.applied == [True]
    assert len(startup.enable_calls) == 1


@pytest.mark.parametrize("glm_first", [False, True])
def test_empty_plan_rejects_cross_model_process_reuse(startup, glm_first):
    first = _config(glm=glm_first)
    assert worker_patch.initialize_glm5_worker(first, rank=0) is glm_first
    assert startup.applied == ([True] if glm_first else [])
    enable_count = len(startup.enable_calls)
    with pytest.raises(activation.ActivationConflict):
        worker_patch.initialize_glm5_worker(_config(glm=not glm_first), rank=0)
    assert startup.applied == ([True] if glm_first else [])
    assert len(startup.enable_calls) == enable_count


def test_ordinary_model_skips_glm_validation_registry_and_flaggems(startup):
    config = _config(glm=False)
    config.parallel_config.pipeline_parallel_size = 2
    config.parallel_config.enable_eplb = True
    config.speculative_config = object()
    config.quant_config = object()
    assert worker_patch.initialize_glm5_worker(config, rank=0) is False
    assert startup.events == []
    assert startup.enable_calls == []
    assert activation.get_active_plan().name == "<no-plan>"


def test_startup_does_not_freeze_indexer_default_selection(startup):
    name = "fp8_fp4_mqa_logits"
    startup.manager.registry.register_many(
        [
            OpImpl(name, "flag", BackendImplKind.DEFAULT, lambda: "flag"),
            OpImpl(name, "cuda", BackendImplKind.VENDOR, lambda: "cuda", vendor="cuda"),
        ]
    )
    assert worker_patch.initialize_glm5_worker(_config(), rank=0) is True
    binding = OperatorBinding(startup.manager, name)
    assert binding() == "cuda"
    startup.policy_manager.set_global_policy(SelectionPolicy.from_dict(prefer="flagos"))
    assert binding() == "flag"
    assert startup.policy_manager.get_policy().get_per_op_order(name) is None


def test_flaggems_enable_failure_is_not_retried_in_the_process(startup, monkeypatch):
    attempts = []

    def fail(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("ATen registration partially failed")

    monkeypatch.setattr(startup.flag_gems, "enable", fail)
    with pytest.raises(RuntimeError, match="partially failed"):
        worker_patch.initialize_glm5_worker(_config(), rank=0)
    with pytest.raises(RuntimeError, match="restart the process"):
        worker_patch.initialize_glm5_worker(_config(), rank=0)
    assert len(attempts) == 1
    assert startup.applied == [True]
    assert startup.numeric_calls == []
