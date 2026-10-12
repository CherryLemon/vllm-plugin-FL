# SPDX-License-Identifier: Apache-2.0
"""GLM activation defaults use the same mutable public dispatch policy."""

from types import SimpleNamespace

import pytest

from vllm_fl import activation
from vllm_fl.dispatch.policy import PolicyManager, SelectionPolicy
from vllm_fl.kernels.glm5_next import provider
from vllm_fl.patches import glm5_next_runtime as glm_patch


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    activation.reset_activation_for_tests()
    monkeypatch.setattr(PolicyManager, "_instance", PolicyManager())
    provider._has_nvidia_reference_kernels.cache_clear()
    yield
    activation.reset_activation_for_tests()
    provider._has_nvidia_reference_kernels.cache_clear()


def _glm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="glm5_next_text", architectures=["Glm5NextForCausalLM"]
            ),
            hf_text_config=SimpleNamespace(model_type="glm5_next_text"),
            architectures=["Glm5NextForCausalLM"],
        )
    )


def _other_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="llama", architectures=["LlamaForCausalLM"]
            ),
            hf_text_config=SimpleNamespace(model_type="llama"),
            architectures=["LlamaForCausalLM"],
        )
    )


def _portable_registered_plan(monkeypatch):
    monkeypatch.setattr(glm_patch, "use_nvidia_reference", lambda: False)
    assert glm_patch.register_glm5_next_support() is True
    plan = activation.resolve_model_plan(_glm_config())
    assert plan is not None
    # Actual registration/defaults, with only the class-patching callback held
    # at its boundary. Real patch ownership is covered by activation tests.
    monkeypatch.setattr(glm_patch, "_apply_glm5_activation", lambda: None)
    return activation.resolve_model_plan(_glm_config())


def test_glm_activation_adds_only_required_moe_defaults(monkeypatch):
    _portable_registered_plan(monkeypatch)
    manager = PolicyManager.get_instance()
    user = SelectionPolicy.from_dict(prefer="vendor", deny_vendors={"cuda"})
    manager.set_global_policy(user)
    assert activation.activate_for_model(_glm_config()) is not None
    policy = manager.get_policy()
    for op in ("moe_align_block_size", "invoke_fused_moe_triton_kernel"):
        assert policy.get_per_op_order(op) == ["flagos", "reference"]
    assert policy.prefer == "vendor"
    assert policy.deny_vendors == user.deny_vendors
    # Indexer defaults are read through common dispatch, rather than frozen
    # into an additional explicit per-op policy at startup.
    assert policy.get_per_op_order("fp8_fp4_mqa_logits") is None
    assert policy.get_default_order()[0] == "vendor"


def test_active_glm_preserves_explicit_policy_and_later_preference(monkeypatch):
    _portable_registered_plan(monkeypatch)
    manager = PolicyManager.get_instance()
    manager.set_global_policy(
        SelectionPolicy.from_dict(
            prefer="vendor", per_op_order={"moe_align_block_size": ["flagos"]}
        )
    )
    activation.activate_for_model(_glm_config())
    assert manager.get_policy().get_per_op_order("moe_align_block_size") == ["flagos"]
    manager.set_global_policy(
        SelectionPolicy.from_dict(
            prefer="reference", per_op_order={"moe_align_block_size": ["flagos"]}
        )
    )
    policy = manager.get_policy()
    assert policy.get_per_op_order("moe_align_block_size") == ["flagos"]
    assert policy.get_per_op_order("fp8_fp4_mqa_logits") is None
    assert policy.get_default_order()[0] == "reference"


def test_glm_provider_leaves_other_model_policy_unchanged(monkeypatch):
    _portable_registered_plan(monkeypatch)
    manager = PolicyManager.get_instance()
    user = SelectionPolicy.from_dict(prefer="reference")
    manager.set_global_policy(user)
    assert activation.activate_for_model(_other_config()) is None
    assert manager.get_policy() is user
    assert manager.get_policy().get_per_op_order("moe_align_block_size") is None


@pytest.mark.parametrize("sparse", [False, True])
def test_attention_override_tracks_public_policy_and_native_capability(
    monkeypatch, sparse
):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm_fl.dispatch.policy import policy_context

    available_checks = []
    native_capability = [True]
    monkeypatch.setattr(glm_patch, "use_nvidia_reference", lambda: native_capability[0])
    monkeypatch.setattr(
        FlagGemsBackend, "is_available", lambda self: available_checks.append(1) or True
    )
    op = "flash_mla_sparse_fwd" if sparse else "attention"
    vendor = SelectionPolicy.from_dict(prefer="vendor")
    flagos = SelectionPolicy.from_dict(prefer="vendor", per_op_order={op: ["flagos"]})
    denied = SelectionPolicy.from_dict(prefer="vendor", deny_vendors={"cuda"})
    expected = "FlagGemsSparseMLABackend" if sparse else "MLAFLBackend"
    # Begin with a valid native selection, then change only one public condition.
    for policy, native, uses_library in (
        (vendor, True, False),
        (flagos, True, True),
        (vendor, True, False),
        (denied, True, True),
        (vendor, False, True),
    ):
        native_capability[0] = native
        before = len(available_checks)
        with policy_context(policy):
            result = glm_patch._glm5_attention_override(True, sparse)
        if uses_library:
            assert result.endswith(expected)
            assert len(available_checks) == before + 1
        else:
            assert result is None
            assert len(available_checks) == before
    # Non-MLA must bypass model-specific capability/dependency work entirely.
    before = len(available_checks)
    with policy_context(flagos):
        assert glm_patch._glm5_attention_override(False, sparse) is None
    assert len(available_checks) == before
