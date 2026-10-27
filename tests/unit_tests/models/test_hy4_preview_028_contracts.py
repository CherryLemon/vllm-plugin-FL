# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Hy4-preview registration, MTP metadata and launch selection.

Load the small modules directly so these tests do not import a CUDA extension,
initialize an accelerator, load weights or simulate model numerical correctness.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "vllm_fl/models/hy_v4"


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qsl(values):
    return SimpleNamespace(tolist=lambda: list(values))


def common(starts, *, query=4, prefilling=None):
    return SimpleNamespace(
        max_query_len=query,
        num_reqs=len(starts) - 1,
        num_actual_tokens=starts[-1],
        query_start_loc_cpu=qsl(starts),
        is_prefilling=None if prefilling is None else qsl(prefilling),
        causal=True,
        mm_req_doc_ranges=None,
    )


def test_native_graph_hook_is_optional_until_requested(monkeypatch):
    registration = load("hy4_registration_contract", "register.py")
    flags = (
        "HY4_MTP_DRAFT_FULL_GRAPH",
        "HY4_MTP_FULL_DRAFT3_GRAPH",
        "HY4_MTP_TAIL2_GRAPH",
        "HY4_PREFILL_BREAKABLE_GRAPH",
        "HY4_MTP_QSL_GPU_STAGING",
        "FLAGOS_ASYNC_INPUT_STAGING",
        "HY4_SPEC_COMMON_REQUEST_IDS",
    )
    for flag in flags:
        monkeypatch.delenv(flag, raising=False)
    registration.validate_native_graph_hooks(object)
    for flag in flags:
        with monkeypatch.context() as scope:
            scope.setenv(flag, "1")
            with pytest.raises(RuntimeError, match="native vLLM 0.28"):
                registration.validate_native_graph_hooks(object)
            registration.validate_native_graph_hooks(
                type("PatchedProposer", (), {"_hy4_call_model": lambda: None})
            )


def test_registration_connects_config_model_and_mtp_hooks(monkeypatch):
    registration = load("hy4_registry_contract", "register.py")
    registration.__package__ = "hy4_registry_fixture"
    configs, models = {}, {}
    text_config, vl_config = type("TextConfig", (), {}), type("VLConfig", (), {})

    class SpeculativeConfig:
        @staticmethod
        def hf_config_override(config):
            return config

        def __post_init__(self):
            return None

    modules = {
        "transformers": {
            "AutoConfig": SimpleNamespace(
                register=lambda key, cls, **kwargs: configs.update({key: cls})
            )
        },
        "vllm.model_executor.models": {
            "ModelRegistry": SimpleNamespace(
                register_model=lambda key, target: models.update({key: target})
            )
        },
        "vllm.config.speculative": {"SpeculativeConfig": SpeculativeConfig},
        "hy4_registry_fixture.config": {
            "HYV4Config": text_config,
            "HYV4VLConfig": vl_config,
        },
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    mtp = load("hy4_registry_fixture.mtp_config", "mtp_config.py")
    monkeypatch.setitem(sys.modules, "hy4_registry_fixture.mtp_config", mtp)
    monkeypatch.setattr(registration, "validate_native_graph_hooks", lambda: None)
    registration.register_hy_v4()
    assert configs == {"hy_v4": text_config, "hy_v4_vl": vl_config}
    assert set(models) == {
        "HYV4ForCausalLM",
        "HYV4VLForConditionalGeneration",
        "HYV4MTP",
    }
    for target in models.values():
        module, _ = target.split(":")
        assert (ROOT / Path(*module.split("."))).with_suffix(".py").is_file()
    assert SpeculativeConfig.hf_config_override is mtp.hyv4_mtp_hf_config_override
    assert SpeculativeConfig.__post_init__ is mtp.hyv4_mtp_speculative_post_init


def test_outer_architecture_keeps_normal_text_inputs_and_rejects_media(monkeypatch):
    class CausalModel:
        def forward(self, *args, **kwargs):
            return args, kwargs

        def load_weights(self, weights):
            return list(weights)

    class TextPositions:
        def __init__(self, count):
            self.count = count

        def repeat(self, axes, copies):
            assert copies == 1
            return [list(range(self.count)) for _ in range(axes)]

    fixtures = {
        "torch": {
            "long": "long",
            "arange": lambda count, **kwargs: TextPositions(count),
        },
        "vllm.model_executor.models.interfaces": {
            "SupportsMRoPE": type("MRoPE", (), {})
        },
        "hy4_text_fixture.model": {"HYV4ForCausalLM": CausalModel},
    }
    for name, attrs in fixtures.items():
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    wrapper = load("hy4_text_fixture.text_compat", "nvidia/text_compat.py")
    model = wrapper.HYV4VLForConditionalGeneration()
    ids, positions, intermediate, embeddings = [object() for _ in range(4)]
    args, kwargs = model.forward(ids, positions, intermediate, embeddings)
    assert args == (ids, positions, intermediate, embeddings)
    assert kwargs == {}
    assert model.supports_multimodal is False
    assert model.get_mrope_input_positions([7, 8, 9], []) == (
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        0,
    )
    for key in (
        "pixel_values",
        "pixel_values_videos",
        "image_embeds",
        "multi_modal_data",
    ):
        with pytest.raises(NotImplementedError, match="text-only"):
            model.forward(ids, positions, **{key: object()})
    with pytest.raises(NotImplementedError, match="text-only"):
        model.get_mrope_input_positions([7], [object()])
    with pytest.raises(NotImplementedError, match="text-only"):
        model.embed_multimodal(pixel_values=object())
    with pytest.raises(NotImplementedError, match="text-only"):
        model.get_placeholder_str("image", 0)
    value = object()
    assert model.load_weights(
        [
            ("model.embed_tokens.weight", value),
            ("language_model.lm_head.weight", value),
            ("vit.embeddings.weight", object()),
            ("visual.layers.0.weight", object()),
        ]
    ) == [("model.embed_tokens.weight", value), ("lm_head.weight", value)]


def test_real_text_config_accepts_checkpoint_header_without_media_components(
    monkeypatch,
):
    pytest.importorskip("transformers")
    pytest.importorskip("huggingface_hub.dataclasses")
    spec = importlib.util.spec_from_file_location(
        "hy4_real_config_contract", PACKAGE / "config.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    quantization = {"format": "int-quantized"}
    config = module.HYV4VLConfig(
        text_config={"hidden_size": 6144, "num_hidden_layers": 78},
        quantization_config=quantization,
        vision_config={"hidden_size": 1152},
        image_token_id=120120,
    )
    assert config.hidden_size == 6144
    assert config.num_hidden_layers == 78
    assert config.get_text_config() is config.text_config
    assert config.text_config.quantization_config == quantization
    assert config.architectures == ["HYV4VLForConditionalGeneration"]
    assert config.text_config.rope_parameters["mrope_section"] == [12, 10, 10]
    assert not hasattr(config, "vision_config")
    assert not hasattr(config, "image_token_id")


def test_uniform_decode_plan_and_first_proposal_boundary():
    graph = load("hy4_draft_contract", "nvidia/mtp_draft_graph.py")
    sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    assert graph.decode_plan(common([0, 4, 8, 12]), sizes, verified_step=True) == (
        4,
        16,
        4,
    )
    assert graph.decode_plan(common([0, 4]), sizes, verified_step=False) is None
    assert graph.decode_plan(common([0, 3, 8]), sizes, verified_step=True) is None
    assert (
        graph.decode_plan(common([0, 4], prefilling=[True]), sizes, verified_step=True)
        is None
    )
    assert (
        graph.decode_plan(common(list(range(66)), query=1), sizes, verified_step=True)
        is None
    )


def test_hy4_config_override_copies_only_the_draft():
    mtp = load("hy4_mtp_config_contract", "mtp_config.py")
    mtp._ORIGINAL_HF_CONFIG_OVERRIDE = lambda config: ("native", config)
    other = SimpleNamespace(model_type="other")
    assert mtp.hyv4_mtp_hf_config_override(other) == ("native", other)

    class Config(SimpleNamespace):
        def update(self, values):
            self.__dict__.update(values)

    text = Config(num_nextn_predict_layers=1, quantization_config=None)
    outer = Config(
        model_type="hy_v4_vl",
        text_config=text,
        quantization_config={"format": "int-quantized"},
    )
    draft = mtp.hyv4_mtp_hf_config_override(outer)
    assert draft.architectures == ["HYV4MTP"]
    assert draft.model_type == "mtp"
    assert draft.n_predict == 1
    assert draft.quantization_config == outer.quantization_config
    assert not hasattr(text, "architectures")
    assert text.quantization_config is None
    with pytest.raises(ValueError, match="at least one MTP"):
        mtp.hyv4_mtp_hf_config_override(
            Config(model_type="hy_v4", num_nextn_predict_layers=0)
        )


def test_mtp_quantization_aliases_preserve_target_metadata():
    mtp = load("hy4_quant_alias_contract", "mtp_config.py")
    original = SimpleNamespace(
        target_scheme_map={
            "model.mtp_layers.0.eh_proj": {"bits": 8},
            "model.layers.0.attn": {"bits": 8},
        },
        ignore=["model.mtp_layers.0.shared_head"],
    )
    mapped = mtp.remap_compressed_mtp_quant_config(original, 78, 1)
    assert mapped.target_scheme_map["model.layers.78.eh_proj"] == {"bits": 8}
    assert "model.layers.0.attn" in mapped.target_scheme_map
    assert "model.layers.78.shared_head" in mapped.ignore
    assert "model.layers.78.eh_proj" not in original.target_scheme_map
    assert original.ignore == ["model.mtp_layers.0.shared_head"]


def test_expert_parallel_mtp_requires_shared_tp_size():
    mtp = load("hy4_parallel_contract", "mtp_config.py")
    calls = []
    mtp._ORIGINAL_SPECULATIVE_POST_INIT = lambda self: calls.append(self)
    target = SimpleNamespace(
        enable_expert_parallel=True,
        tensor_parallel_size=16,
        nnodes=2,
        node_rank=1,
        world_size=16,
    )
    spec = SimpleNamespace(
        method="mtp",
        draft_tensor_parallel_size=16,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["HYV4MTP"])
        ),
        target_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="hy_v4")
        ),
        target_parallel_config=target,
    )
    mtp.hyv4_mtp_speculative_post_init(spec)
    assert calls == [spec]
    assert spec.draft_parallel_config is not target
    assert vars(spec.draft_parallel_config) == vars(target)
    spec.draft_tensor_parallel_size = 8
    with pytest.raises(ValueError, match="draft_tensor_parallel_size"):
        mtp.hyv4_mtp_speculative_post_init(spec)
