# SPDX-License-Identifier: Apache-2.0
"""CPU runtime and patch contracts; no fabricated native modules or GPU work."""

import ast
import importlib.util
import inspect
import sys
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

PLUGIN = Path(__file__).resolve().parents[3] / "vllm_fl/models/hy_v4"


def source_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, PLUGIN / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def graph(monkeypatch):
    package = ModuleType("hy4_graph_contract")
    package.__path__ = [str(PLUGIN / "nvidia")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    return source_module(
        "hy4_graph_contract.mtp_draft_graph", "nvidia/mtp_draft_graph.py"
    )


def common(q=4, reqs=2):
    return SimpleNamespace(
        max_query_len=q,
        num_reqs=reqs,
        num_actual_tokens=q * reqs,
        query_start_loc_cpu=torch.arange(reqs + 1) * q,
        is_prefilling=torch.zeros(reqs, dtype=torch.bool),
        causal=True,
        mm_req_doc_ranges=None,
    )


def test_draft_plan_accepts_verified_batch(graph):
    assert graph.decode_plan(
        common(), (1, 2, 4, 8, 16, 32, 64, 128, 256), verified_step=True
    ) == (4, 8, 2)


@pytest.mark.parametrize(
    "condition",
    ["unverified", "nonuniform", "prefill", "noncausal", "media", "capacity"],
)
def test_draft_plan_declines_before_running_model(graph, condition):
    c = common()
    verified = True
    sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    if condition == "unverified":
        verified = False
    elif condition == "nonuniform":
        c.query_start_loc_cpu[1] = 3
    elif condition == "prefill":
        c.is_prefilling[1] = True
    elif condition == "noncausal":
        c.causal = False
    elif condition == "media":
        c.mm_req_doc_ranges = {0: [(0, 2)]}
    elif condition == "capacity":
        sizes = (1, 2, 4)
    assert graph.decode_plan(c, sizes, verified_step=verified) is None


def test_q1_capacity_is_distinct_from_q4_rows(graph):
    sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    assert graph.decode_plan(common(1, 64), sizes, verified_step=True) == (1, 64, 64)
    assert graph.decode_plan(common(1, 65), sizes, verified_step=True) is None


@pytest.fixture
def staging(monkeypatch):
    package = ModuleType("hy4_staging_contract")
    package.__path__ = [str(PLUGIN / "nvidia")]
    kernel = ModuleType("hy4_staging_contract.runtime_input_kernels")

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU metadata contract must not launch a kernel")

    kernel.async_input_correct = kernel.async_input_gather = forbidden
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    return source_module(
        "hy4_staging_contract.runtime_staging", "nvidia/runtime_staging.py"
    )


def test_input_maps_follow_reordered_request_history(staging):
    runner = SimpleNamespace(
        prev_positions=SimpleNamespace(np=[-1, 1, 0]),
        input_batch=SimpleNamespace(req_ids=["new", "old-a", "old-b"]),
        prev_num_spec_tokens=3,
    )
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={"old-a": [123]})
    maps = staging.build_input_maps(runner, scheduled, 3, 8, [2, 6, 8])
    assert maps.sample_dst == [4, 7]
    assert maps.sample_src == [1, 0]
    assert maps.draft_dst == [5]
    assert maps.draft_src == [3]
    assert maps.num_sampled == 2
    assert maps.copy_base is True


def test_unchanged_dense_sample_map_stays_native(staging):
    runner = SimpleNamespace(
        prev_positions=SimpleNamespace(np=[0, 1]),
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
        prev_num_spec_tokens=3,
    )
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={})
    assert staging.build_input_maps(runner, scheduled, 2, 2, [1, 2]) is None


def test_host_packet_layout_and_untouched_tail(staging):
    import numpy as np

    packet = np.full(32, 99, dtype=np.int64)
    used = staging.pack_host_packet(
        packet, [4, 7, 5], [1, 0, 3], [-1, 1, 0], [0, 8, 16], [0, 0, 0, 0, 0]
    )
    assert used == 17
    assert packet[:used].tolist() == [
        4,
        7,
        5,
        1,
        0,
        3,
        -1,
        1,
        0,
        0,
        8,
        16,
        0,
        0,
        0,
        0,
        0,
    ]
    assert packet[used:].tolist() == [99] * 15


def test_packet_proof_copies_only_the_transferred_facts(staging):
    from dataclasses import dataclass

    import numpy as np

    @dataclass(frozen=True)
    class Proof:
        issuer: str
        generation: int
        request_ids: tuple
        query_starts: tuple
        computed: tuple
        prompt_lengths: tuple
        previous_positions: tuple
        previous_drafts: tuple

    packet = np.array(
        [4, 7, 5, 1, 0, 3, -1, 1, 0, 0, 8, 16, 0, 0, 0, 0, 0], dtype=np.int64
    )
    ids = ("new", "old-a", "old-b")
    args = [
        Proof,
        "owner",
        1,
        ids,
        packet,
        3,
        3,
        5,
        [2, 6, 8],
        [20, 40, 60],
        {"old-a": 1, "old-b": 0},
    ]
    result = staging._cpu_sequence_proof(*args)
    assert result.query_starts == (0, 2, 6, 8)
    assert result.computed == (0, 8, 16)
    assert result.previous_positions == (-1, 1, 0)
    assert result.previous_drafts == (-1, 0, 0)
    packet[:] = 999
    assert result.computed == (0, 8, 16)
    args[2] = 0
    assert staging._cpu_sequence_proof(*args) is None


@pytest.fixture
def patch_modules(monkeypatch):
    package = ModuleType("hy4_patch_contract")
    package.__path__ = [str(PLUGIN / "nvidia")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    proof = source_module("hy4_patch_contract.runtime_proof", "nvidia/runtime_proof.py")
    hooks = source_module("hy4_patch_contract.runtime_hooks", "nvidia/runtime_hooks.py")
    patch = source_module("hy4_patch_contract.runtime_patch", "nvidia/runtime_patch.py")
    return patch, hooks, proof


def test_patch_plan_preflights_all_owners(patch_modules):
    patch, _, _ = patch_modules

    class Owner:
        one = object()
        two = object()

    old = Owner.one
    with pytest.raises(RuntimeError, match="owner changed"):
        patch.apply_patch_plan(
            [(Owner, "one", old, object()), (Owner, "two", object(), object())]
        )
    assert Owner.one is old


def test_patch_plan_restores_partial_assignment(patch_modules):
    patch, _, _ = patch_modules

    class FailOnce(type):
        def __setattr__(cls, name, value):
            super().__setattr__(name, value)
            if name == "two" and getattr(cls, "fail", False):
                super().__setattr__("fail", False)
                raise RuntimeError("assignment failed")

    class Owner(metaclass=FailOnce):
        one = object()
        two = object()
        fail = True

    one, two = Owner.one, Owner.two
    with pytest.raises(RuntimeError, match="assignment failed"):
        patch.apply_patch_plan(
            [(Owner, "one", one, object()), (Owner, "two", two, object())]
        )
    assert Owner.one is one and Owner.two is two


def test_ordinary_call_preserves_arguments_and_return(patch_modules):
    patch, _, _ = patch_modules
    calls = []

    def original(self, *args, **kwargs):
        calls.append((args, kwargs))
        return "native"

    def implementation(*args, **kwargs):
        raise AssertionError("unselected path")

    fn = patch._scoped_method(original, implementation, None, "runner", "execute_model")
    assert fn(SimpleNamespace(), 1, named=2) == "native"
    assert calls == [((1,), {"named": 2})]


def test_selected_failure_restores_context_without_retry(patch_modules):
    patch, _, proof = patch_modules
    old_proof = object()
    new_proof = object()
    owner = SimpleNamespace(_hy4_runtime_enabled=True)
    proof.set_current_cpu_seq_lens_proof(old_proof)
    previous = patch._CURRENT_RUNNER.set("caller")
    calls = []

    def original(*args, **kwargs):
        raise AssertionError("execution cannot retry")

    def implementation(self, *args, **kwargs):
        calls.append((args, kwargs))
        assert patch.current_runner() is self
        proof.set_current_cpu_seq_lens_proof(new_proof)
        raise RuntimeError("selected execution failed")

    fn = patch._scoped_method(original, implementation, None, "runner", "execute_model")
    with pytest.raises(RuntimeError, match="selected execution failed"):
        fn(owner, 3, option=4)
    assert calls == [((3,), {"option": 4})]
    assert proof.get_current_cpu_seq_lens_proof() is old_proof
    assert patch.current_runner() == "caller"
    patch._CURRENT_RUNNER.reset(previous)
    proof.set_current_cpu_seq_lens_proof(None)


def test_constructor_delegates_existing_wrapper_once(patch_modules, monkeypatch):
    patch, hooks, _ = patch_modules
    common = object()
    stager = object()
    runtime = SimpleNamespace(
        make_common_metadata_graph=lambda: common,
        make_input_staging=lambda runner: stager,
    )
    monkeypatch.setattr(hooks, "create_hy4_runtime", lambda config: runtime)
    calls = []

    def native(self, config, device):
        calls.append("native")
        self.uses_mrope = False

    @wraps(native)
    def existing(self, *args, **kwargs):
        calls.append("existing")
        return native(self, *args, **kwargs)

    owner = SimpleNamespace()
    fn = patch._scoped_method(existing, None, SimpleNamespace(), "runner", "__init__")
    assert fn(owner, "config", "device") is None
    assert calls == ["existing", "native"]
    assert (
        owner.common_attention_metadata_graph is common
        and owner._fl_async_input_staging is stager
    )


def test_other_model_worker_never_imports_target_capabilities(
    patch_modules, monkeypatch
):
    _, hooks, _ = patch_modules

    def forbidden(config):
        raise AssertionError("non-target capability import")

    monkeypatch.setattr(hooks, "create_hy4_runtime", forbidden)
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="other"))
    )
    assert hooks.prepare_runtime_for_worker(cfg) is False


def test_explicit_v2_rejected_before_install(patch_modules, monkeypatch):
    _, hooks, _ = patch_modules
    monkeypatch.setattr(hooks, "create_hy4_runtime", lambda config: object())
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="hy_v4")),
        use_v2_model_runner=True,
    )
    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        hooks.prepare_runtime_for_worker(cfg)


def test_install_is_idempotent_and_preserves_outer_wrapper(patch_modules, monkeypatch):
    patch, _, _ = patch_modules

    class Owner:
        def read(self, value=0):
            return value

    def implementation(self, value=0):
        return value + 1

    methods = SimpleNamespace(runnerread=implementation)
    schema = {"runner.read": patch.signature_contract(Owner.read)}
    owners = {"runner": (Owner, sys.modules[__name__])}
    assert patch._install_bindings(owners, methods, schema, {"runner": []}) is True
    installed = Owner.read

    @wraps(installed)
    def outer(self, *args, **kwargs):
        return installed(self, *args, **kwargs)

    monkeypatch.setattr(Owner, "read", outer)
    assert patch._install_bindings(owners, methods, schema, {"runner": []}) is False
    assert Owner.read is outer and Owner().read(5) == 5
    selected = Owner()
    selected._hy4_runtime_enabled = True
    assert selected.read(5) == 6


def test_install_conflict_has_no_partial_changes(patch_modules):
    patch, _, _ = patch_modules

    class Owner:
        def one(self, value):
            return value

        def two(self, value):
            return value

    original = Owner.one
    methods = SimpleNamespace(
        runnerone=lambda self, value: value, runnertwo=lambda self, value: value
    )
    schema = {
        "runner.one": patch.signature_contract(Owner.one),
        "runner.two": (("bad", "POSITIONAL_OR_KEYWORD", "<required>"),),
    }
    with pytest.raises(RuntimeError, match="signature"):
        patch._install_bindings(
            {"runner": (Owner, sys.modules[__name__])}, methods, schema, {"runner": []}
        )
    assert Owner.one is original


def host_method(name):
    """Load a host adapter without importing target CUDA dependencies."""
    from contextlib import contextmanager

    source = PLUGIN / "nvidia/runtime_methods.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = [
        n
        for n in fn.decorator_list
        if isinstance(n, ast.Name) and n.id == "contextmanager"
    ]
    body = ast.parse("from __future__ import annotations").body + [fn]
    namespace = {"contextmanager": contextmanager}
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace[name]


def test_private_adapter_arguments_retain_stock_call_shapes():
    bookkeeping = inspect.signature(host_method("runner_bookkeeping_sync"))
    native_args = [object()] * 6
    bound = bookkeeping.bind(*native_args)
    bound.apply_defaults()
    assert bound.arguments["num_scheduled_tokens"] is native_args[-1]
    assert bound.arguments["prompt_logprobs_hidden_states"] is None
    prompt = inspect.signature(host_method("runner_get_prompt_logprobs_dict"))
    bound = prompt.bind(object(), object(), {})
    bound.apply_defaults()
    assert bound.arguments["retained_prompt_hidden_states"] is None
    for name, parameter in [
        ("runner_get_slot_mappings", "slot_mapping_is_current"),
        ("runner_build_attention_metadata", "block_table_rows_are_current"),
    ]:
        assert (
            inspect.signature(host_method(name)).parameters[parameter].default is False
        )


@pytest.mark.parametrize("failure", [False, True])
def test_native_input_event_lease_order_and_owner_retention(failure):
    events = []
    owner = object()
    stager = SimpleNamespace(plan=owner, in_step=False)

    def begin():
        assert events[-1] == "wait"
        stager.plan = None
        stager.in_step = True
        events.append("begin")

    def end():
        stager.in_step = False
        events.append("end")

    stager.begin_step, stager.end_step = begin, end
    event = SimpleNamespace(
        synchronize=lambda: events.append("wait"),
        record=lambda: events.append("record"),
    )
    runner = SimpleNamespace(
        _hy4_runtime_module=None,
        prepare_inputs_event=event,
        _fl_async_input_staging=stager,
    )
    method = host_method("runnersynchronize_input_prep")
    try:
        with method(runner):
            assert stager.in_step
            stager.plan = owner
            events.append("body")
            if failure:
                raise RuntimeError("body failed")
    except RuntimeError:
        assert failure
    assert events == ["wait", "begin", "body", "record", "end"]
    assert stager.plan is owner and not stager.in_step
    with method(runner):
        assert stager.plan is None


def test_foreign_existing_method_wrapper_declines_before_mutation(patch_modules):
    patch, _, _ = patch_modules

    class Owner:
        def read(self, value):
            return value

    original = Owner.read

    @wraps(original)
    def foreign(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    Owner.read = foreign
    schema = {"runner.read": patch.signature_contract(original)}
    module = SimpleNamespace(__name__="official.other.module")
    with pytest.raises(RuntimeError, match="another owner"):
        patch._install_bindings(
            {"runner": (Owner, module)},
            SimpleNamespace(runnerread=original),
            schema,
            {},
        )
    assert Owner.read is foreign
