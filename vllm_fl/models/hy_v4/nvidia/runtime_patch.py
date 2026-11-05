# SPDX-License-Identifier: Apache-2.0
"""Atomic, model-scoped method patches for official vLLM runtime APIs."""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from functools import wraps
from threading import RLock

_ABSENT = object()
_LOCK = RLock()
_INSTALLED = {}
_CURRENT_RUNNER = ContextVar("hy4_runtime_runner", default=None)


def current_runner():
    return _CURRENT_RUNNER.get()


def _contains(current, expected):
    seen = set()
    while callable(current) and id(current) not in seen:
        if current is expected:
            return True
        seen.add(id(current))
        current = getattr(current, "__wrapped__", None)
    return False


def apply_patch_plan(plan):
    """Validate ownership first, then publish or restore the exact old objects.

    Each item is (owner, attribute, expected_current, replacement). Missing
    attributes use _ABSENT. No tensor operation or dependency import occurs.
    """
    with _LOCK:
        for owner, name, expected, replacement in plan:
            current = inspect.getattr_static(owner, name, _ABSENT)
            if current is not expected:
                raise RuntimeError(
                    f"runtime patch owner changed: {owner.__name__}.{name}"
                )
            if replacement is _ABSENT:
                raise ValueError("a runtime replacement is required")
        changed = []
        try:
            for owner, name, expected, replacement in plan:
                changed.append((owner, name, expected))
                setattr(owner, name, replacement)
        except BaseException:
            for owner, name, expected in reversed(changed):
                if expected is _ABSENT:
                    if inspect.getattr_static(owner, name, _ABSENT) is not _ABSENT:
                        delattr(owner, name)
                else:
                    setattr(owner, name, expected)
            raise


def signature_contract(function):
    return tuple(
        (
            parameter.name,
            parameter.kind.name,
            (
                "<required>"
                if parameter.default is inspect.Parameter.empty
                else repr(parameter.default)
            ),
        )
        for parameter in inspect.signature(function).parameters.values()
    )


def _native_owner(function, module):
    """Accept native inference/tracing decorators; preserve other owners intact."""
    current = function
    seen = set()
    allowed = {module.__name__, "torch.utils._contextlib", "contextlib", "vllm.tracing"}
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        namespace = getattr(current, "__globals__", {})
        if namespace.get("__name__") not in allowed:
            return False
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is None:
            return namespace is vars(module)
        current = wrapped
    return False


def _scoped_method(original, implementation, module, kind, name):
    @wraps(original if original is not _ABSENT else implementation)
    def scoped(self, *args, **kwargs):
        from .runtime_hooks import create_hy4_runtime

        enabled = getattr(self, "_hy4_runtime_enabled", False)
        if name == "__init__":
            self._hy4_runtime_enabled = False
            config = kwargs.get("vllm_config", args[0] if args else None)
            runtime = create_hy4_runtime(config) if config is not None else None
            enabled = runtime is not None
            if kind == "proposer":
                runner = kwargs.get("runner", args[3] if len(args) > 3 else None)
                enabled = (
                    enabled and getattr(runner, "_model_runtime", None) is not None
                )
            if enabled:
                self._hy4_runtime_module = module
                self._hy4_runtime_enabled = True
                if kind == "runner":
                    self._model_runtime = runtime
                    self.common_attention_metadata_graph = None
                    self._fl_async_input_staging = None
                    self._mrope_decode_state = None
                    self._mrope_decode_active = False
                else:
                    self._model_draft_graph = None
        if not enabled:
            if original is _ABSENT:
                raise AttributeError(name)
            return original(self, *args, **kwargs)
        runner_token = None
        proof_token = None
        if kind == "runner":
            runner_token = _CURRENT_RUNNER.set(self)
            if name in ("execute_model", "sample_tokens", "_dummy_run"):
                from .runtime_proof import push_current_cpu_seq_lens_proof

                proof_token = push_current_cpu_seq_lens_proof(None)
        try:
            if name == "__init__":
                # Delegate the real constructor, retaining every installed
                # wrapper and native initialization side effect exactly once.
                result = original(self, *args, **kwargs)
                if kind == "runner":
                    runtime = self._model_runtime
                    self.common_attention_metadata_graph = (
                        runtime.make_common_metadata_graph()
                    )
                    if self.uses_mrope:
                        self._mrope_decode_state = self._make_buffer(
                            self.max_num_reqs + 1, dtype=module.torch.int64
                        )
                    self._fl_async_input_staging = runtime.make_input_staging(self)
                else:
                    self._model_draft_graph = runner._model_runtime.make_draft_graph(
                        self, runner
                    )
                return result
            return implementation(self, *args, **kwargs)
        finally:
            if proof_token is not None:
                from .runtime_proof import reset_current_cpu_seq_lens_proof

                reset_current_cpu_seq_lens_proof(proof_token)
            if runner_token is not None:
                _CURRENT_RUNNER.reset(runner_token)

    scoped._hy4_runtime_patch = True
    scoped._hy4_runtime_original = original
    return scoped


def install_runtime_patches(runner_module=None, proposer_module=None):
    """Install after a real NVIDIA Hy4 worker config has selected this runtime."""
    if runner_module is None:
        from vllm.v1.worker import gpu_model_runner as runner_module
    if proposer_module is None:
        from vllm.v1.spec_decode import llm_base_proposer as proposer_module
    from . import runtime_methods
    from .runtime_signatures import EXPECTED_PARAMETERS, REQUIRED_GLOBALS

    owners = {
        "runner": (runner_module.GPUModelRunner, runner_module),
        "proposer": (proposer_module.SpecDecodeBaseProposer, proposer_module),
    }
    return _install_bindings(
        owners, runtime_methods, EXPECTED_PARAMETERS, REQUIRED_GLOBALS
    )


def _install_bindings(owners, runtime_methods, expected_parameters, required_globals):
    with _LOCK:
        keys = tuple(
            (owners[kind][0], name)
            for kind, name in (key.split(".", 1) for key in expected_parameters)
        )
        present = [key in _INSTALLED for key in keys]
        if any(present):
            if not all(present):
                raise RuntimeError("partial Hy4 runtime installation state")
            for owner, name in keys:
                current = inspect.getattr_static(owner, name, _ABSENT)
                if not _contains(current, _INSTALLED[(owner, name)]):
                    raise RuntimeError(
                        f"runtime patch owner changed: {owner.__name__}.{name}"
                    )
            return False
        for kind, names in required_globals.items():
            module = owners[kind][1]
            missing = [name for name in names if not hasattr(module, name)]
            if missing:
                raise RuntimeError(
                    f"native runtime capabilities are missing: {missing}"
                )
        plan = []
        for key, expected_signature in expected_parameters.items():
            kind, name = key.split(".", 1)
            owner, module = owners[kind]
            original = inspect.getattr_static(owner, name, _ABSENT)
            if expected_signature is None:
                if original is not _ABSENT:
                    raise RuntimeError(
                        f"runtime helper already owned: {owner.__name__}.{name}"
                    )
            else:
                if original is _ABSENT or signature_contract(original) != tuple(
                    tuple(x) for x in expected_signature
                ):
                    raise RuntimeError(
                        f"unsupported native runtime signature: {owner.__name__}.{name}"
                    )
                if name != "__init__" and not _native_owner(original, module):
                    raise RuntimeError(
                        f"runtime method has another owner: {owner.__name__}.{name}"
                    )
            implementation = (
                getattr(runtime_methods, kind + name, None)
                if name == "__init__"
                else getattr(runtime_methods, kind + name)
            )
            replacement = _scoped_method(original, implementation, module, kind, name)
            plan.append((owner, name, original, replacement))
        apply_patch_plan(plan)
        for owner, name, _, replacement in plan:
            _INSTALLED[(owner, name)] = replacement
        return True
