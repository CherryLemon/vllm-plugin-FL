# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.24 KV-cache binding adapter.

vLLM 0.24's ``bind_kv_cache`` fills the runner's cache list and assigns
``forward_context[layer_name].kv_cache`` directly; it does not call a per-layer
hook.  Some model side caches (for example QSA's raw key/position state) need a
hook after that assignment to build typed views over the shared storage.

This module provides the version-correct sequence:

    allocate / reshape cache
      -> shared-cache alias
      -> upstream bind_kv_cache (once)
      -> registered cache-owner bind hook (once)

Owners are identified by an explicit registry (a class opts in at import) rather
than by a fragile module-name suffix.  A newer ABI that already calls the hook
is detected and used natively, so the hook is never invoked twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, MutableMapping

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "bind_kv_cache",
    "bind_kv_cache_owners",
    "is_registered_owner",
    "register_kv_cache_owner",
    "registered_owner_types",
    "resolve_kv_bind_abi",
]

# Explicit ABI adapter table.  Keyed by the installed vLLM release: True means
# the upstream ``bind_kv_cache`` already invokes a per-layer owner hook, so the
# adapter must not call it again.  vLLM 0.24.0 assigns ``forward_context[
# layer].kv_cache`` directly and does not call a hook.
_KV_BIND_HOOK_ABI: dict[tuple[int, int], bool] = {
    (0, 24): False,
}


def _vllm_release() -> tuple[int, int] | None:
    try:
        parts = version("vllm").split("+", 1)[0].split(".")
        return int(parts[0]), int(parts[1])
    except (PackageNotFoundError, ValueError, IndexError):
        return None


def _probe_upstream_hook() -> bool:
    """Behavioral fallback for an unlisted ABI.

    Runs the real upstream helper against a synthetic layer and observes
    whether it invokes the owner hook.  Avoids guessing from version strings.
    """
    from vllm.v1.worker.utils import bind_kv_cache as upstream_bind_kv_cache

    class _Probe:
        def __init__(self) -> None:
            self.bound = False

        def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
            self.bound = True

    probe = _Probe()
    layer_name = "model.layers.0.attn"
    upstream_bind_kv_cache(
        {layer_name: torch.zeros(1)},
        {layer_name: probe},
        [],
        1,
    )
    return probe.bound


def resolve_kv_bind_abi() -> bool:
    """Return whether the installed vLLM calls the owner hook natively.

    The ABI must either be declared in :data:`_KV_BIND_HOOK_ABI` or the
    behavioral probe must succeed.  An unknown ABI whose probe fails is refused
    rather than guessed: calling (or not calling) the hook incorrectly would
    either double-bind or leave typed views unbound.
    """
    release = _vllm_release()
    if release in _KV_BIND_HOOK_ABI:
        return _KV_BIND_HOOK_ABI[release]
    try:
        result = _probe_upstream_hook()
    except Exception as exc:
        raise RuntimeError(
            f"Unverified vLLM KV-bind ABI for release {release} and the "
            f"behavioral probe failed ({exc}). Add the release to "
            f"_KV_BIND_HOOK_ABI after verifying upstream bind_kv_cache."
        ) from exc
    logger.warning(
        "Unverified vLLM KV-bind ABI for release %s; probed owner-hook "
        "behavior: calls_hook=%s. Add it to _KV_BIND_HOOK_ABI once verified.",
        release,
        result,
    )
    return result

_REGISTERED_OWNER_TYPES: set[type] = set()


def register_kv_cache_owner(cls: type) -> type:
    """Register a layer class whose instances need a post-bind hook.

    The class must expose ``bind_kv_cache(tensor)``.  Intended to be called at
    module import by the model that owns the cache, so the common runner never
    imports model-specific code.
    """
    if not callable(getattr(cls, "bind_kv_cache", None)):
        raise TypeError(
            f"{cls!r} cannot be a KV-cache owner: it has no callable "
            "bind_kv_cache method"
        )
    _REGISTERED_OWNER_TYPES.add(cls)
    return cls


def registered_owner_types() -> frozenset[type]:
    return frozenset(_REGISTERED_OWNER_TYPES)


def is_registered_owner(layer: Any) -> bool:
    return any(
        isinstance(layer, owner_type) for owner_type in _REGISTERED_OWNER_TYPES
    )


def _collect_owners(
    forward_context: MutableMapping[str, Any],
    kv_caches: MutableMapping[str, Any],
) -> list[tuple[str, Any, Any]]:
    owners = []
    for layer_name, kv_cache in kv_caches.items():
        layer = forward_context.get(layer_name)
        if layer is not None and is_registered_owner(layer):
            owners.append((layer_name, layer, kv_cache))
    return owners


def _preflight_owners(owners: list[tuple[str, Any, Any]]) -> None:
    """Validate every owner before anything is mutated.

    Raises before ``runner_kv_caches`` or any ``layer.kv_cache`` is touched, so
    an unsupported dtype/layout/shape cannot leave a half-bound cache.
    """
    for layer_name, layer, kv_cache in owners:
        validate = getattr(layer, "validate_kv_cache", None)
        if callable(validate):
            validate(kv_cache)


@dataclass
class _BindingSnapshot:
    attributes: dict[str, Any]
    containers: list[tuple[Any, Any]]


def _snapshot_state(layer: Any) -> _BindingSnapshot:
    # Bind hooks replace tensor/view references; they must not write cache
    # contents.  Preserve these references, not copies of the (potentially
    # enormous) cache storage.  Mutable containers need separate snapshots:
    # nn.Module.__setattr__ changes _buffers in place when binding a buffer.
    snapshot = _BindingSnapshot(dict(vars(layer)), [])
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (dict, list, set)):
            saved = value.copy()
            snapshot.containers.append((value, saved))
            items = saved.values() if isinstance(saved, dict) else saved
            for item in items:
                visit(item)
        elif isinstance(value, tuple):
            for item in value:
                visit(item)

    for value in snapshot.attributes.values():
        visit(value)
    return snapshot


def _restore_state(layer: Any, snapshot: _BindingSnapshot) -> None:
    for container, saved in reversed(snapshot.containers):
        if isinstance(container, list):
            container[:] = saved
        else:
            container.clear()
            container.update(saved)
    # Restore the raw attribute table to avoid running nn.Module/custom
    # setters again while unwinding the original error.
    vars(layer).clear()
    vars(layer).update(snapshot.attributes)


def bind_kv_cache_owners(
    forward_context: MutableMapping[str, Any],
    kv_caches: MutableMapping[str, Any],
) -> int:
    """Call the bind hook once for each registered owner.  Returns the count."""
    owners = _collect_owners(forward_context, kv_caches)
    _preflight_owners(owners)
    snapshots = [(layer, _snapshot_state(layer)) for _, layer, _ in owners]
    try:
        for _layer_name, layer, kv_cache in owners:
            layer.bind_kv_cache(kv_cache)
    except Exception:
        for layer, snapshot in reversed(snapshots):
            _restore_state(layer, snapshot)
        raise
    return len(owners)


def bind_kv_cache(
    kv_caches: MutableMapping[str, Any],
    forward_context: MutableMapping[str, Any],
    runner_kv_caches: list,
    num_attn_module: int = 1,
) -> None:
    """Bind the allocated KV cache to the runner and to cache owners.

    Order: preflight owners/ABI -> upstream bind (once) -> owner hook (once,
    unless the ABI already did it). On failure restore the runner list, tensor
    references and mutable binding containers. Hooks must not mutate tensor
    contents: rollback intentionally never copies the cache storage.
    """
    from vllm.v1.worker.utils import bind_kv_cache as upstream_bind_kv_cache

    owners = _collect_owners(forward_context, kv_caches)
    # Validate before any mutation at all.
    _preflight_owners(owners)
    calls_owner_hook = resolve_kv_bind_abi()

    # Upstream assigns ``kv_cache`` on *every* layer in ``forward_context``, not
    # only the registered owners, so snapshot and restore all of them.
    bound_layers: list[tuple[str, Any]] = [
        (layer_name, forward_context[layer_name])
        for layer_name in kv_caches
        if layer_name in forward_context
    ]

    runner_snapshot = list(runner_kv_caches)
    layer_snapshots = {
        layer_name: _snapshot_state(layer)
        for layer_name, layer in bound_layers
    }

    try:
        upstream_bind_kv_cache(
            kv_caches, forward_context, runner_kv_caches, num_attn_module
        )
        if calls_owner_hook:
            return
        for _layer_name, layer, kv_cache in owners:
            layer.bind_kv_cache(kv_cache)
    except Exception:
        del runner_kv_caches[:]
        runner_kv_caches.extend(runner_snapshot)
        for layer_name, layer in bound_layers:
            _restore_state(layer, layer_snapshots[layer_name])
        logger.exception("Rolled back KV-cache bind after failure")
        raise
