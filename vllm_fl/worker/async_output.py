# SPDX-License-Identifier: Apache-2.0
"""CUDA stream notifications for asynchronous CPU output."""

from __future__ import annotations

import math
import os
import select
import threading
import time
from typing import Any, NamedTuple

# The native callback needs neither the GIL nor CUDA calls. Keep its descriptors
# alive until notification, including cancellation and accelerator failure.
_ASYNC_OUTPUT_COMPLETION_POOL_SIZE = 64
_ASYNC_OUTPUT_WAIT_TIMEOUT_S = 30.0
_ASYNC_OUTPUT_POLL_INTERVAL_S = 0.1


class _NativeEventfdCompletionPool:
    """Process-local pool of native stream-callback completion events."""

    def __init__(
        self, enqueue_op: Any, capacity: int = _ASYNC_OUTPUT_COMPLETION_POOL_SIZE
    ):
        if capacity <= 0:
            raise ValueError("Native completion pool capacity must be positive.")
        event_fds: list[int] = []
        try:
            for _ in range(capacity):
                event_fds.append(os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK))
        except Exception:
            for event_fd in event_fds:
                os.close(event_fd)
            raise

        self._enqueue_op = enqueue_op
        self._event_fds = event_fds
        self._available = list(range(capacity - 1, -1, -1))
        self._lock = threading.Lock()
        self._supported = True
        self._closed = threading.Event()
        self._quarantined: set[int] = set()

    def acquire(self) -> int | None:
        with self._lock:
            if not self._supported or not self._available:
                return None
            return self._available.pop()

    def enqueue(self, stream: Any) -> _NativeEventfdCompletion | None:
        slot = self.acquire()
        if slot is None:
            return None
        try:
            status = int(
                self._enqueue_op(int(stream.cuda_stream), self._event_fds[slot])
            )
            if status != 0:
                raise RuntimeError(
                    f"native CUDA host callback enqueue failed with status {status}"
                )
        except BaseException:
            self.retire(slot)
            raise
        return _NativeEventfdCompletion(self, slot)

    def wait(self, slot: int, event: Any) -> None:
        """Wait for and consume the callback notification.

        Query the copy event only on the slow path to expose CUDA errors if
        the driver skips the callback. A completed copy does not prove that a
        pending callback has relinquished its descriptor.
        """
        deadline = time.monotonic() + _ASYNC_OUTPUT_WAIT_TIMEOUT_S
        poller = select.poll()
        poller.register(self._event_fds[slot], select.POLLIN)
        while True:
            if self._closed.is_set():
                raise RuntimeError("Native async-output completion pool is closed")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Native async-output completion timed out")
            ready = poller.poll(
                max(1, math.ceil(min(remaining, _ASYNC_OUTPUT_POLL_INTERVAL_S) * 1000))
            )
            if ready:
                if ready[0][1] != select.POLLIN:
                    raise OSError("Native completion descriptor is unavailable")
                if os.eventfd_read(self._event_fds[slot]) != 1:
                    raise RuntimeError("Invalid native completion eventfd payload")
                return
            # Do not replace this with synchronize(): a missed callback or a
            # stuck stream must not turn the bounded wait into another hang.
            event.query()

    def release(self, slot: int) -> None:
        with self._lock:
            if slot not in self._quarantined and not self._closed.is_set():
                self._available.append(slot)

    def retire(self, slot: int) -> None:
        with self._lock:
            self._quarantined.add(slot)
            self._supported = False
        # Keep the fd open. The C++ callback carries only its integer value;
        # closing it would let a late write target an unrelated reused fd.
        # The fixed pool bounds this retention to 64 descriptors per process.

    def close(self) -> None:
        """Cancel pending waits without invalidating late callback descriptors."""
        with self._lock:
            self._supported = False
            self._closed.set()
            # Only free slots have no outstanding callback. Retain the rest.
            for slot in self._available:
                os.close(self._event_fds[slot])
            self._available.clear()


class _NativeEventfdCompletion(NamedTuple):
    pool: _NativeEventfdCompletionPool
    slot: int


_native_completion_pool: _NativeEventfdCompletionPool | bool | None = None
_native_completion_pool_lock = threading.Lock()


def _shutdown_native_completion_pool() -> None:
    """Cancel waits before worker teardown enters accelerator synchronization.

    The pool has a single worker-process lifetime. Do not recreate it
    after shutdown: callbacks from that lifetime may still reference its fds.
    """
    global _native_completion_pool
    with _native_completion_pool_lock:
        pool = _native_completion_pool
        _native_completion_pool = False
        if isinstance(pool, _NativeEventfdCompletionPool):
            pool.close()


def _get_native_completion_pool() -> _NativeEventfdCompletionPool | None:
    global _native_completion_pool
    if _native_completion_pool is False:
        return None
    from vllm.platforms import current_platform

    if not current_platform.is_cuda() or current_platform.is_rocm():
        return None
    if _native_completion_pool is None:
        with _native_completion_pool_lock:
            if _native_completion_pool is None:
                _native_completion_pool = _create_native_completion_pool() or False
    return (
        _native_completion_pool
        if isinstance(_native_completion_pool, _NativeEventfdCompletionPool)
        else None
    )


def _create_native_completion_pool() -> _NativeEventfdCompletionPool | None:
    # Select Event completion before submitting a callback when the platform or
    # optional extension cannot provide native notification. Execution errors
    # after selection must propagate instead of silently changing mechanisms.
    if not hasattr(os, "eventfd"):
        return None
    import importlib

    import torch

    try:
        importlib.import_module("vllm_fl._C")
    except ModuleNotFoundError as exc:
        if exc.name == "vllm_fl._C":
            return None
        raise
    if not torch.ops.vllm_fl.cuda_eventfd_completion_supported():
        return None
    return _NativeEventfdCompletionPool(
        torch.ops.vllm_fl.enqueue_cuda_eventfd_completion
    )


def _enqueue_native_completion(stream: Any) -> _NativeEventfdCompletion | None:
    pool = _get_native_completion_pool()
    return pool.enqueue(stream) if pool is not None else None


def _wait_for_async_output_event(
    event: Any, completion: _NativeEventfdCompletion | None
) -> None:
    """Wait until async D2H copies are safe to consume on the CPU."""
    if completion is None:
        event.synchronize()
        return
    try:
        completion.pool.wait(completion.slot, event)
    except BaseException:
        # Includes cancellation. Never release a slot whose callback may still
        # arrive, and propagate accelerator errors rather than hiding them.
        completion.pool.retire(completion.slot)
        raise
    completion.pool.release(completion.slot)
