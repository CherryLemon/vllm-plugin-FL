# SPDX-License-Identifier: Apache-2.0
"""Real Linux eventfd waits, including missing and late notifications."""

import contextlib
import os
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_fl.worker import async_output as output


@pytest.fixture
def pool(monkeypatch):
    if not hasattr(os, "eventfd"):
        pytest.skip("Linux eventfd unavailable")
    instance = output._NativeEventfdCompletionPool.__new__(
        output._NativeEventfdCompletionPool
    )
    instance._event_fds = [os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)]
    instance._available = [0]
    instance._lock = threading.Lock()
    instance._supported = True
    instance._closed = threading.Event()
    instance._quarantined = set()
    instance._enqueue_op = Mock(return_value=0)
    monkeypatch.setattr(output, "_ASYNC_OUTPUT_WAIT_TIMEOUT_S", 0.04)
    monkeypatch.setattr(output, "_ASYNC_OUTPUT_POLL_INTERVAL_S", 0.005)
    yield instance
    # No actual CUDA callbacks in these CPU tests; explicit final cleanup is safe.
    for fd in instance._event_fds:
        with contextlib.suppress(OSError):
            os.close(fd)


def test_event_completion_synchronizes():
    event = Mock()
    output._wait_for_async_output_event(event, None)
    event.synchronize.assert_called_once_with()


def test_supported_platform_uses_native_without_enable_flag(monkeypatch):
    monkeypatch.setattr(output, "_native_completion_pool", None)
    platform = SimpleNamespace(is_cuda=lambda: True, is_rocm=lambda: False)
    monkeypatch.setitem(
        sys.modules, "vllm.platforms", SimpleNamespace(current_platform=platform)
    )
    pool = Mock(spec=output._NativeEventfdCompletionPool)
    factory = Mock(return_value=pool)
    monkeypatch.setattr(output, "_create_native_completion_pool", factory)
    assert output._get_native_completion_pool() is pool
    assert output._get_native_completion_pool() is pool
    factory.assert_called_once_with()


def test_notification_releases_slot_without_cuda_calls(pool):
    completion = pool.enqueue(SimpleNamespace(cuda_stream=123))
    event = Mock()
    os.eventfd_write(pool._event_fds[0], 1)
    output._wait_for_async_output_event(event, completion)
    event.query.assert_not_called()
    event.synchronize.assert_not_called()
    assert pool.acquire() == 0


def test_no_callback_times_out_and_late_write_cannot_be_reused(pool):
    completion = pool.enqueue(SimpleNamespace(cuda_stream=123))
    event = Mock(query=Mock(return_value=False))
    with pytest.raises(TimeoutError):
        output._wait_for_async_output_event(event, completion)
    event.synchronize.assert_not_called()
    assert pool._quarantined == {0}
    pool.close()
    # The descriptor remains valid for a callback that arrives after timeout.
    os.eventfd_write(pool._event_fds[0], 1)
    assert pool.acquire() is None


@pytest.mark.parametrize(
    "error", [RuntimeError("CUDA context lost"), KeyboardInterrupt()]
)
def test_query_error_or_cancellation_propagates_and_retires(pool, error):
    completion = pool.enqueue(SimpleNamespace(cuda_stream=123))
    event = Mock(query=Mock(side_effect=error))
    with pytest.raises(type(error)):
        output._wait_for_async_output_event(event, completion)
    assert pool.acquire() is None
    event.synchronize.assert_not_called()


def test_completed_copy_without_callback_still_times_out(pool):
    completion = pool.enqueue(SimpleNamespace(cuda_stream=123))
    event = Mock(query=Mock(return_value=True))
    with pytest.raises(TimeoutError):
        output._wait_for_async_output_event(event, completion)
    event.synchronize.assert_not_called()
    assert pool._quarantined == {0}


def test_close_interrupts_wait_and_retains_callback_fd(pool):
    completion = pool.enqueue(SimpleNamespace(cuda_stream=123))
    event = Mock(query=Mock(return_value=False))
    errors = []

    def wait():
        try:
            output._wait_for_async_output_event(event, completion)
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=wait)
    thread.start()
    pool.close()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert errors and "closed" in errors[0]
    os.eventfd_write(pool._event_fds[0], 1)


def test_exhaustion_uses_event_but_enqueue_failure_propagates(pool):
    stream = SimpleNamespace(cuda_stream=123)
    completion = pool.enqueue(stream)
    assert completion is not None
    assert pool.enqueue(stream) is None
    pool.release(0)
    pool._enqueue_op.return_value = 999
    with pytest.raises(RuntimeError, match="status 999"):
        pool.enqueue(stream)
    assert not pool._supported


def test_worker_shutdown_disables_pool_and_preserves_pending_callback(
    pool, monkeypatch
):
    pool.enqueue(SimpleNamespace(cuda_stream=123))
    monkeypatch.setattr(output, "_native_completion_pool", pool)
    output._shutdown_native_completion_pool()
    assert output._native_completion_pool is False
    assert pool._closed.is_set()
    os.eventfd_write(pool._event_fds[0], 1)


@pytest.mark.parametrize("cuda,rocm", [(False, False), (False, True), (True, True)])
def test_non_nvidia_platform_never_loads_extension(monkeypatch, cuda, rocm):
    monkeypatch.setattr(output, "_native_completion_pool", None)
    platform = SimpleNamespace(is_cuda=lambda: cuda, is_rocm=lambda: rocm)
    monkeypatch.setitem(
        sys.modules, "vllm.platforms", SimpleNamespace(current_platform=platform)
    )
    constructor = Mock(side_effect=AssertionError("not NVIDIA"))
    monkeypatch.setattr(output, "_NativeEventfdCompletionPool", constructor)
    assert output._get_native_completion_pool() is None
    constructor.assert_not_called()


def test_missing_extension_retains_event_fallback(monkeypatch):
    monkeypatch.setattr(output, "_native_completion_pool", None)
    platform = SimpleNamespace(is_cuda=lambda: True, is_rocm=lambda: False)
    monkeypatch.setitem(
        sys.modules, "vllm.platforms", SimpleNamespace(current_platform=platform)
    )

    factory = Mock(return_value=None)
    monkeypatch.setattr(output, "_create_native_completion_pool", factory)
    assert output._get_native_completion_pool() is None
    assert output._native_completion_pool is False
    assert output._get_native_completion_pool() is None
    factory.assert_called_once_with()


def test_partial_construction_closes_created_descriptors(monkeypatch):
    if not hasattr(os, "eventfd"):
        pytest.skip("Linux eventfd unavailable")
    fd = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
    monkeypatch.setattr(output.os, "eventfd", Mock(side_effect=[fd, OSError("limit")]))
    with pytest.raises(OSError, match="limit"):
        output._NativeEventfdCompletionPool(Mock(), capacity=2)
    with pytest.raises(OSError):
        os.fstat(fd)


def test_runtime_initialization_error_propagates(monkeypatch):
    monkeypatch.setattr(output, "_native_completion_pool", None)
    platform = SimpleNamespace(is_cuda=lambda: True, is_rocm=lambda: False)
    monkeypatch.setitem(
        sys.modules, "vllm.platforms", SimpleNamespace(current_platform=platform)
    )
    monkeypatch.setattr(
        output,
        "_create_native_completion_pool",
        Mock(side_effect=RuntimeError("broken extension")),
    )
    with pytest.raises(RuntimeError, match="broken extension"):
        output._get_native_completion_pool()
