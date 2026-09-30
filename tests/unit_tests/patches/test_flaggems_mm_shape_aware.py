from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_fl import flaggems_runtime as runtime
from vllm_fl.patches import flaggems_mm_shape_aware as shape_aware


class _FakeTensor:
    def __init__(
        self,
        *,
        m=1,
        ndim=2,
        dtype=torch.bfloat16,
        device_type="cuda",
        stride=(4096, 1),
    ):
        self.shape = (m, 4096) if ndim == 2 else (m, 4096, 1)
        self.ndim = ndim
        self.dtype = dtype
        self.device = SimpleNamespace(type=device_type)
        self._stride = stride

    def stride(self):
        return self._stride


class _FakeLibrary:
    def __init__(self):
        self.impl_calls = []

    def impl(
        self,
        op_name,
        fn,
        dispatch_key,
        *,
        with_keyset=False,
        allow_override=False,
    ):
        self.impl_calls.append((op_name, fn, dispatch_key, with_keyset, allow_override))


class _FakeSafeKernel:
    def __init__(self, fn):
        self.fn = fn

    def call_boxed(self, dispatch_keys, *args):
        return self.fn(dispatch_keys, *args)


def test_default_is_disabled_and_does_not_touch_torch(monkeypatch):
    monkeypatch.setattr(runtime, "_STATE", None)
    monkeypatch.setattr(runtime, "_FAILED", False)
    monkeypatch.delenv(shape_aware.ENABLE_ENV, raising=False)
    monkeypatch.setattr(
        shape_aware,
        "capture_native_mm_kernel",
        lambda: pytest.fail("captured disabled MM"),
    )
    assert runtime.configure_flaggems(lambda library: None).status == "disabled"


def test_caller_default_can_enable_but_explicit_disable_wins(monkeypatch):
    monkeypatch.delenv(shape_aware.ENABLE_ENV, raising=False)
    assert shape_aware.is_shape_aware_mm_enabled(default=True) is True

    monkeypatch.setenv(shape_aware.ENABLE_ENV, "0")
    assert shape_aware.is_shape_aware_mm_enabled(default=True) is False


@pytest.mark.parametrize(
    ("whitelist", "blacklist", "expected"),
    [
        (None, None, True),
        (None, ["linear"], True),
        (None, ["mm"], False),
        (["mm", "rms_norm"], None, True),
        (["linear"], None, False),
        ([], ["mm"], False),
    ],
)
def test_mm_dispatch_guard_preserves_explicit_flaggems_selection(
    whitelist, blacklist, expected
):
    assert shape_aware.is_mm_dispatch_enabled(whitelist, blacklist) is expected


@pytest.mark.parametrize("value", ["", "2 ", " 2", "+2", "-1", "1.0", "abc"])
def test_threshold_rejects_ambiguous_values(monkeypatch, value):
    monkeypatch.setenv(shape_aware.THRESHOLD_ENV, value)
    with pytest.raises(ValueError, match=shape_aware.THRESHOLD_ENV):
        shape_aware._parse_threshold_env()


def test_default_threshold_covers_running_64_and_128_is_configurable(monkeypatch):
    monkeypatch.delenv(shape_aware.THRESHOLD_ENV, raising=False)
    assert shape_aware._parse_threshold_env() == 64

    monkeypatch.setenv(shape_aware.THRESHOLD_ENV, "128")
    assert shape_aware._parse_threshold_env() == 128


@pytest.mark.parametrize("value", ["0", "false"])
def test_disable_values_are_strict_but_supported(monkeypatch, value):
    monkeypatch.setenv(shape_aware.ENABLE_ENV, value)
    assert shape_aware._parse_bool_env(shape_aware.ENABLE_ENV) is False


def test_enable_value_rejects_ambiguous_values(monkeypatch):
    monkeypatch.setenv(shape_aware.ENABLE_ENV, "TRUE")
    with pytest.raises(ValueError, match=shape_aware.ENABLE_ENV):
        shape_aware._parse_bool_env(shape_aware.ENABLE_ENV)


def test_native_candidate_boundary_dtype_and_stride():
    a = _FakeTensor(m=1, stride=(4096, 1))
    # A transposed/column-major weight is a normal vLLM linear layout.
    b_column_major = _FakeTensor(m=4096, stride=(1, 4096))
    assert shape_aware._is_native_candidate(a, b_column_major, 1)

    assert shape_aware._is_native_candidate(_FakeTensor(m=64), b_column_major, 64)
    assert not shape_aware._is_native_candidate(_FakeTensor(m=65), b_column_major, 64)

    assert not shape_aware._is_native_candidate(_FakeTensor(m=2), b_column_major, 1)
    assert not shape_aware._is_native_candidate(
        a, _FakeTensor(dtype=torch.float16, stride=(1, 4096)), 1
    )
    assert not shape_aware._is_native_candidate(a, _FakeTensor(stride=(8192, 2)), 1)
    assert not shape_aware._is_native_candidate(
        _FakeTensor(stride=(8192, 2)), b_column_major, 1
    )
    assert not shape_aware._is_native_candidate(
        _FakeTensor(device_type="cpu"), b_column_major, 1
    )
    assert not shape_aware._is_native_candidate(
        _FakeTensor(ndim=3), _FakeTensor(ndim=3), 1
    )


def test_apply_captures_flaggems_before_override_and_routes_shapes(monkeypatch):
    native = _FakeSafeKernel(lambda keys, a, b: "native")
    flaggems = _FakeSafeKernel(lambda keys, a, b: "flaggems")
    library = _FakeLibrary()
    observed = SimpleNamespace(
        mm=flaggems, mm_callable=lambda a, b: None, mm_registration=None
    )
    monkeypatch.setattr(shape_aware.torch.library, "Library", lambda *args: library)
    state = shape_aware.apply_shape_aware_mm(native, observed, 2)
    assert state.native_mm is native and state.flaggems_mm is flaggems
    op, wrapper, key, with_keyset, allow_override = library.impl_calls[0]
    assert (op, key, with_keyset, allow_override) == ("mm", "CUDA", True, True)
    b = _FakeTensor(m=4096, stride=(1, 4096))
    assert [wrapper("keys", _FakeTensor(m=rows), b) for rows in (1, 2, 3, 1)] == [
        "native",
        "native",
        "flaggems",
        "native",
    ]


def test_apply_fails_without_safe_override_api(monkeypatch):
    class OldLibrary:
        def impl(self, name, fn, key):
            pytest.fail("unsafe registration attempted")

    safe = _FakeSafeKernel(lambda *args: None)
    observed = SimpleNamespace(
        mm=safe, mm_callable=lambda a, b: None, mm_registration=None
    )
    monkeypatch.setattr(
        shape_aware.torch.library, "Library", lambda *args: OldLibrary()
    )
    with pytest.raises(RuntimeError, match="with_keyset|allow_override"):
        shape_aware.apply_shape_aware_mm(safe, observed, 2)


def test_apply_requires_capture_before_flaggems(monkeypatch):
    monkeypatch.setattr(runtime, "_STATE", None)
    monkeypatch.setattr(runtime, "_FAILED", False)
    monkeypatch.setenv(shape_aware.ENABLE_ENV, "1")
    monkeypatch.setattr(
        shape_aware,
        "capture_native_mm_kernel",
        lambda: (_ for _ in ()).throw(RuntimeError("capture before flag_gems.enable")),
    )
    enabled = []
    with pytest.raises(RuntimeError, match="before flag_gems.enable"):
        runtime.configure_flaggems(enabled.append)
    assert enabled == []
    assert runtime._STATE is None and runtime._FAILED
