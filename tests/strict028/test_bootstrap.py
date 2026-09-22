# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys

import pytest


def run_isolated(code, strict="1"):
    env = dict(os.environ, VLLM_FL_STRICT028=strict, VLLM_PLUGINS="fl")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_root_import_does_not_import_torch_fl_kernels_or_host():
    run_isolated("""
import sys
import vllm_fl
assert not any(name == 'torch' or name == 'vllm' or name.startswith('flag_gems')
               or name.startswith('vllm_fl.patches.') for name in sys.modules)
""")


def test_registration_is_idempotent_and_does_not_initialize_cuda():
    run_isolated("""
import importlib.util
import torch
from vllm_fl.strict028.bootstrap import register_models, register_platform
assert not torch.cuda.is_initialized()
assert register_platform() == 'vllm_fl.strict028.platform.PlatformFL028'
register_models()
register_models()
from vllm.tokenizers.registry import TokenizerRegistry
assert TokenizerRegistry.load_tokenizer_cls('fl_deepseek_v41').__module__ == 'vllm_fl.strict028.tokenizer'
assert importlib.util.find_spec('vllm._C_stable_libtorch') is None
assert importlib.util.find_spec('vllm._C') is None
assert not torch.cuda.is_initialized()
""")


@pytest.mark.parametrize(
    "version", ["0.28.0", "0.28.0+cu129", "0.29.0+empty", "0.28.0rc1+empty"]
)
def test_rejects_nonempty_or_wrong_host_version(monkeypatch, version):
    from vllm_fl.strict028 import bootstrap

    monkeypatch.setattr(bootstrap, "version", lambda _: version)
    with pytest.raises(RuntimeError, match="requires"):
        bootstrap.validate_host()


def test_ambiguous_strict_mode_fails_instead_of_using_legacy_bootstrap():
    run_isolated(
        """
try:
    import vllm_fl
except ValueError as exc:
    assert 'VLLM_FL_STRICT028' in str(exc)
else:
    raise AssertionError('invalid mode accepted')
""",
        strict="true",
    )
