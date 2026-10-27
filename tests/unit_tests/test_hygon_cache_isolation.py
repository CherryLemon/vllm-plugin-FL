# SPDX-License-Identifier: Apache-2.0
"""Exercise the real runner and Python startup across independent workers."""

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from tests.run import TestCase as Case, TestRunner as Runner


def test_hygon_e2e_workers_use_distinct_autotuning_databases(monkeypatch, tmp_path):
    original_cache = str(tmp_path / "persistent")
    monkeypatch.setenv("FLAGGEMS_CACHE_DIR", original_cache)
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "sitecustomize.py").write_text(
        "import os\n"
        "marker = str(os.getpid())\n"
        "from sitecustomize import marker as current\n"
        "assert current == marker\n"
        "os.environ['FL_CACHE_TEST_STARTUP_PID'] = marker\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(startup))
    runner = Runner("hygon", device="bw1000", scope="e2e")
    probe = textwrap.dedent(
        """
        import json
        import os
        import sqlite3
        import subprocess
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        assert os.environ['FL_CACHE_TEST_STARTUP_PID'] == str(os.getpid())
        cache = Path(os.environ['FLAGGEMS_CACHE_DIR'])
        cache.mkdir(parents=True)
        with sqlite3.connect(cache / 'tuning.db') as db:
            db.execute('CREATE TABLE benchmark (pid INTEGER)')
            db.execute('INSERT INTO benchmark VALUES (?)', (os.getpid(),))
        child = (
            "import os; "
            "assert os.environ['FL_CACHE_TEST_STARTUP_PID'] == str(os.getpid()); "
            "print(os.environ['FLAGGEMS_CACHE_DIR'])"
        )
        with ThreadPoolExecutor(max_workers=2) as workers:
            paths = list(workers.map(
                lambda _: subprocess.check_output(
                    [sys.executable, '-c', child], text=True
                ).strip(), range(2)
            ))
        print(json.dumps([str(cache), *paths]))
        """
    )
    monkeypatch.setattr(
        runner, "_build_pytest_cmd", lambda _: [sys.executable, "-c", probe]
    )
    result = runner._run_single(
        Case("cache-probe", "unused", task="inference"), _quiet=True
    )
    assert result.passed, result.stdout
    paths = json.loads(result.stdout)
    assert len(set(paths)) == 3
    parent = Path(paths[0])
    assert all(Path(path).parent == parent for path in paths[1:])
    assert not parent.parent.exists(), "case-local tuning caches must be cleaned up"
    assert os.environ["FLAGGEMS_CACHE_DIR"] == original_cache


@pytest.mark.parametrize(
    "platform,device,task",
    [("hygon", "bw1000", "unit"), ("nvidia", "a100", "inference")],
)
def test_other_scopes_preserve_cache_environment(
    monkeypatch, tmp_path, platform, device, task
):
    cache = str(tmp_path / "persistent")
    monkeypatch.setenv("FLAGGEMS_CACHE_DIR", cache)
    runner = Runner(platform, device=device, scope=task)
    probe = "import os; print(os.environ['FLAGGEMS_CACHE_DIR'])"
    monkeypatch.setattr(
        runner, "_build_pytest_cmd", lambda _: [sys.executable, "-c", probe]
    )
    result = runner._run_single(Case("cache-probe", "unused", task=task), _quiet=True)
    assert result.passed, result.stdout
    assert result.stdout.strip() == cache
