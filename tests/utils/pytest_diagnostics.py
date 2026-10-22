# SPDX-License-Identifier: Apache-2.0
"""Preserve failures and active-test stacks before a CI job times out."""

import faulthandler
import json
import threading
import time
from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("test diagnostics")
    group.addoption("--test-diagnostics-dir", type=Path)
    group.addoption("--test-stack-timeout", type=float, default=120.0)


def pytest_configure(config):
    directory = config.getoption("test_diagnostics_dir")
    if directory is not None:
        config.pluginmanager.register(_Diagnostics(config, directory))


class _Diagnostics:
    def __init__(self, config, directory):
        self.config = config
        self.worker = getattr(config, "workerinput", {}).get("workerid", "controller")
        self.timeout = config.getoption("test_stack_timeout")
        directory.mkdir(parents=True, exist_ok=True)
        self.events = (directory / f"{self.worker}.jsonl").open("a", buffering=1)
        self.stacks = (directory / f"{self.worker}-stacks.log").open("a", buffering=1)

    def _record(self, **event):
        self.events.write(json.dumps({"time": time.time(), **event}) + "\n")

    def pytest_runtest_logstart(self, nodeid, location):
        self._record(event="start", nodeid=nodeid)

    def pytest_runtest_logreport(self, report):
        event = dict(
            event="report",
            nodeid=report.nodeid,
            phase=report.when,
            outcome=report.outcome,
            worker=getattr(report, "worker_id", self.worker),
        )
        if report.failed:
            event["traceback"] = str(report.longrepr)
            reporter = self.config.pluginmanager.getplugin("terminalreporter")
            if reporter is not None and self.worker == "controller":
                reporter.write_sep("=", f"FAILED NOW: {report.nodeid} ({report.when})")
                reporter.write_line(str(report.longrepr))
                reporter.flush()
        self._record(**event)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item):
        def dump():
            self.stacks.write(f"Active test after {self.timeout}s: {item.nodeid}\n")
            faulthandler.dump_traceback(file=self.stacks, all_threads=True)

        timer = None
        if self.timeout > 0:
            timer = threading.Timer(self.timeout, dump)
            timer.daemon = True
            timer.start()
        try:
            yield
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()

    def pytest_unconfigure(self):
        self.events.close()
        self.stacks.close()
