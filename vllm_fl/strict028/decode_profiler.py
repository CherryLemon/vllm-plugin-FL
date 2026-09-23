# SPDX-License-Identifier: Apache-2.0
"""Per-rank, bounded Decode profiling through the public Worker extension API."""

import json
from contextlib import contextmanager
from pathlib import Path

import torch


class DecodeProfiler:
    def __init__(self, directory, rank, *, steps=10, active=20):
        if steps < 1 or active < 1:
            raise ValueError("profile steps and active request count must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank, self.steps, self.active = rank, steps, active
        self.profiler = None
        self.completed = 0
        self.rows = []
        self.done = False

    @contextmanager
    def step(self, runner, output):
        count = len(output.num_scheduled_tokens)
        if self.done or (self.profiler is None and count != self.active):
            yield
            return
        if count != self.active:
            raise RuntimeError("Decode occupancy changed inside the profiling window")
        if self.profiler is None:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
            )
            self.profiler.start()
        positions = [runner.requests[r].computed for r in output.num_scheduled_tokens]
        before = runner.graphs.stats()
        try:
            with torch.profiler.record_function(f"fl_decode_step_{self.completed}"):
                yield
        except Exception:
            self.profiler.stop()
            self.done = True
            raise
        self.completed += 1
        after = runner.graphs.stats()
        self.rows.append(
            dict(
                step=self.completed,
                active=count,
                min_position=min(positions),
                max_position=max(positions),
                target_replays=after["target_replays"] - before["target_replays"],
                draft_replays=after["draft_replays"] - before["draft_replays"],
            )
        )
        self.profiler.step()
        if self.completed == self.steps:
            self.profiler.stop()
            self.profiler.export_chrome_trace(
                str(self.directory / f"rank{self.rank}.json.gz")
            )
            (self.directory / f"rank{self.rank}.receipt.json").write_text(
                json.dumps(
                    dict(
                        rank=self.rank,
                        steps=self.completed,
                        required_active=self.active,
                        rows=self.rows,
                        scope="profiled Decode steps; throughput must come from an unprofiled run",
                    ),
                    indent=2,
                )
                + "\n"
            )
            self.done = True
