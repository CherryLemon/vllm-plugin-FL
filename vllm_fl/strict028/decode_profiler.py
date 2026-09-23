# SPDX-License-Identifier: Apache-2.0
"""Per-rank, bounded Decode profiling through the public Worker extension API."""

import ctypes
import json
from contextlib import contextmanager
from pathlib import Path

import torch


def use_host_activity_buffers():
    """Keep CUPTI trace buffers out of the nearly full serving device.

    CUPTI's public activity attribute 8 takes a uint8_t. The CUDA 12 runtime
    library is already loaded by this NVIDIA-only worker's PyTorch build.
    Configure before creating a CUDA context in Worker.init_device; setting
    it only when profiling starts does not relocate existing device buffers.
    Fail before arming if the requested allocation mode is unavailable.
    """
    cupti = ctypes.CDLL("libcupti.so.12")
    value, size = ctypes.c_uint8(1), ctypes.c_size_t(1)
    result = cupti.cuptiActivitySetAttribute(8, ctypes.byref(size), ctypes.byref(value))
    if result:
        raise RuntimeError(f"CUPTI host activity buffer configuration failed: {result}")
    value.value = 0
    result = cupti.cuptiActivityGetAttribute(8, ctypes.byref(size), ctypes.byref(value))
    if result or value.value != 1:
        raise RuntimeError(f"CUPTI host activity buffer verification failed: {result}")


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
            use_host_activity_buffers()
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
                        cupti_activity_buffer_location="host_pinned",
                        rows=self.rows,
                        scope="profiled Decode steps; throughput must come from an unprofiled run",
                    ),
                    indent=2,
                )
                + "\n"
            )
            self.done = True
