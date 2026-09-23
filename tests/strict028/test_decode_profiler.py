# SPDX-License-Identifier: Apache-2.0
"""Check that bounded profiling exports real Graph replay activity."""

import gzip
import json
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.strict028.decode_profiler import DecodeProfiler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_graph_profile_waits_for_occupancy_and_stops(tmp_path):
    value = torch.ones(32, 32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        value @ value
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = value @ value
    counts = dict(target_replays=0, draft_replays=0)
    runner = SimpleNamespace(
        graphs=SimpleNamespace(stats=lambda: dict(counts)),
        requests={"a": SimpleNamespace(computed=131072)},
    )
    profiler = DecodeProfiler(tmp_path, 3, steps=2, active=1)
    with profiler.step(runner, SimpleNamespace(num_scheduled_tokens={})):
        pass
    assert profiler.profiler is None
    for _ in range(4):
        with profiler.step(runner, SimpleNamespace(num_scheduled_tokens={"a": 1})):
            graph.replay()
            counts["target_replays"] += 1
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.full_like(result, 32))
    receipt = json.loads((tmp_path / "rank3.receipt.json").read_text())
    assert receipt["steps"] == 2
    assert [r["target_replays"] for r in receipt["rows"]] == [1, 1]
    with gzip.open(tmp_path / "rank3.json.gz", "rt") as f:
        trace = json.load(f)
    launches = [
        e
        for e in trace["traceEvents"]
        if e.get("name", "").startswith("cudaGraphLaunch")
    ]
    assert len(launches) == 2
