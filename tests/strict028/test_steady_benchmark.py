# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

source = Path(__file__).parents[2] / "tools/strict028/benchmark_pd_decode.py"
spec = importlib.util.spec_from_file_location("steady_benchmark", source)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_stream_boundaries_exclude_delayed_start_and_finished_requests():
    samples = [{"time_s": t} for t in (100, 101, 102, 103, 104, 105, 106)]
    requests = [
        {"token_arrivals": [[100, 5], [102, 4], [103, 2], [104, 3]]},
        {"token_arrivals": [[102, 5], [103, 4], [104, 2], [106, 1]]},
    ]
    result = bench.measure_steady_window(samples, requests, origin=99)
    assert (result["start_s"], result["end_s"]) == (3, 5)
    assert result["tokens_per_request"] == [5, 6]
    assert result["aggregate_tps"] == pytest.approx(5.5)
    assert result["median_request_tps"] == pytest.approx(2.75)


def test_stale_occupancy_without_common_stream_interval_is_rejected():
    requests = [
        {"token_arrivals": [[0, 1], [1, 1]]},
        {"token_arrivals": [[2, 1], [3, 1]]},
    ]
    assert bench.measure_steady_window([{"time_s": 0}, {"time_s": 3}], requests, 0) is None


def test_steady_window_excludes_preemption_and_missing_samples():
    samples = [
        dict(time_s=t, running=[20.0] * 4, waiting=[0.0] * 4, preemptions=p)
        for t, p in [(0, 0), (.5, 0), (1, 1), (1.5, 1), (2, 1), (6, 1)]
    ]
    assert [s["time_s"] for s in bench.longest_steady_window(samples)] == [1, 1.5, 2]
