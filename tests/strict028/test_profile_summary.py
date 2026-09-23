# SPDX-License-Identifier: Apache-2.0

import gzip
import importlib.util
import json
from pathlib import Path

import pytest


source = Path(__file__).parents[2] / "tools/strict028/summarize_decode_profiles.py"
spec = importlib.util.spec_from_file_location("profile_summary", source)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def write_profile(directory, *, missing_launch=False, active=20):
    rows, events = [], []
    for step in range(10):
        rows.append(dict(active=active, min_position=131073 + step,
                         max_position=131074 + step, target_replays=2, draft_replays=1))
        events.append(dict(ph="X", cat="user_annotation", name=f"fl_decode_step_{step}", ts=step * 100, dur=40))
        # Kineto also projects one CPU scope onto multiple GPU streams/ranges.
        # These are annotations of the same step, not additional steps.
        for _ in range(3):
            events.append(dict(ph="X", cat="gpu_user_annotation", name=f"fl_decode_step_{step}", ts=step * 100, dur=40))
        for _ in range(3):
            events.append(dict(ph="X", cat="cuda_runtime", name="cudaGraphLaunch", dur=1))
        for offset in (0, 5):
            events.append(dict(ph="X", cat="kernel", name="example", ts=step * 100 + offset, dur=10))
    if missing_launch:
        events.remove(next(e for e in events if e["name"] == "cudaGraphLaunch"))
    (directory / "rank0.receipt.json").write_text(json.dumps(
        dict(rank=0, steps=10, required_active=20, rows=rows)))
    with gzip.open(directory / "rank0.json.gz", "wt") as f:
        json.dump(dict(traceEvents=events), f)


def test_profile_summary_counts_launches_and_overlapping_kernels(tmp_path):
    write_profile(tmp_path)
    result = summary.summarize_rank(tmp_path, 0)
    assert result["graph_launches"] == 30
    assert result["kernel_sum_ms_per_step"] == pytest.approx(.020)
    assert result["gpu_busy_union_ms_per_step"] == pytest.approx(.015)
    assert result["gpu_span_ms_per_step"] == pytest.approx(.0915)


@pytest.mark.parametrize("kwargs", [dict(missing_launch=True), dict(active=19)])
def test_profile_summary_rejects_incomplete_evidence(tmp_path, kwargs):
    write_profile(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        summary.summarize_rank(tmp_path, 0)
