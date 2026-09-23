# SPDX-License-Identifier: Apache-2.0
"""PD admission must skip exactly the state transferred by Prefill."""

from types import SimpleNamespace

import pytest

from vllm_fl.strict028.pd_connector import DeepseekV41Scheduler, _one_block
from vllm_fl.strict028.worker import WorkerFL028


def scheduler_and_request(extra=None):
    scheduler = object.__new__(DeepseekV41Scheduler)
    scheduler.kv_role = "kv_consumer"
    scheduler.state_spec = SimpleNamespace(
        state_layout_hash="layout", state_page_bytes=2048, layout_version="v1"
    )
    scheduler.model_signature = "model"
    params = {
        "do_remote_prefill": True,
        "num_external_tokens": 4,
        "state_layout_hash": "layout",
        "state_page_bytes": 2048,
        "state_layout_version": "v1",
        "model_signature": "model",
    }
    params.update(extra or {})
    return scheduler, SimpleNamespace(kv_transfer_params=params, num_prompt_tokens=5)


def test_decode_skips_prefill_state_but_consumes_first_output_token():
    scheduler, request = scheduler_and_request()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (4, True)
    assert scheduler.get_num_new_matched_tokens(request, 4) == (0, False)


@pytest.mark.parametrize(
    "override",
    [
        {"num_external_tokens": 5},
        {"state_layout_hash": "wrong"},
        {"state_page_bytes": 2049},
        {"model_signature": "other"},
    ],
)
def test_decode_rejects_invalid_handoff(override):
    scheduler, request = scheduler_and_request(override)
    with pytest.raises(ValueError):
        scheduler.get_num_new_matched_tokens(request, 0)


def test_pd_only_accepts_one_complete_state_page():
    assert _one_block([[7]]) == 7
    with pytest.raises(ValueError):
        _one_block([[7, 8]])


def test_fl_worker_initializes_connector_without_vllm_tp_group(monkeypatch):
    events = []
    storage = object()

    def connector(config, role, cache_config):
        events.append(("create", config, role, cache_config))
        return SimpleNamespace(
            register_kv_caches=lambda caches: events.append(("register", caches))
        )

    monkeypatch.setattr("vllm_fl.strict028.worker.DeepseekV41FLConnector", connector)
    monkeypatch.setattr("vllm_fl.strict028.worker.torch.cuda.empty_cache", lambda: None)
    state = SimpleNamespace(
        allocate=lambda config, device: events.append(("allocate", config, device)),
        storage=storage,
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(state=state, graph_enabled=False),
        vllm_config=SimpleNamespace(kv_transfer_config=object()),
        device="cuda:0",
        pd_connector=None,
    )
    WorkerFL028.initialize_from_config(worker, "cache")

    assert [event[0] for event in events] == ["allocate", "create", "register"]
    assert events[-1][1] == {"fl_request_state": storage}
    assert worker.pd_connector is not None
