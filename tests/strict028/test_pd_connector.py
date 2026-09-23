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


@pytest.mark.parametrize("dp", range(4))
def test_asymmetric_handoff_covers_every_producer_copy_once(dp):
    from vllm_fl.strict028.pd_connector import source_plan

    owners = [source_plan(dp * 2 + tp, 2, 8, 8) for tp in range(2)]
    assert [source for source, _ in owners] == [dp * 2, dp * 2 + 1]
    completed = [rank for source, released in owners for rank in [source, *released]]
    assert sorted(completed) == list(range(8))
    assert all(len(released) == 3 for _, released in owners)
    assert source_plan(dp, 8, 8, 8) == (dp, [])
    with pytest.raises(ValueError):
        source_plan(dp * 2, 2, 8, 4)


def test_release_is_validated_and_completes_without_copying_state():
    import threading

    from vllm_fl.strict028.pd_connector import FullStateFlagCXWorker

    worker = object.__new__(FullStateFlagCXWorker)
    worker.storage = worker.staging = object()  # Must never be dereferenced.
    worker.spec = SimpleNamespace(
        state_layout_hash="layout", layout_version="v1", state_page_bytes=2048
    )
    worker.rank, worker.tp_size, worker.model_signature = 4, 8, "model"
    worker.timeout = 0.01
    worker._condition, worker._stop = threading.Condition(), threading.Event()
    worker._pending_send = {"transfer": ("request", 3)}
    worker._seen_transfer_ids, worker._sent_ids = set(), set()
    worker.released_pages = 0
    message = dict(
        action="release",
        layout_hash="layout",
        layout_version="v1",
        page_bytes=2048,
        rank=4,
        tp_size=8,
        model_signature="model",
        transfer_id="transfer",
    )
    with pytest.raises(ValueError):
        worker._send_one(dict(message, model_signature="other"))
    assert "transfer" in worker._pending_send
    worker._send_one(message)
    assert worker._sent_ids == {"request"}
    assert worker.released_pages == 1
    assert not worker._pending_send
    with pytest.raises(ValueError, match="duplicate"):
        worker._send_one(message)


def test_official_aggregator_waits_for_transfers_and_unused_copy_releases():
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

    aggregator = KVOutputAggregator(8)

    def outputs(finished):
        return [
            ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                kv_connector_output=KVConnectorOutput(
                    finished_sending={"request"} if rank in finished else None
                ),
            )
            for rank in range(8)
        ]

    # The two pulled copies alone cannot release the producer's scheduler page.
    result = aggregator.aggregate(outputs({4, 5}))
    assert not result.kv_connector_output.finished_sending
    result = aggregator.aggregate(outputs({0, 1, 2, 3, 6, 7}))
    assert result.kv_connector_output.finished_sending == {"request"}


def test_mixed_axis_scheduler_requires_release_capable_prefill():
    scheduler, request = scheduler_and_request()
    scheduler.data_parallel = True
    with pytest.raises(ValueError, match="protocol v2"):
        scheduler.get_num_new_matched_tokens(request, 0)
    request.kv_transfer_params["state_transfer_protocol"] = 2
    assert scheduler.get_num_new_matched_tokens(request, 0) == (4, True)


@pytest.mark.parametrize("transfer_fails", [False, True])
def test_unused_sources_release_only_after_successful_gpu_install(
    monkeypatch, transfer_fails
):
    import threading

    from vllm_fl.strict028.pd_connector import FullStateFlagCXWorker

    events, errors = [], []

    class Socket:
        def setsockopt(self, *args):
            pass

        def connect(self, *args):
            pass

        def send_json(self, message):
            self.message = message
            events.append((message["action"], message["rank"]))

        def recv_json(self):
            return {"status": "error" if transfer_fails else "done"}

        def close(self):
            pass

    worker = object.__new__(FullStateFlagCXWorker)
    worker.rank, worker.tp_size, worker.world_size = 4, 2, 8
    worker.device, worker.host, worker.rpc_port, worker.timeout = (
        "cuda:4",
        "decoder",
        1,
        1,
    )
    worker.spec = SimpleNamespace(
        state_layout_hash="layout", layout_version="v1", state_page_bytes=2048
    )
    worker.model_signature = "model"
    worker.storage = [SimpleNamespace(copy_=lambda *a, **kw: events.append("gpu-copy"))]
    worker.staging = [SimpleNamespace(data_ptr=lambda: 123)]
    worker._context = SimpleNamespace(socket=lambda *a: Socket())
    worker._condition = threading.Condition()
    worker._received_ids = set()
    worker.received_pages = worker.received_bytes = 0
    worker._fail = errors.append
    monkeypatch.setattr(
        "vllm_fl.strict028.pd_connector.torch.cuda.set_device", lambda *a: None
    )
    monkeypatch.setattr(
        "vllm_fl.strict028.pd_connector.torch.cuda.synchronize",
        lambda *a: events.append("gpu-ready"),
    )
    meta = SimpleNamespace(
        local_block_ids=[[0]],
        transfer_id="transfer",
        remote_tp_size=8,
        remote_host="producer",
        remote_port=10,
    )
    worker._receive_one("request", meta)
    if transfer_fails:
        assert errors and events == [("transfer", 4)]
        assert not worker._received_ids
    else:
        assert not errors
        assert events == [
            ("transfer", 4),
            "gpu-copy",
            "gpu-ready",
            ("release", 0),
            ("release", 2),
            ("release", 6),
        ]
        assert worker._received_ids == {"request"}
        assert worker.received_pages == 1 and worker.received_bytes == 2048
