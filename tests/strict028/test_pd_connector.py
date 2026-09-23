# SPDX-License-Identifier: Apache-2.0
"""PD admission must skip exactly the state transferred by Prefill."""

from types import SimpleNamespace

import pytest

from vllm_fl.strict028.pd_connector import DeepseekV41Scheduler, _one_block


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
