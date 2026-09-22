# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams

from vllm_fl.strict028.platform import PlatformFL028
from vllm_fl.strict028.worker import validate_sampling


def test_greedy_defaults_and_host_stop_handling_are_accepted():
    params = SamplingParams(
        temperature=0, max_tokens=16, stop=["END"], ignore_eos=False
    )
    # vLLM normalizes bad_words=None to [], which is still an unrestricted request.
    assert params.bad_words == []
    validate_sampling(params)
    PlatformFL028.validate_request(
        {"type": "token", "prompt_token_ids": [0, 19]}, params
    )


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": 1},
        {"bad_words": ["secret"]},
        {"min_tokens": 1},
        {"repetition_penalty": 1.1},
        {"logprobs": 0},
        {"logit_bias": {3: 1}},
        {"allowed_token_ids": [3, 4]},
        {"presence_penalty": 1},
        {"prompt_logprobs": 0},
    ],
)
def test_unimplemented_sampling_is_rejected_explicitly(change):
    params = SamplingParams(**({"temperature": 0} | change))
    with pytest.raises(ValueError, match="does not yet support"):
        validate_sampling(params)
    with pytest.raises(VLLMValidationError, match="does not yet support"):
        PlatformFL028.validate_request(
            {"type": "token", "prompt_token_ids": [0, 19]}, params
        )


@pytest.mark.parametrize("kind", ["embeds", "multimodal", "enc_dec"])
def test_non_token_input_is_rejected_before_worker_dispatch(kind):
    with pytest.raises(VLLMValidationError, match="text token generation only"):
        PlatformFL028.validate_request({"type": kind}, SamplingParams(temperature=0))


def test_async_generation_preserves_client_error_and_http_status():
    from vllm.entrypoints.serve import create_error_response
    from vllm.v1.engine.async_llm import AsyncLLM

    class AdmissionOnlyClient:
        log_requests = False

        async def add_request(self, request_id, prompt, params, **kwargs):
            PlatformFL028.validate_request(prompt, params)
            raise AssertionError("unsupported request passed frontend admission")

    async def rejected_request():
        with pytest.raises(VLLMValidationError, match="temperature != 0") as caught:
            async for _ in AsyncLLM.generate(
                AdmissionOnlyClient(),
                {"type": "token", "prompt_token_ids": [0, 19]},
                SamplingParams(temperature=1),
                "invalid-sampling",
            ):
                raise AssertionError("unsupported request produced output")
        response = create_error_response(caught.value)
        assert response.error.code == 400
        assert "temperature != 0" in response.error.message

    asyncio.run(rejected_request())
