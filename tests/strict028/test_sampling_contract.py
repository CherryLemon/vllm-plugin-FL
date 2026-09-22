# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm.sampling_params import SamplingParams

from vllm_fl.strict028.worker import validate_sampling


def test_greedy_defaults_and_host_stop_handling_are_accepted():
    params = SamplingParams(
        temperature=0, max_tokens=16, stop=["END"], ignore_eos=False
    )
    # vLLM normalizes bad_words=None to [], which is still an unrestricted request.
    assert params.bad_words == []
    validate_sampling(params)


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
