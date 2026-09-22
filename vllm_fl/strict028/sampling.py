# SPDX-License-Identifier: Apache-2.0
"""Sampling contract shared by frontend admission and the reference Runner."""


def validate_sampling(params):
    if params is None:
        raise ValueError("FL reference profile only supports text generation")
    unsupported = []
    if params.temperature != 0:
        unsupported.append("temperature != 0")
    for name in ("presence_penalty", "frequency_penalty", "min_tokens"):
        if getattr(params, name, 0):
            unsupported.append(name)
    if params.repetition_penalty != 1:
        unsupported.append("repetition_penalty")
    for name in (
        "logprobs",
        "prompt_logprobs",
        "structured_outputs",
        "logit_bias",
        "allowed_token_ids",
    ):
        value = getattr(params, name, None)
        if value is not None and value != {}:
            unsupported.append(name)
    for name in ("bad_words", "logits_processors"):
        if getattr(params, name, None):
            unsupported.append(name)
    if unsupported:
        raise ValueError(
            "FL Eager reference profile does not yet support: " + ", ".join(unsupported)
        )
