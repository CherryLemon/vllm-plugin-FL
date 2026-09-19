# SPDX-License-Identifier: Apache-2.0
"""Loading checks for the unquantized GLM5-Next v0.24 contract."""

from contextlib import contextmanager

import torch


def unquantized_weights(weights):
    """Also reject undeclared FP8 storage before an ordinary loader casts it."""
    for name, weight in weights:
        if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(
                f"GLM5-Next unquantized checkpoint required: {name} has "
                f"dtype {weight.dtype}; quantized projection loading is unsupported"
            )
        yield name, weight


def _required_shards(model, name):
    owner_name, attr = name.rsplit(".", 1)
    owner = model.get_submodule(owner_name)
    if attr in ("w13_weight", "w2_weight"):
        expert_map = owner.expert_map
        if expert_map is None:
            expert_ids = range(model.config.n_routed_experts)
        else:
            # Loading-time check only; never read device state in forward.
            expert_ids = [
                i for i, local_id in enumerate(expert_map.tolist()) if local_id >= 0
            ]
        shards = ("w1", "w3") if attr == "w13_weight" else ("w2",)
        return {(shard, expert) for expert in expert_ids for shard in shards}
    if owner_name.rsplit(".", 1)[-1] in (
        "gate_up_proj",
        "fused_qkv_a_proj",
        "wk_weights_proj",
    ):
        return {(0, None), (1, None)}
    return None


@contextmanager
def audit_packed_weights(model):
    """Track successful loader calls by destination, shard and local expert.

    The upstream set of loaded parameter names cannot detect a missing half
    of a packed parameter. Wrap only this model's parameter loaders during
    loading, and restore them even when loading or the audit raises.
    """
    originals = []
    missing = {}

    def wrap(name, original, required):
        seen = set()
        missing[name] = (required, seen)
        is_expert = any(expert is not None for _, expert in required)

        def load(param, weight, *args, **kwargs):
            if is_expert:
                slots = {(kwargs["shard_id"], kwargs["expert_id"])}
            else:
                shard = kwargs.get("shard_id", args[0] if args else None)
                slots = required if shard is None else {(shard, None)}
            if slots & seen:
                raise ValueError(f"Duplicate GLM5-Next packed weight: {name}, {slots}")
            result = original(param, weight, *args, **kwargs)
            # Expert loaders return False for experts not owned by this rank.
            if result is not False:
                seen.update(slots)
            return result

        return load

    try:
        for name, param in model.named_parameters():
            required = _required_shards(model, name)
            if not required:
                continue
            original = param.weight_loader
            originals.append((param, original))
            param.weight_loader = wrap(name, original, required)
        yield
        absent = [
            f"{name}: shard={shard}, expert={expert}"
            for name, (required, seen) in missing.items()
            for shard, expert in sorted(required - seen)
        ]
        if absent:
            raise RuntimeError(
                "GLM5-Next checkpoint is missing packed weight shards: "
                + "; ".join(absent[:32])
            )
    finally:
        for param, original in originals:
            param.weight_loader = original
