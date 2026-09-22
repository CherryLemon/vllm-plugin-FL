# SPDX-License-Identifier: Apache-2.0
"""Stream checkpoint experts into the documented FlagGems MXFP4 layout.

This is the expert component of the V4.1 loader. It does not register an
incomplete whole-model loader or interpret E2M1 bytes as integer weights.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

LAYOUT = "dsv41-mxfp4-rowmajor-k2-e8m0-g32-v1"


@dataclass(frozen=True)
class ExpertShard:
    expert_ids: tuple[int, ...]
    tp_rank: int = 0
    tp_size: int = 1

    def bounds(self, intermediate: int, num_experts: int) -> tuple[int, int]:
        if not self.expert_ids or len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("expert_ids must be nonempty and unique")
        if any(type(e) is not int or not 0 <= e < num_experts for e in self.expert_ids):
            raise ValueError("expert_ids are outside the checkpoint expert range")
        if self.tp_size < 1 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("Invalid tensor-parallel rank/size")
        if intermediate % (32 * self.tp_size):
            raise ValueError(
                "TP partitions must preserve complete 32-element scale groups"
            )
        width = intermediate // self.tp_size
        return self.tp_rank * width, (self.tp_rank + 1) * width


@dataclass
class PackedExperts:
    gate_up: torch.Tensor
    down: torch.Tensor
    gate_up_scale: torch.Tensor
    down_scale: torch.Tensor
    shard: ExpertShard
    logical_intermediate_size: int
    provenance: dict

    @property
    def storage_bytes(self) -> int:
        return sum(
            x.numel() * x.element_size()
            for x in (self.gate_up, self.down, self.gate_up_scale, self.down_scale)
        )


def load_expert_shard(
    checkpoint: str | Path,
    layer: int,
    shard: ExpertShard,
    *,
    device: torch.device | str,
) -> PackedExperts:
    """Read only this rank's gate/up rows and down columns, retaining E2M1.

    Padding is local to the kernel layout. For example, the released model's
    2304-wide experts partition to 288 at TP=8, then pad to the kernel's 128
    alignment (384). Padded weights are zero, with E8M0 unit scales. Source
    checkpoint tensors, quantization config, and model dimensions stay intact.
    No implicit EP remapping or collective is performed by this function.
    """
    root = Path(checkpoint).resolve()
    config_bytes = (root / "config.json").read_bytes()
    index_bytes = (root / "model.safetensors.index.json").read_bytes()
    config = json.loads(config_bytes)
    expected_quant = {
        "activation_scheme": "dynamic",
        "expert_dtype": "fp4",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [32, 32],
    }
    if (
        config.get("model_type") != "deepseek_v41"
        or config.get("quantization_config") != expected_quant
    ):
        raise ValueError("Expected the original V4.1 FP8/MXFP4 checkpoint contract")
    text = config["text_config"]
    if not 0 <= layer < text["num_hidden_layers"]:
        raise ValueError("This loader component supports backbone experts only")
    hidden, intermediate = text["hidden_size"], text["moe_intermediate_size"]
    if hidden % 128:
        raise ValueError("FlagGems MXFP4 requires hidden_size aligned to 128")
    start, end = shard.bounds(intermediate, text["n_routed_experts"])
    local_i = end - start
    padded_i = ((local_i + 127) // 128) * 128
    count = len(shard.expert_ids)
    weight_map = json.loads(index_bytes)["weight_map"]
    names = [
        f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"
        for expert in shard.expert_ids
        for projection in ("w1", "w3", "w2")
        for suffix in ("weight", "scale")
    ]
    if missing := set(names) - weight_map.keys():
        raise ValueError(f"Missing expert tensors: {sorted(missing)}")

    # Validate the complete set before allocating accelerator memory.
    paths = {}
    for name in names:
        path = (root / weight_map[name]).resolve()
        if path.parent != root or path.suffix != ".safetensors":
            raise ValueError(f"Invalid checkpoint shard path for {name}")
        paths[name] = path
    with ExitStack() as stack:
        files = {
            path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for path in set(paths.values())
        }
        for name in names:
            source = files[paths[name]].get_slice(name)
            rows, cols = (
                (hidden, intermediate) if ".w2." in name else (intermediate, hidden)
            )
            divisor, dtype = (32, "F8_E8M0") if name.endswith(".scale") else (2, "I8")
            if (
                source.get_shape() != [rows, cols // divisor]
                or source.get_dtype() != dtype
            ):
                raise ValueError(f"Unexpected packed expert shape/dtype: {name}")

        gate_up = torch.zeros(
            (count, 2 * padded_i, hidden // 2), device=device, dtype=torch.uint8
        )
        down = torch.zeros(
            (count, hidden, padded_i // 2), device=device, dtype=torch.uint8
        )
        # Construct scales as bytes: arbitrary fill_ on E8M0 is not portable.
        gate_up_scale = torch.full(
            (count, 2 * padded_i, hidden // 32), 127, device=device, dtype=torch.uint8
        )
        down_scale = torch.full(
            (count, hidden, padded_i // 32), 127, device=device, dtype=torch.uint8
        )
        for local, expert in enumerate(shard.expert_ids):
            prefix = f"layers.{layer}.ffn.experts.{expert}"
            for projection, offset in (("w1", 0), ("w3", padded_i)):
                for suffix, target in (("weight", gate_up), ("scale", gate_up_scale)):
                    name = f"{prefix}.{projection}.{suffix}"
                    value = files[paths[name]].get_slice(name)[start:end, :]
                    target[local, offset : offset + local_i].copy_(
                        value.view(torch.uint8)
                    )
            for suffix, target, divisor in (
                ("weight", down, 2),
                ("scale", down_scale, 32),
            ):
                name = f"{prefix}.w2.{suffix}"
                value = files[paths[name]].get_slice(name)[
                    :, start // divisor : end // divisor
                ]
                target[local, :, : local_i // divisor].copy_(value.view(torch.uint8))

    return PackedExperts(
        gate_up,
        down,
        gate_up_scale.view(torch.float8_e8m0fnu),
        down_scale.view(torch.float8_e8m0fnu),
        shard,
        local_i,
        {
            "layout": LAYOUT,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "layer": layer,
            "expert_ids": list(shard.expert_ids),
            "tp_rank": shard.tp_rank,
            "tp_size": shard.tp_size,
            "source_intermediate_range": [start, end],
            "padded_intermediate_size": padded_i,
            "weight_encoding": "E2M1 low-nibble first",
            "scale_encoding": "E8M0 per 32 along K",
            "payload_hashes_verified": False,
        },
    )
