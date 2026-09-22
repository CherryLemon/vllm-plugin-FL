# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_fl.strict028.weights import ExpertShard, load_expert_shard


@pytest.fixture
def checkpoint(tmp_path):
    config = {
        "model_type": "deepseek_v41",
        "quantization_config": {
            "activation_scheme": "dynamic",
            "expert_dtype": "fp4",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [32, 32],
        },
        "text_config": {
            "num_hidden_layers": 1,
            "hidden_size": 128,
            "moe_intermediate_size": 576,
            "n_routed_experts": 2,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(17)
    tensors = {}
    for expert in range(2):
        for projection, rows, cols in (
            ("w1", 576, 128),
            ("w3", 576, 128),
            ("w2", 128, 576),
        ):
            name = f"layers.0.ffn.experts.{expert}.{projection}"
            tensors[name + ".weight"] = torch.randint(
                0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator
            ).view(torch.int8)
            tensors[name + ".scale"] = torch.randint(
                120, 129, (rows, cols // 32), dtype=torch.uint8, generator=generator
            ).view(torch.float8_e8m0fnu)
    save_file(tensors, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in tensors}})
    )
    return tmp_path, tensors


@pytest.mark.parametrize("rank", [0, 1])
def test_streaming_tp_slices_preserve_codes_scales_gate_order_and_zero_padding(
    checkpoint, rank
):
    root, tensors = checkpoint
    shard = load_expert_shard(root, 0, ExpertShard((1, 0), rank, 2), device="cpu")
    assert shard.logical_intermediate_size == 288
    assert shard.gate_up.shape == (2, 768, 64)
    assert shard.down.shape == (2, 128, 192)
    begin, end = rank * 288, (rank + 1) * 288
    for local, expert in enumerate((1, 0)):
        prefix = f"layers.0.ffn.experts.{expert}"
        for projection, offset in (("w1", 0), ("w3", 384)):
            for suffix, target in (
                ("weight", shard.gate_up),
                ("scale", shard.gate_up_scale),
            ):
                torch.testing.assert_close(
                    target[local, offset : offset + 288].view(torch.uint8),
                    tensors[f"{prefix}.{projection}.{suffix}"][begin:end].view(
                        torch.uint8
                    ),
                    rtol=0,
                    atol=0,
                )
            assert torch.all(shard.gate_up[local, offset + 288 : offset + 384] == 0)
            assert torch.all(
                shard.gate_up_scale[local, offset + 288 : offset + 384].view(
                    torch.uint8
                )
                == 127
            )
        for suffix, target, divisor in (
            ("weight", shard.down, 2),
            ("scale", shard.down_scale, 32),
        ):
            torch.testing.assert_close(
                target[local, :, : 288 // divisor].view(torch.uint8),
                tensors[f"{prefix}.w2.{suffix}"][
                    :, begin // divisor : end // divisor
                ].view(torch.uint8),
                rtol=0,
                atol=0,
            )
        assert torch.all(shard.down[local, :, 144:] == 0)
        assert torch.all(shard.down_scale[local, :, 9:].view(torch.uint8) == 127)


@pytest.mark.parametrize(
    "shard",
    [
        ExpertShard(()),
        ExpertShard((0, 0)),
        ExpertShard((2,)),
        ExpertShard((0,), 0, 5),
        ExpertShard((0,), 2, 2),
    ],
)
def test_invalid_rank_or_expert_selection_is_rejected(checkpoint, shard):
    with pytest.raises(ValueError):
        load_expert_shard(checkpoint[0], 0, shard, device="cpu")


def test_missing_scale_is_rejected_before_device_allocation(checkpoint):
    root, _ = checkpoint
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    del index["weight_map"]["layers.0.ffn.experts.0.w3.scale"]
    path.write_text(json.dumps(index))
    # An invalid device also demonstrates that validation runs first.
    with pytest.raises(ValueError, match="Missing expert tensors"):
        load_expert_shard(root, 0, ExpertShard((0,)), device="missing_accelerator")
