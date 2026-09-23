# SPDX-License-Identifier: Apache-2.0
"""Mixed sharding must retain EP ownership while changing attention TP."""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.strict028.models.deepseek_v41.loader import _load_original_checkpoint
from vllm_fl.strict028.parallel import ParallelLayout


@pytest.mark.parametrize("rank", range(8))
def test_target_axes_and_checkpoint_slices(tmp_path, monkeypatch, rank):
    layout = ParallelLayout(2, 4, rank)
    assert layout.members("tp") == [rank // 2 * 2, rank // 2 * 2 + 1]
    assert layout.members("dp") == list(range(rank % 2, 8, 2))
    assert layout.members("ep") == list(range(8))
    tensors = {
        "embed.weight": torch.arange(256, dtype=torch.float32).reshape(8, 32),
        "layers.0.engram.embed.weight": torch.arange(59 * 32)
        .remainder(64)
        .reshape(59, 32)
        .to(torch.float8_e4m3fn),
        "layers.0.engram.embed.scale": torch.arange(59, dtype=torch.uint8)
        .reshape(59, 1)
        .add(100)
        .view(torch.float8_e8m0fnu),
    }
    tensors.update(
        {
            f"layers.0.ffn.experts.{i}.w1.weight": torch.full((4, 32), float(i))
            for i in range(8)
        }
    )
    params = {
        "embed.weight": torch.empty(4, 32),
        "layers.0.engram.embed.weight": torch.empty(8, 32, dtype=torch.float8_e4m3fn),
        "layers.0.engram.embed.scale": torch.empty(8, 1, dtype=torch.float8_e8m0fnu),
        f"layers.0.ffn.experts.{rank}.w1.weight": torch.empty(4, 32),
    }
    model = SimpleNamespace(
        named_parameters=lambda: params.items(),
        get_submodule=lambda _: SimpleNamespace(n_routed_experts=8),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in tensors}})
    )

    class View:
        def __init__(self, value):
            self.value = value

        def get_shape(self):
            return self.value.shape

        def __getitem__(self, bounds):
            return self.value[bounds]

    opened = lambda _: SimpleNamespace(get_slice=lambda name: View(tensors[name]))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    report = _load_original_checkpoint(
        model, tmp_path, layout.tensor_rank, 2, opened, expert_rank=rank, expert_size=8
    )
    torch.testing.assert_close(
        params["embed.weight"],
        tensors["embed.weight"][layout.tensor_rank * 4 : (layout.tensor_rank + 1) * 4],
    )
    for suffix, padding in (("weight", 0), ("scale", 127)):
        key = "layers.0.engram.embed." + suffix
        source = tensors[key][rank * 8 : (rank + 1) * 8].view(torch.uint8)
        actual = params[key].view(torch.uint8)
        torch.testing.assert_close(actual[: len(source)], source, rtol=0, atol=0)
        assert torch.all(actual[len(source) :] == padding)
    assert torch.all(params[f"layers.0.ffn.experts.{rank}.w1.weight"] == rank)
    assert report["skipped_tensor_counts"] == {"owned by another expert rank": 7}
    assert report["world_size"] == 2 and report["expert_size"] == 8


@pytest.mark.parametrize("tp,dp,rank", [(0, 4, 0), (2, 0, 0), (2, 4, 8), (2, 4, -1)])
def test_invalid_layout_is_rejected(tp, dp, rank):
    with pytest.raises(ValueError):
        ParallelLayout(tp, dp, rank)


@pytest.mark.parametrize("global_rank", range(8))
def test_worker_preserves_executor_rank_for_local_message_queue(
    monkeypatch, global_rank
):
    from types import SimpleNamespace

    from vllm_fl.strict028.worker import WorkerFL028

    calls = []
    tp_rank, dp_rank = global_rank % 2, global_rank // 2
    worker = SimpleNamespace(
        rank=tp_rank,
        local_rank=tp_rank,
        model_config=SimpleNamespace(seed=0),
        distributed_init_method="tcp://127.0.0.1:1",
        parallel_config=SimpleNamespace(
            data_parallel_size=4,
            data_parallel_rank=dp_rank,
            data_parallel_rank_local=dp_rank,
            tensor_parallel_size=2,
            world_size=2,
            world_size_across_dp=8,
            data_parallel_master_ip="127.0.0.1",
            get_next_dp_init_port=lambda: 29570,
        ),
    )
    monkeypatch.setattr("vllm_fl.strict028.worker.requested_backend", lambda: "flagcx")
    monkeypatch.setattr(
        "vllm_fl.strict028.worker.torch.cuda.set_device", lambda *a: None
    )
    monkeypatch.setattr(
        "vllm_fl.strict028.worker.dist.init_process_group",
        lambda backend, **kw: calls.append(kw),
    )
    monkeypatch.setattr(
        "vllm_fl.strict028.worker.init_tp_collectives", lambda *a, **kw: None
    )
    WorkerFL028.init_device(worker)
    assert worker.rank == tp_rank  # vLLM's TP-local MessageQueue.create_from_handle.
    assert worker.global_rank == global_rank
    assert worker.device.index == global_rank
    assert calls[0]["rank"] == global_rank and calls[0]["world_size"] == 8
