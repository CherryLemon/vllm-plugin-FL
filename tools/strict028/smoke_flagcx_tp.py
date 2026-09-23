#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun to check strict028 FlagCX TP on real CUDA devices.

Example: VLLM_FL_TP_BACKEND=flagcx FLAGCX_PATH=/opt/flagcx \
  python -m torch.distributed.run --standalone --nproc_per_node=8 smoke_flagcx_tp.py
"""

import os

import torch
import torch.distributed as dist

from vllm_fl.strict028.collectives import (
    all_gather_last,
    all_reduce_,
    init_tp_collectives,
    tp_collective_stats,
)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("gloo")
    init_tp_collectives(device)
    for dtype in (torch.bfloat16, torch.float32):
        for size in (1, 128, 4096):
            tensor = torch.full((size,), rank + 1, dtype=dtype, device=device)
            all_reduce_(tensor)
            expected = torch.full_like(tensor, world * (world + 1) // 2)
            assert torch.equal(tensor, expected), (rank, dtype, size, "all_reduce")
            shard = torch.full((2, 16), rank + 1, dtype=dtype, device=device)
            result = all_gather_last(shard)
            expected = torch.cat(
                [torch.full_like(shard, peer + 1) for peer in range(world)], dim=-1
            )
            assert torch.equal(result, expected), (rank, dtype, size, "all_gather")
    stats = tp_collective_stats()
    assert stats == {
        "backend": "flagcx",
        "control_group": "gloo",
        "flagcx_active": True,
        "all_reduce": 6,
        "all_gather": 6,
    }, stats
    print(f"FLAGCX_TP_SMOKE_PASS rank={rank} world={world} stats={stats}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
