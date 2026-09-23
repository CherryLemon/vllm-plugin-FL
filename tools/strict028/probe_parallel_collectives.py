#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""torchrun --standalone --nproc-per-node=8 probe_parallel_collectives.py.

Exercise the actual mixed-axis device communicators and changing Graph inputs.
This is a correctness probe, not a throughput benchmark.
"""

import json
import os

import torch
import torch.distributed as dist

from vllm_fl.strict028.collectives import (
    all_gather,
    all_reduce_,
    init_tp_collectives,
    parallel_layout,
    tp_collective_stats,
)


def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("gloo")
    if dist.get_world_size() != 8:
        raise ValueError("this probe verifies TP2 x DP4 / EP8 on eight devices")
    init_tp_collectives(device, tensor_size=2)
    layout = parallel_layout()
    source = torch.tensor([float(layout.global_rank)], device=device)

    def forward():
        return (
            all_reduce_(source.clone()),
            all_gather(source, axis="dp"),
            all_reduce_(source.clone(), axis="ep"),
        )

    def verify(results, offset):
        tp, dp, ep = results
        expected_tp = sum(layout.members("tp")) + 2 * offset
        expected_dp = [rank + offset for rank in layout.members("dp")]
        expected_ep = 28 + 8 * offset
        assert tp.item() == expected_tp
        assert dp.tolist() == expected_dp
        assert ep.item() == expected_ep

    verify(forward(), 0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    for offset in (0, 10, -2):
        source.fill_(layout.global_rank + offset)
        graph.replay()
        verify(captured, offset)
    rows = [None] * 8
    dist.all_gather_object(rows, {"passed": True, **tp_collective_stats()})
    if layout.global_rank == 0:
        print(json.dumps({"results": rows, "replays": 3}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
