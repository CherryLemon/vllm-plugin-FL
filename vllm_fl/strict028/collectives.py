# SPDX-License-Identifier: Apache-2.0
"""Explicit TP/DP/EP collectives owned by the empty-build FL worker."""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .parallel import ParallelLayout


@dataclass
class _Axis:
    group: object
    size: int
    communicator: object = None


_axes = {}
_layout = None
_backend = "nccl"
_counts = {"all_reduce": 0, "all_gather": 0}


def parallel_layout():
    if _layout is not None:
        return _layout
    size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    return ParallelLayout(size, 1, rank)


def requested_backend() -> str:
    backend = os.environ.get("VLLM_FL_TP_BACKEND", "nccl").strip().lower()
    if backend not in ("nccl", "flagcx"):
        raise ValueError("VLLM_FL_TP_BACKEND must be 'nccl' or 'flagcx'")
    return backend


def init_tp_collectives(device: torch.device, tensor_size=None) -> None:
    """Create every subgroup in global order before any model work starts.

    The default preserves homogeneous TP. A smaller tensor_size creates local
    attention TP, strided request DP and global expert/Engram communicators.
    Gloo exchanges FlagCX IDs; all device payloads use the requested backend.
    """
    global _backend, _layout
    _backend = requested_backend()
    _axes.clear()
    _counts.update(all_reduce=0, all_gather=0)
    size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    tensor_size = size if tensor_size is None else tensor_size
    if tensor_size < 1 or size % tensor_size:
        raise ValueError("tensor size must divide the global process group")
    _layout = ParallelLayout(tensor_size, size // tensor_size, rank)
    shared = {}
    for name in ("tp", "dp", "ep"):
        for members in _layout.groups(name):
            key = tuple(members)
            if key not in shared:
                group = None
                communicator = None
                if len(members) > 1:
                    group = (
                        dist.group.WORLD
                        if len(members) == size
                        else dist.new_group(members)
                    )
                    if rank in members and _backend == "flagcx":
                        from vllm_fl.distributed.device_communicators.flagcx import (
                            PyFlagcxCommunicator,
                        )

                        communicator = PyFlagcxCommunicator(group=group, device=device)
                        if not communicator.available or communicator.disabled:
                            raise RuntimeError(
                                f"requested FlagCX {name} communicator failed to initialize"
                            )
                shared[key] = _Axis(group, len(members), communicator)
            if rank in members:
                _axes[name] = shared[key]


def _axis(name):
    if name not in ("tp", "dp", "ep"):
        raise ValueError(f"unknown collective axis: {name}")
    if name in _axes:
        return _axes[name]
    if not dist.is_initialized():
        return _Axis(None, 1)
    if name == "dp":
        return _Axis(None, 1)
    return _Axis(dist.group.WORLD, dist.get_world_size())


def all_reduce_(tensor: torch.Tensor, *, axis="tp") -> torch.Tensor:
    group = _axis(axis)
    if group.size == 1:
        return tensor
    if _backend == "flagcx":
        if group.communicator is None:
            raise RuntimeError(f"FlagCX {axis} communicator is not initialized")
        if not tensor.is_contiguous():
            raise ValueError("FlagCX all_reduce requires a contiguous tensor")
        group.communicator.all_reduce(tensor, out_tensor=tensor)
    else:
        dist.all_reduce(tensor, group=group.group)
    _counts["all_reduce"] += 1
    return tensor


def all_gather(tensor: torch.Tensor, *, axis="tp", dim=0) -> torch.Tensor:
    group = _axis(axis)
    if group.size == 1:
        return tensor
    input_tensor = tensor.contiguous()
    output = torch.empty(
        (group.size, *input_tensor.shape),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    if _backend == "flagcx":
        if group.communicator is None:
            raise RuntimeError(f"FlagCX {axis} communicator is not initialized")
        group.communicator.all_gather(output, input_tensor)
    else:
        dist.all_gather(list(output.unbind(0)), input_tensor, group=group.group)
    _counts["all_gather"] += 1
    return torch.cat(output.unbind(0), dim=dim)


def all_gather_last(tensor: torch.Tensor) -> torch.Tensor:
    return all_gather(tensor, dim=-1)


def tp_collective_stats() -> dict:
    layout = parallel_layout()
    return {
        "backend": _backend,
        "control_group": dist.get_backend() if dist.is_initialized() else None,
        "flagcx_active": any(a.communicator is not None for a in _axes.values()),
        "tensor_size": layout.tensor_size,
        "data_size": layout.data_size,
        "expert_size": layout.world_size,
        "global_rank": layout.global_rank,
        **_counts,
    }
