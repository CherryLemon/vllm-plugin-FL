# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel collectives for the strict vLLM 0.28 reference runner."""

import os

import torch
import torch.distributed as dist

_flagcx_comm = None
_backend = "nccl"
_counts = {"all_reduce": 0, "all_gather": 0}


def requested_backend() -> str:
    backend = os.environ.get("VLLM_FL_TP_BACKEND", "nccl").strip().lower()
    if backend not in ("nccl", "flagcx"):
        raise ValueError("VLLM_FL_TP_BACKEND must be 'nccl' or 'flagcx'")
    return backend


def init_tp_collectives(device: torch.device) -> None:
    """Set up FlagCX after the Gloo control group has exchanged ranks."""
    global _backend, _flagcx_comm
    _backend = requested_backend()
    _flagcx_comm = None
    _counts.update(all_reduce=0, all_gather=0)
    if _backend != "flagcx" or not dist.is_initialized():
        return
    from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator

    communicator = PyFlagcxCommunicator(group=dist.group.WORLD, device=device)
    if not communicator.available or communicator.disabled:
        raise RuntimeError(
            "FlagCX TP was requested but its native communicator could not be initialized; "
            "set FLAGCX_PATH to the matching FlagCX build tree"
        )
    _flagcx_comm = communicator


def all_reduce_(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    if _backend == "flagcx":
        if _flagcx_comm is None:
            raise RuntimeError("FlagCX TP communicator is not initialized")
        if not tensor.is_contiguous():
            raise ValueError("FlagCX TP all_reduce requires a contiguous tensor")
        _flagcx_comm.all_reduce(tensor, out_tensor=tensor)
    else:
        dist.all_reduce(tensor)
    _counts["all_reduce"] += 1
    return tensor


def all_gather_last(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    world_size = dist.get_world_size()
    if _backend == "flagcx":
        if _flagcx_comm is None:
            raise RuntimeError("FlagCX TP communicator is not initialized")
        input_tensor = tensor.contiguous()
        output = torch.empty(
            (world_size, *input_tensor.shape),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        _flagcx_comm.all_gather(output, input_tensor)
        _counts["all_gather"] += 1
        return torch.cat(output.unbind(0), dim=-1)
    output = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(output, tensor)
    _counts["all_gather"] += 1
    return torch.cat(output, dim=-1)


def tp_collective_stats() -> dict:
    return {
        "backend": _backend,
        "control_group": dist.get_backend() if dist.is_initialized() else None,
        "flagcx_active": _flagcx_comm is not None and not _flagcx_comm.disabled,
        **_counts,
    }
