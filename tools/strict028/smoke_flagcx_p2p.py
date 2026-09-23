#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the FlagCX P2P engine's write path across two local CUDA processes."""

import multiprocessing as mp
import os
import traceback


def receiver(pipe):
    import torch
    from plugin.interservice.flagcx_wrapper import FLAGCXLibrary

    from vllm.utils.network_utils import get_ip

    torch.cuda.set_device(1)
    target = torch.zeros(1024, dtype=torch.uint8, pin_memory=True)
    flagcx = FLAGCXLibrary(os.environ["FLAGCX_LIB_PATH"])
    engine = flagcx.flagcxP2pEngineCreate()
    try:
        flagcx.flagcxP2pRegisterHost(engine, target.data_ptr(), target.numel())
        port = flagcx.flagcxP2pGetRpcPort(engine)
        flagcx.flagcxP2pStartRpcServer(engine)
        pipe.send(("ready", get_ip(), port, target.data_ptr()))
        if pipe.recv() != "written":
            raise RuntimeError("sender did not complete")
        result = target.to("cuda:1", non_blocking=True)
        torch.cuda.synchronize()
        expected = torch.arange(1024, device="cuda:1", dtype=torch.int32).to(
            torch.uint8
        )
        torch.testing.assert_close(result, expected)
        pipe.send(("passed", int(result.sum().item())))
    finally:
        flagcx.flagcxP2pEngineDestroy(engine)


def sender(pipe):
    import torch
    from plugin.interservice.flagcx_wrapper import FLAGCXLibrary

    torch.cuda.set_device(0)
    source_gpu = torch.arange(1024, device="cuda:0", dtype=torch.int32).to(torch.uint8)
    source = torch.empty(1024, dtype=torch.uint8, pin_memory=True)
    source.copy_(source_gpu)
    torch.cuda.synchronize()
    flagcx = FLAGCXLibrary(os.environ["FLAGCX_LIB_PATH"])
    engine = flagcx.flagcxP2pEngineCreate()
    try:
        flagcx.flagcxP2pRegisterHost(engine, source.data_ptr(), source.numel())
        status, host, port, remote_addr = pipe.recv()
        if status != "ready":
            raise RuntimeError(f"receiver status: {status}")
        conn = flagcx.flagcxP2pGetConn(engine, f"{host}:{port}")
        flagcx.flagcxP2pBatchWriteSync(
            conn, [source.data_ptr()], [remote_addr], [source.numel()]
        )
        pipe.send("written")
        status, checksum = pipe.recv()
        if status != "passed":
            raise RuntimeError(f"receiver status: {status}")
        print(
            f"FlagCX P2P two-process pinned-host write PASS: 1024 bytes, checksum={checksum}"
        )
    finally:
        flagcx.flagcxP2pEngineDestroy(engine)


if __name__ == "__main__":
    context = mp.get_context("spawn")
    left, right = context.Pipe()
    recv_proc = context.Process(target=receiver, args=(left,))
    recv_proc.start()
    try:
        sender(right)
    except Exception:
        traceback.print_exc()
        recv_proc.terminate()
        raise
    finally:
        recv_proc.join(timeout=15)
    if recv_proc.exitcode != 0:
        raise RuntimeError(f"receiver exit code {recv_proc.exitcode}")
