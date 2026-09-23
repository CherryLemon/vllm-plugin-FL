#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-host pinned-page FlagCX P2P admission; shared path carries metadata."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from plugin.interservice.flagcx_wrapper import FLAGCXLibrary


def wait_for(path: Path, seconds=90):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path.name}")
        time.sleep(0.1)
    return json.loads(path.read_text())


def save(path: Path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("receive", "send"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--meta", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    gpu = torch.arange(4096, device="cuda:0", dtype=torch.int32).to(torch.uint8)
    stage = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
    flagcx = FLAGCXLibrary(os.environ["FLAGCX_LIB_PATH"])
    engine = flagcx.flagcxP2pEngineCreate()
    done = args.meta.with_suffix(".done.json")
    try:
        flagcx.flagcxP2pRegisterHost(engine, stage.data_ptr(), stage.numel())
        if args.role == "receive":
            stage.zero_()
            port = flagcx.flagcxP2pGetRpcPort(engine)
            flagcx.flagcxP2pStartRpcServer(engine)
            save(args.meta, {"host": args.host, "port": port, "addr": stage.data_ptr()})
            wait_for(done)
            result = stage.to("cuda:0", non_blocking=True)
            torch.cuda.synchronize()
            torch.testing.assert_close(result, gpu)
            print(
                f"cross-host FlagCX P2P receive PASS: 4096 bytes, sum={result.sum().item()}"
            )
        else:
            stage.copy_(gpu)
            torch.cuda.synchronize()
            meta = wait_for(args.meta)
            conn = flagcx.flagcxP2pGetConn(engine, f"{meta['host']}:{meta['port']}")
            flagcx.flagcxP2pBatchWriteSync(
                conn, [stage.data_ptr()], [meta["addr"]], [stage.numel()]
            )
            save(done, {"status": "written"})
            print("cross-host FlagCX P2P send PASS: 4096 bytes")
    finally:
        flagcx.flagcxP2pEngineDestroy(engine)


if __name__ == "__main__":
    main()
