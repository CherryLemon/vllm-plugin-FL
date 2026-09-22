#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Differential against the published TileLang kernels, not a local imitation."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from flag_gems.fused.dsv41_reference_ops import (
    fp4_quantize_reference,
    sparse_attention_with_sink,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(41)
    source = args.model / "inference/kernel.py"
    spec = importlib.util.spec_from_file_location("published_dsv41_kernels", source)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    records = []
    for group, sfmt in ((32, "e8m0"), (16, "e4m3")):
        sdtype = torch.float8_e8m0fnu if sfmt == "e8m0" else torch.float8_e4m3fn
        for rows in (1, 7, 33):
            x = torch.randn(rows, 512, dtype=torch.bfloat16, device="cuda")
            x[0] = 0
            if rows > 1:
                x[1, :16] = torch.tensor(
                    [
                        0.25,
                        -0.25,
                        0.75,
                        -0.75,
                        1.25,
                        -1.25,
                        1.75,
                        -1.75,
                        2.5,
                        -2.5,
                        3.5,
                        -3.5,
                        5.0,
                        -5.0,
                        6.0,
                        -6.0,
                    ],
                    device="cuda",
                )
            actual = fp4_quantize_reference(
                x.clone(), group, scale_format=sfmt, inplace=True
            )
            expected = reference.fp4_act_quant(x.clone(), group, True, sdtype)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            aq, ass = fp4_quantize_reference(x, group, scale_format=sfmt)
            eq, ess = reference.fp4_act_quant(x, group, False, sdtype)
            torch.testing.assert_close(aq, eq.view(torch.uint8), atol=0, rtol=0)
            torch.testing.assert_close(
                ass.view(torch.uint8), ess.view(torch.uint8), atol=0, rtol=0
            )
            records.append(
                {
                    "op": "fp4_quantize_reference",
                    "rows": rows,
                    "group": group,
                    "scale": sfmt,
                    "max_abs_error": 0,
                }
            )
    for h, d, count in ((8, 512, 33), (16, 128, 97), (64, 512, 129)):
        q = torch.randn(1, 3, h, d, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(1, 256, d, dtype=torch.bfloat16, device="cuda")
        sink = torch.randn(h, dtype=torch.float32, device="cuda") * 3
        indices = torch.randint(
            -1, 256, (1, 3, count), dtype=torch.int32, device="cuda"
        )
        indices[:, 0] = -1
        actual = sparse_attention_with_sink(q, kv, sink, indices, d**-0.5)
        expected = reference.sparse_attn(q, kv, sink, indices, d**-0.5)
        torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.02)
        records.append(
            {
                "op": "sparse_attention_with_sink",
                "heads": h,
                "dim": d,
                "indices": count,
                "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "device": torch.cuda.get_device_name(),
                "cases": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
