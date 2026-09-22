# SPDX-License-Identifier: Apache-2.0
"""Read-only V4.1 packed-weight contract validation, without loading tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_header(path: Path) -> tuple[dict, str, int]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors prefix: {path.name}")
        size = struct.unpack("<Q", prefix)[0]
        if not 1 <= size <= min(path.stat().st_size - 8, 100_000_000):
            raise ValueError(f"Invalid safetensors header size: {path.name}")
        raw = stream.read(size)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header, _sha256(prefix + raw), size + 8


def _tensor_kind(name: str, tensor: dict, tensors: dict) -> str:
    dtype, shape = tensor["dtype"], tensor["shape"]
    if dtype in {"BF16", "F16", "F32", "I32", "I64", "U8"}:
        return dtype
    if dtype == "F8_E8M0":
        weight_name = name.removesuffix(".scale") + ".weight"
        if not name.endswith(".scale") or weight_name not in tensors:
            raise ValueError(f"Orphan E8M0 scale: {name}")
        if tensors[weight_name]["dtype"] not in {"I8", "F8_E4M3"}:
            raise ValueError(f"Unexpected scaled weight dtype: {weight_name}")
        return "ue8m0_scale"
    if dtype not in {"I8", "F8_E4M3"} or len(shape) != 2:
        raise ValueError(f"Unrecognized packed tensor: {name}: {dtype} {shape}")
    scale_name = name.removesuffix(".weight") + ".scale"
    scale = tensors.get(scale_name)
    if not name.endswith(".weight") or scale is None:
        raise ValueError(f"Missing scale for {name}")
    if scale["dtype"] != "F8_E8M0":
        raise ValueError(f"Expected E8M0 scale for {name}")
    rows, cols = shape
    if dtype == "I8":
        if ".ffn.experts." not in name:
            raise ValueError(f"I8 outside the MXFP4 expert contract: {name}")
        kind, expected = "mxfp4_e2m1_packed", [rows, math.ceil(cols * 2 / 32)]
    elif ".engram.embed.weight" in name:
        kind, expected = "engram_fp8_row32", [rows, math.ceil(cols / 32)]
    else:
        kind, expected = "fp8_block32x32", [math.ceil(rows / 32), math.ceil(cols / 32)]
    if scale["shape"] != expected:
        raise ValueError(
            f"Scale shape mismatch for {name}: {scale['shape']} != {expected}"
        )
    return kind


def inspect_checkpoint(model_path: str | Path, *, revision: str | None = None) -> dict:
    root = Path(model_path).resolve()
    config_bytes = (root / "config.json").read_bytes()
    config = json.loads(config_bytes)
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("Expected the original deepseek_v41 checkpoint config")
    quant = config.get("quantization_config", {})
    expected_quant = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    }
    if quant != expected_quant:
        raise ValueError(f"Unvalidated quantization contract: {quant}")
    text = config["text_config"]
    index_bytes = (root / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_bytes)
    weight_map = index["weight_map"]
    tensors, shards = {}, []
    dtype_bytes = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "I32": 4,
        "I64": 8,
        "I8": 1,
        "U8": 1,
        "F8_E4M3": 1,
        "F8_E8M0": 1,
    }
    for name in sorted(set(weight_map.values())):
        path = (root / name).resolve()
        if path.parent != root or not name.endswith(".safetensors"):
            raise ValueError(f"Shard must be directly inside checkpoint: {name}")
        header, header_hash, offset = _read_header(path)
        previous = 0
        for key, tensor in sorted(
            header.items(), key=lambda pair: pair[1]["data_offsets"]
        ):
            if key in tensors or weight_map.get(key) != name:
                raise ValueError(f"Duplicate or incorrect shard mapping: {key}")
            shape, dtype = tensor["shape"], tensor["dtype"]
            if dtype not in dtype_bytes or any(
                type(d) is not int or d < 0 for d in shape
            ):
                raise ValueError(f"Unsupported tensor descriptor: {key}")
            start, end = tensor["data_offsets"]
            if (
                start != previous
                or end - start != math.prod(shape) * dtype_bytes[dtype]
            ):
                raise ValueError(f"Invalid tensor data span: {key}")
            previous = end
            tensors[key] = tensor
        if offset + previous != path.stat().st_size:
            raise ValueError(f"Truncated or trailing shard data: {name}")
        shards.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "header_sha256": header_hash,
                "tensor_count": len(header),
            }
        )
    if set(tensors) != set(weight_map):
        raise ValueError(
            f"Missing {len(set(weight_map) - set(tensors))} indexed tensors"
        )

    storage, counts, expanded = Counter(), Counter(), Counter()
    for name, tensor in tensors.items():
        kind = _tensor_kind(name, tensor, tensors)
        size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
        storage[kind] += size
        counts[kind] += 1
        expanded[kind] += (
            4 * size
            if kind == "mxfp4_e2m1_packed"
            else 2 * size
            if kind in {"fp8_block32x32", "engram_fp8_row32"}
            else 0
            if kind == "ue8m0_scale"
            else size
        )
    if sum(storage.values()) != index["metadata"]["total_size"]:
        raise ValueError("Index total_size disagrees with tensor payload sizes")
    for layer, rows in zip(
        text["engram_layer_ids"], text["engram_num_embeddings"], strict=True
    ):
        key = f"layers.{layer}.engram.embed.weight"
        if key not in tensors or tensors[key]["shape"] != [
            rows,
            text["engram_head_dim"],
        ]:
            raise ValueError(f"Required Engram table missing or malformed: {key}")

    return {
        "schema_version": 1,
        "checkpoint_revision_claim": revision,
        "config_sha256": _sha256(config_bytes),
        "index_sha256": _sha256(index_bytes),
        "payload_hashes_verified": False,
        "identity_scope": "config, index, shard headers and sizes; not full weight payload hashes",
        "model_type": config["model_type"],
        "architectures": config["architectures"],
        "quantization_config": quant,
        "required_semantics": {
            key: text[key]
            for key in (
                "num_hidden_layers",
                "hidden_size",
                "n_routed_experts",
                "num_experts_per_tok",
                "scoring_func",
                "norm_topk_prob",
                "routed_scaling_factor",
                "swiglu_limit",
                "compress_ratios",
                "kv_source_layer_ids",
                "index_source_layer_ids",
                "candidate_source_layer_id",
                "candidate_topk_blocks",
                "candidate_block_size",
                "index_topk",
                "engram_layer_ids",
                "engram_num_embeddings",
                "num_nextn_predict_layers",
                "dspark_block_size",
            )
        },
        "tensor_count": len(tensors),
        "shards": shards,
        "storage_bytes": dict(storage),
        "tensor_counts": dict(counts),
        "reference16_tensor_bytes": dict(expanded),
        "reference16_total_tensor_bytes": sum(expanded.values()),
        "capacity_note": "Unsharded tensor payload only; excludes KV, workspace, graph, communication and conversion peaks.",
        "status": "headers_validated",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_checkpoint(args.model, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Validated {report['tensor_count']} tensors in {len(report['shards'])} shards"
    )


if __name__ == "__main__":
    main()
