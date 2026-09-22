# SPDX-License-Identifier: Apache-2.0
"""Stream original safetensors into the rank-local graph; never rewrite source.

Sharding follows the published inference/convert.py. The packed E2M1 expert
payload remains packed and E8M0 scales remain unpermuted. Only wo_a is expanded
to BF16; output/HC/compressor FP32 parameters are explicit reference semantics.
"""

import hashlib
import json
import logging
import math
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)


def canonical_name(source):
    name = source.removeprefix("model.").replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    return name.replace("weight_scale_inv", "scale").replace(
        "e_score_correction_bias", "bias"
    )


def shard_axis(name):
    if name.endswith(".attn_sink"):
        return 0
    if "." not in name:
        return None
    key = name.split(".")[-2]
    return {
        "embed": 0,
        "wq_b": 0,
        "wo_a": 0,
        "wo_b": 1,
        "head": 0,
        "weights_proj": 0,
    }.get(key)


def load_original_checkpoint(model, root, rank, world_size):
    with ExitStack() as stack:
        handles = {}

        def opened(path):
            path = Path(path)
            if path not in handles:
                handles[path] = stack.enter_context(
                    safe_open(path, framework="pt", device="cpu")
                )
            return handles[path]

        return _load_original_checkpoint(model, root, rank, world_size, opened)


def _load_original_checkpoint(model, root, rank, world_size, opened):
    root = Path(root)
    index_bytes = (root / "model.safetensors.index.json").read_bytes()
    weight_map = json.loads(index_bytes)["weight_map"]
    sources = {canonical_name(name): name for name in weight_map}
    if len(sources) != len(weight_map):
        raise ValueError("checkpoint names collide after normalization")
    params = dict(model.named_parameters())
    used, skipped = set(), {}
    copied_bytes = 0

    def read(name, bounds=None):
        source = sources[name]
        path = root / weight_map[source]
        if path.resolve().parent != root.resolve():
            raise ValueError("shard path must remain inside the checkpoint")
        view = opened(path).get_slice(source)
        shape = tuple(view.get_shape())
        return (view[:] if bounds is None else view[bounds]), shape

    for i, (name, target) in enumerate(params.items()):
        if name not in sources:
            raise ValueError(f"missing required weight: {name}")
        source = sources[name]
        shape = tuple(opened(root / weight_map[source]).get_slice(source).get_shape())
        axis = shard_axis(name)
        eng = ".engram.embed." in name
        if ".experts." in name:
            axis = None  # EP, not TP: only locally owned experts exist in this module.
        bounds = [slice(None)] * len(shape)
        expected = list(shape)
        if axis is not None:
            size = (
                math.ceil(shape[axis] / world_size)
                if eng
                else shape[axis] // world_size
            )
            if not eng and shape[axis] % world_size:
                raise ValueError(f"unaligned TP weight: {name} {shape}")
            expected[axis] = size
            bounds[axis] = slice(rank * size, min((rank + 1) * size, shape[axis]))
        if tuple(expected) != tuple(target.shape):
            raise ValueError(
                f"{name}: checkpoint shard {expected} != parameter {list(target.shape)}"
            )

        if eng:
            # Huge tables are copied in <=64 MiB source chunks, including a
            # deterministic final padded row range on the last rank.
            target.view(torch.uint8).fill_(127 if name.endswith(".scale") else 0)
            lo, hi = bounds[0].start, bounds[0].stop
            rows = max(1, (64 << 20) // math.prod(shape[1:]))
            view = opened(root / weight_map[source]).get_slice(source)
            for begin in range(lo, hi, rows):
                end = min(begin + rows, hi)
                tensor = view[begin:end]
                if tensor.dtype != target.dtype:
                    raise ValueError(f"Engram dtype mismatch: {name}")
                target[begin - lo : end - lo].view(torch.uint8).copy_(
                    tensor.view(torch.uint8)
                )
        else:
            tensor, _ = read(name, tuple(bounds))
            if name.endswith("wo_a.weight"):
                scale_name = name.removesuffix("weight") + "scale"
                scale, _ = read(
                    scale_name,
                    (slice(bounds[0].start // 32, bounds[0].stop // 32), slice(None)),
                )
                tensor = (
                    (
                        tensor.float().unflatten(0, (-1, 32)).unflatten(-1, (-1, 32))
                        * scale.float()[:, None, :, None]
                    )
                    .flatten(2, 3)
                    .flatten(0, 1)
                    .bfloat16()
                )
                used.add(scale_name)
            if target.dtype == torch.float4_e2m1fn_x2:
                if tensor.dtype != torch.int8:
                    raise ValueError(f"expected original I8-packed E2M1: {name}")
                target.view(torch.uint8).copy_(tensor.view(torch.uint8))
            elif target.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                if tensor.dtype != target.dtype:
                    raise ValueError(
                        f"quantized dtype mismatch: {name}: {tensor.dtype} vs {target.dtype}"
                    )
                target.view(torch.uint8).copy_(tensor.view(torch.uint8))
            else:
                if tensor.dtype not in (torch.bfloat16, torch.float32):
                    raise ValueError(f"unexpected dense dtype: {name}: {tensor.dtype}")
                target.copy_(tensor)
        used.add(name)
        copied_bytes += target.numel() * target.element_size()
        if i % 1000 == 0:
            logger.warning(
                "FL rank %d loaded %d/%d tensors (%.2f GiB)",
                rank,
                i,
                len(params),
                copied_bytes / 2**30,
            )

    for name in sources.keys() - used:
        if name.startswith("mtp.") and not len(model.mtp):
            reason = "speculative decoding disabled"
        elif ".experts." in name:
            expert = int(name.split(".experts.")[1].split(".")[0])
            owner = model.get_submodule(name.split(".experts.")[0])
            count = owner.n_routed_experts // world_size
            if rank * count <= expert < (rank + 1) * count:
                raise ValueError(f"unconsumed local expert weight: {name}")
            reason = "owned by another expert rank"
        else:
            raise ValueError(f"unconsumed required checkpoint tensor: {name}")
        skipped[reason] = skipped.get(reason, 0) + 1
    torch.cuda.synchronize()
    return {
        "rank": rank,
        "world_size": world_size,
        "loaded_tensors": len(params),
        "parameter_bytes": copied_bytes,
        "skipped_tensor_counts": skipped,
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "layout": "original-row-major-k2-e8m0-g32",
        "full_payload_hashed": False,
    }
