# SPDX-License-Identifier: Apache-2.0
"""One scheduler-owned opaque state page per request for P1 Eager inference.

block_size equals the deployment context limit. Each page includes every layer's
window, compressed KV, index K, incomplete compressor state and Engram history.
Prefix reuse/chunked prefill are not supported by this reference layout.
"""

import hashlib
import json
from dataclasses import dataclass

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec


@dataclass(frozen=True, kw_only=True)
class FLRequestStateSpec(FullAttentionSpec):
    state_page_bytes: int
    layout_version: str = "dsv41-eager-state-v1"
    state_layout_hash: str = ""

    @property
    def page_size_bytes(self):
        return self.state_page_bytes

    @classmethod
    def merge(cls, specs):
        if not specs or any(spec != specs[0] for spec in specs):
            raise ValueError("request state specs must match exactly")
        return specs[0]


@dataclass(frozen=True)
class StateField:
    module: str
    name: str
    shape: tuple
    dtype: torch.dtype
    offset: int
    nbytes: int
    fill: float


class RequestState:
    def __init__(self, model, max_model_len):
        self.model = model
        self.fields = []
        offset = 0
        state_names = {
            "window_kv_cache",
            "compress_kv_cache",
            "k_cache",
            "kv_state",
            "score_state",
        }
        for module_name, module in model.named_modules():
            for name, tensor in module.named_buffers(recurse=False):
                if name not in state_names and (module_name, name) != (
                    "engram_hash",
                    "cache",
                ):
                    continue
                offset = (offset + 255) // 256 * 256
                size = tensor.numel() * tensor.element_size()
                self.fields.append(
                    StateField(
                        module_name,
                        name,
                        tuple(tensor.shape),
                        tensor.dtype,
                        offset,
                        size,
                        -float("inf") if name == "score_state" else 0,
                    )
                )
                offset += size
        if not self.fields:
            raise ValueError("model exposes no request state")
        self.page_bytes = (offset + 255) // 256 * 256
        self.layout_hash = hashlib.sha256(
            json.dumps(
                [
                    (
                        field.module,
                        field.name,
                        field.shape,
                        str(field.dtype),
                        field.offset,
                        field.nbytes,
                    )
                    for field in self.fields
                ],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.spec = FLRequestStateSpec(
            block_size=max_model_len,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            state_page_bytes=self.page_bytes,
            state_layout_hash=self.layout_hash,
        )
        self.storage = None
        self.active_block = None

    def allocate(self, kv_cache_config, device):
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("expected one complete request-state group")
        group = kv_cache_config.kv_cache_groups[0]
        if group.kv_cache_spec != self.spec or group.layer_names != [
            "fl_request_state"
        ]:
            raise ValueError("scheduler state layout differs from model")
        actual_bytes = sum(tensor.size for tensor in kv_cache_config.kv_cache_tensors)
        expected_bytes = kv_cache_config.num_blocks * self.page_bytes
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"cache allocation mismatch: {actual_bytes} != {expected_bytes}"
            )
        self.storage = torch.zeros(
            kv_cache_config.num_blocks,
            self.page_bytes,
            dtype=torch.uint8,
            device=device,
        )
        # Release construction-time request buffers; all future views are pages
        # whose allocation, ownership and reuse are managed by the host scheduler.
        self.bind(0, reset=True)

    def bind(self, block_id, *, reset=False):
        if self.storage is None or not 0 <= block_id < self.storage.shape[0]:
            raise ValueError("invalid request-state block")
        page = self.storage[block_id]
        if reset:
            # PD copies the whole page, including alignment gaps. Never expose
            # bytes left by the previous request that owned this block.
            page.zero_()
        for field in self.fields:
            module = self.model.get_submodule(field.module)
            view = (
                page[field.offset : field.offset + field.nbytes]
                .view(field.dtype)
                .view(field.shape)
            )
            setattr(module, field.name, view)
            if reset and field.fill != 0:
                view.fill_(field.fill)
        # Shared references must point to the current page even if no new
        # compression group completes on this decode token.
        from .models.deepseek_v41.model import shared_attn

        shared_attn.compress_kv = None
        shared_attn.index_k = None
        shared_attn.topk_idxs = None
        shared_attn.candidates = None
        self.active_block = block_id
