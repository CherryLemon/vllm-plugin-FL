# SPDX-License-Identifier: Apache-2.0
"""Bounded CUDA Graph cache keyed by batch size, never absolute token position."""

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .batched_decode import DecodeBatch
from .models.deepseek_v41.model import MoE, set_dtype
from .models.deepseek_v41.ops import decode_dense_batch


@contextmanager
def device_routing(core):
    modules = [m for m in core.modules() if isinstance(m, MoE)]
    previous = [getattr(m, "device_routing", False) for m in modules]
    for module in modules:
        module.device_routing = True
    try:
        yield
    finally:
        for module, value in zip(modules, previous):
            module.device_routing = value


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    ids: torch.Tensor
    context: DecodeBatch
    hidden: torch.Tensor | None
    result: tuple


class BatchedDecodeGraphs:
    def __init__(self, model, state, device):
        self.model, self.state, self.device = model, state, device
        self.targets, self.drafts = {}, {}
        self.target_replays = self.draft_replays = 0
        self.capture_seconds = 0.0

    def stats(self):
        return {
            "enabled": True,
            "mode": "device_position_batch",
            "target_graphs": len(self.targets),
            "draft_graphs": len(self.drafts),
            "target_batch_sizes": sorted(self.targets),
            "draft_batch_sizes": sorted(self.drafts),
            "target_replays": self.target_replays,
            "draft_replays": self.draft_replays,
            "capture_seconds": self.capture_seconds,
            "page_copy_bytes": 0,
        }

    def _forward(self, token, context, hidden):
        core = self.model.core
        with (
            torch.device(self.device),
            set_dtype(torch.bfloat16),
            device_routing(core),
            decode_dense_batch(token.numel()),
        ):
            if hidden is not None:
                return core.forward_spec(token, hidden, context)
            _, logits, main_hidden = core(token[:, None], context)
            if self.model.speculative_config is not None:
                core.store_spec_context(main_hidden, context)
            return logits, main_hidden

    @torch.inference_mode()
    def _capture(self, tokens, pages, positions, active, hidden):
        begin = time.monotonic()
        ids = tokens.clone()
        context = DecodeBatch(
            self.state,
            pages.clone(),
            positions.reshape(-1, 1).clone(),
            torch.zeros_like(active.reshape(-1, 1)),
        )
        static_hidden = None if hidden is None else hidden.clone()
        # Warmup and capture must not commit a token twice. The mask lives on
        # device; it is updated to real activity before the first replay.
        self._forward(ids, context, static_hidden)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            result = self._forward(ids, context, static_hidden)
        record = _Graph(graph, ids, context, static_hidden, result)
        self.capture_seconds += time.monotonic() - begin
        return record

    @torch.inference_mode()
    def _replay(self, tokens, pages, positions, active, hidden=None):
        if (
            tokens.ndim != 1
            or tokens.dtype != torch.int64
            or pages.shape != tokens.shape
            or pages.dtype != torch.int64
            or positions.numel() != tokens.numel()
            or positions.dtype != torch.int64
            or active.numel() != tokens.numel()
            or active.dtype != torch.bool
            or any(t.device != tokens.device for t in (pages, positions, active))
        ):
            raise ValueError(
                "expected equally sized device token/page/position/activity vectors"
            )
        cache = self.targets if hidden is None else self.drafts
        batch = tokens.numel()
        record = cache.get(batch)
        if record is None:
            record = self._capture(tokens, pages, positions, active, hidden)
            cache[batch] = record
        record.ids.copy_(tokens)
        record.context.pages.copy_(pages)
        record.context.positions.copy_(positions.reshape(-1, 1))
        record.context.active.copy_(active.reshape(-1, 1))
        if hidden is not None:
            record.hidden.copy_(hidden)
        record.graph.replay()
        if hidden is None:
            self.target_replays += 1
        else:
            self.draft_replays += 1
        return record.result

    def target_batch(self, tokens, pages, positions, active):
        return self._replay(tokens, pages, positions, active)

    def draft_batch(self, tokens, hidden, pages, positions, active):
        return self._replay(tokens, pages, positions, active, hidden)

    def _single_metadata(self, position):
        if self.state.active_block is None:
            raise ValueError(
                "a request page must be selected for single-request replay"
            )
        return (
            torch.tensor([self.state.active_block], device=self.device),
            torch.tensor([position], device=self.device),
            torch.ones(1, device=self.device, dtype=torch.bool),
        )

    def target(self, tokens, position):
        return self.target_batch(tokens, *self._single_metadata(position))

    def draft(self, tokens, hidden, position):
        return self.draft_batch(tokens, hidden, *self._single_metadata(position))
