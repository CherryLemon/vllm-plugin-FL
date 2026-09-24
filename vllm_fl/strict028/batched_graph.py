# SPDX-License-Identifier: Apache-2.0
"""Bounded CUDA Graph cache keyed by batch size, never absolute token position."""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .batched_decode import DecodeBatch
from .models.deepseek_v41.model import MoE, set_dtype
from .models.deepseek_v41.ops import decode_dense_batch

logger = logging.getLogger(__name__)


def reclaim_capture_cache(device, stage):
    # CUDA graph instantiation allocates outside PyTorch's allocator. It cannot
    # reclaim the default pool's unused warmup blocks on an allocation failure.
    # Live state and graph-owned allocations remain retained by empty_cache().
    torch.cuda.empty_cache()
    logger.warning(
        "FL graph %s: device=%s free=%d allocated=%d reserved=%d",
        stage,
        device,
        torch.cuda.mem_get_info(device)[0],
        torch.cuda.memory_allocated(device),
        torch.cuda.memory_reserved(device),
    )


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
    def __init__(self, model, state, device, *, batch_capacity=None, verify_width=6):
        if batch_capacity is not None and batch_capacity < 1:
            raise ValueError("a fixed graph batch must have positive capacity")
        if verify_width not in (1, 2, 3, 6):
            raise ValueError("verify width must divide the six-position draft block")
        self.model, self.state, self.device = model, state, device
        self.batch_capacity = batch_capacity
        self.verify_width = verify_width
        self.targets, self.drafts = {}, {}
        self.verifiers = {}
        self.verify_replays = 0
        self.target_replays = self.draft_replays = 0
        self.capture_seconds = 0.0

    def stats(self):
        return {
            "enabled": True,
            "mode": "device_position_batch",
            "target_graphs": len(self.targets) + len(self.verifiers),
            "draft_graphs": len(self.drafts),
            "target_batch_sizes": sorted(
                set(self.targets) | {b for b, _ in self.verifiers}
            ),
            "draft_batch_sizes": sorted(self.drafts),
            "target_replays": self.target_replays,
            "draft_replays": self.draft_replays,
            "capture_seconds": self.capture_seconds,
            "page_copy_bytes": 0,
            "batch_capacity": self.batch_capacity,
            "verify_graphs": len(self.verifiers),
            "verify_replays": self.verify_replays,
            "verify_widths": sorted({s for _, s in self.verifiers}),
            "max_verify_width": self.verify_width,
            "verify_journal_bytes": sum(
                r.journal_bytes for r in self.verifiers.values()
            ),
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
        reclaim_capture_cache(self.device, "before capture")
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            result = self._forward(ids, context, static_hidden)
        reclaim_capture_cache(self.device, "before instantiate")
        graph.instantiate()
        reclaim_capture_cache(self.device, "after instantiate")
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
        real_batch = tokens.numel()
        batch = real_batch if self.batch_capacity is None else self.batch_capacity
        if real_batch > batch or not batch:
            raise ValueError("request batch exceeds the configured graph capacity")
        if real_batch < batch:
            # Every DP rank enters equal-sized EP collectives, including idle
            # ranks. Page zero is reserved, and inactive lanes never write it.
            def pad(value, tail_shape=()):
                output = value.new_zeros((batch, *tail_shape))
                output[:real_batch].copy_(value.reshape(real_batch, *tail_shape))
                return output

            tokens, pages = pad(tokens), pad(pages)
            positions, active = pad(positions), pad(active)
            if hidden is not None:
                hidden = pad(hidden, hidden.shape[1:])
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
        return tuple(
            None if value is None else value[:real_batch] for value in record.result
        )

    def target_batch(self, tokens, pages, positions, active):
        return self._replay(tokens, pages, positions, active)

    def _verify_partitioned(self, tokens, pages, positions, active):
        # Reuse one graph for consecutive position blocks. Clone its outputs
        # before replaying it again: every replay writes the same result buffers.
        selections = []
        total = torch.zeros_like(pages)
        alive = active[:, 0]
        last_hidden = None
        all_finite = None
        previous = None
        for begin in range(0, tokens.shape[1], self.verify_width):
            end = min(begin + self.verify_width, tokens.shape[1])
            if previous is not None:
                alive = alive & (previous == tokens[:, begin])
            mask = active[:, begin:end] & alive[:, None]
            selected, kept, hidden, finite = self.verify_batch(
                tokens[:, begin:end], pages, positions[:, begin:end], mask
            )
            selected, kept, finite = selected.clone(), kept.clone(), finite.clone()
            hidden = None if hidden is None else hidden.clone()
            selections.append(selected)
            total = total + kept
            if hidden is not None:
                last_hidden = hidden if last_hidden is None else torch.where(
                    (kept > 0)[:, None, None], hidden, last_hidden
                )
            all_finite = finite if all_finite is None else all_finite & finite
            alive = alive & (kept == end - begin)
            previous = selected[:, -1]
        return torch.cat(selections, 1), total, last_hidden, all_finite

    @torch.inference_mode()
    def verify_batch(self, tokens, pages, positions, active):
        """Return per-position selections, committed lengths and final context."""
        from .verify_graph import capture_verify

        if (
            tokens.ndim != 2
            or not 1 <= tokens.shape[1] <= 6
            or tokens.dtype != torch.int64
            or pages.shape != tokens.shape[:1]
            or pages.dtype != torch.int64
            or positions.shape != tokens.shape
            or positions.dtype != torch.int64
            or active.shape != tokens.shape
            or active.dtype != torch.bool
            or any(t.device != tokens.device for t in (pages, positions, active))
        ):
            raise ValueError(
                "expected device ids/positions/active[B,S], pages[B], S<=6"
            )
        if tokens.shape[1] > self.verify_width:
            return self._verify_partitioned(tokens, pages, positions, active)
        real, width = tokens.shape
        batch = real if self.batch_capacity is None else self.batch_capacity
        if real > batch or not batch:
            raise ValueError("verify batch exceeds graph capacity")
        if real < batch:

            def pad(value):
                out = value.new_zeros((batch, *value.shape[1:]))
                out[:real].copy_(value)
                return out

            tokens, pages, positions, active = map(
                pad, (tokens, pages, positions, active)
            )
        key = (batch, width)
        record = self.verifiers.get(key)
        if record is None:
            begin = time.monotonic()
            record = capture_verify(self, tokens, pages, positions, active)
            self.verifiers[key] = record
            self.capture_seconds += time.monotonic() - begin
        record.ids.copy_(tokens)
        record.context.pages.copy_(pages)
        record.context.positions.copy_(positions)
        record.context.active.copy_(active)
        record.graph.replay()
        self.target_replays += 1
        self.verify_replays += 1
        selected, kept, hidden, finite = record.result
        return (
            selected[:real],
            kept[:real],
            None if hidden is None else hidden[:real],
            finite,
        )

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
