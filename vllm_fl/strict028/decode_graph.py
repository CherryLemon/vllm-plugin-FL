# SPDX-License-Identifier: Apache-2.0
"""Explicit CUDA Graph replay for the fixed-position FL Decode operations.

The reference model specializes cache slices and compressor branches on a
Python position. Each position therefore has its own graph. All graphs share
one scratch state page and graph memory pool; request pages are copied in and
out around replay. This keeps the scheduler's state ownership and FlagCX PD
transfers unchanged while moving the target and DSpark GPU work onto graphs.
"""

import time
from dataclasses import dataclass

import torch


@dataclass
class _TargetGraph:
    graph: torch.cuda.CUDAGraph
    token: torch.Tensor
    logits: torch.Tensor
    hidden: torch.Tensor | None


@dataclass
class _DraftGraph:
    graph: torch.cuda.CUDAGraph
    token: torch.Tensor
    hidden: torch.Tensor
    result: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class DecodeGraphs:
    def __init__(self, model, state, device):
        if state.graph_scratch is None:
            raise ValueError("Decode Graph requires a dedicated state scratch page")
        self.model = model
        self.state = state
        self.device = device
        self.pool = torch.cuda.graph_pool_handle()
        self.targets: dict[int, _TargetGraph] = {}
        self.drafts: dict[int, _DraftGraph] = {}
        self.target_replays = 0
        self.draft_replays = 0
        self.capture_seconds = 0.0
        self.page_copy_bytes = 0

    def stats(self):
        return {
            "enabled": True,
            "target_graphs": len(self.targets),
            "draft_graphs": len(self.drafts),
            "target_replays": self.target_replays,
            "draft_replays": self.draft_replays,
            "capture_seconds": self.capture_seconds,
            "page_copy_bytes": self.page_copy_bytes,
            "positions": sorted(self.targets),
        }

    def _active_page(self):
        block = self.state.active_block
        if block is None:
            raise ValueError("a request state page must be bound before Decode Graph")
        return block, self.state.storage[block]

    def _copy_in(self, page):
        self.state.graph_scratch.copy_(page)
        self.page_copy_bytes += self.state.page_bytes

    def _copy_out(self, page):
        page.copy_(self.state.graph_scratch)
        self.page_copy_bytes += self.state.page_bytes

    def _capture_target(self, position, sample):
        block, page = self._active_page()
        static_token = torch.empty_like(sample)
        static_token.copy_(sample)
        start = time.monotonic()

        def forward():
            logits, hidden = self.model.forward_with_aux(
                static_token, start_pos=position
            )
            if self.model.speculative_config is not None:
                self.model.store_draft_context(hidden, position)
            return logits, hidden

        try:
            self._copy_in(page)
            self.state.bind_graph_scratch()
            forward()  # Compile and initialize every kernel outside capture.
            torch.cuda.synchronize(self.device)
            self._copy_in(page)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, pool=self.pool, capture_error_mode="thread_local"
            ):
                logits, hidden = forward()
        finally:
            self.state.bind(block)
        record = _TargetGraph(graph, static_token, logits, hidden)
        self.targets[position] = record
        self.capture_seconds += time.monotonic() - start
        return record

    def target(self, sample, position):
        block, page = self._active_page()
        record = self.targets.get(position)
        if record is None:
            record = self._capture_target(position, sample)
        record.token.copy_(sample)
        self._copy_in(page)
        record.graph.replay()
        self._copy_out(page)
        self.target_replays += 1
        return record.logits, record.hidden

    def _capture_draft(self, position, token, hidden):
        block, page = self._active_page()
        static_token = torch.empty_like(token)
        static_hidden = torch.empty_like(hidden)
        static_token.copy_(token)
        static_hidden.copy_(hidden)
        start = time.monotonic()

        def forward():
            return self.model.propose_draft(
                static_token, static_hidden, position
            )

        try:
            self._copy_in(page)
            self.state.bind_graph_scratch()
            forward()
            torch.cuda.synchronize(self.device)
            self._copy_in(page)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, pool=self.pool, capture_error_mode="thread_local"
            ):
                result = forward()
        finally:
            self.state.bind(block)
        record = _DraftGraph(graph, static_token, static_hidden, result)
        self.drafts[position] = record
        self.capture_seconds += time.monotonic() - start
        return record

    def draft(self, token, hidden, position):
        block, page = self._active_page()
        record = self.drafts.get(position)
        if record is None:
            record = self._capture_draft(position, token, hidden)
        record.token.copy_(token)
        record.hidden.copy_(hidden)
        self._copy_in(page)
        record.graph.replay()
        self._copy_out(page)
        self.draft_replays += 1
        return record.result
