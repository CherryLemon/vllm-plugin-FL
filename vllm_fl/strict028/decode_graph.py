# SPDX-License-Identifier: Apache-2.0
"""Explicit CUDA Graph replay for the fixed-position FL Decode operations.

The reference model specializes cache slices and compressor branches on a
Python position. Each position therefore has its own graph. All graphs share
one scratch state page; request pages are copied in and out around replay.
Each graph has a separate allocator pool because DSpark acceptance changes the
replay order of target and draft positions. This keeps scheduler state
ownership and FlagCX PD transfers unchanged.
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
    external_inputs: tuple[torch.Tensor, ...]


@dataclass
class _DraftGraph:
    graph: torch.cuda.CUDAGraph
    token: torch.Tensor
    hidden: torch.Tensor
    result: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    external_inputs: tuple[torch.Tensor, ...]


class DecodeGraphs:
    def __init__(self, model, state, device):
        if state.graph_scratch is None:
            raise ValueError("Decode Graph requires a dedicated state scratch page")
        self.model = model
        self.state = state
        self.device = device
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

    def _external_inputs(self, position, *, draft):
        """Retain tensors created before capture that its kernels still read.

        The model caches sliding-window and DSpark index tensors in separate
        one-entry LRUs. A later position evicts those tensors, but a captured
        graph retains their raw device addresses. Each graph must own a Python
        reference until it is destroyed.
        """
        args = getattr(self.model, "args", None)
        if args is None:
            return ()
        from .models.deepseek_v41.model import (
            get_dspark_topk_idxs,
            get_window_topk_idxs,
        )

        with torch.device(self.device):
            if draft:
                return (
                    get_dspark_topk_idxs(
                        args.window_size, 1, args.dspark_block_size, position
                    ),
                )
            return (get_window_topk_idxs(args.window_size, 1, 1, position),)

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
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                logits, hidden = forward()
        finally:
            self.state.bind(block)
        record = _TargetGraph(
            graph, static_token, logits, hidden,
            self._external_inputs(position, draft=False),
        )
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
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                result = forward()
        finally:
            self.state.bind(block)
        record = _DraftGraph(
            graph, static_token, static_hidden, result,
            self._external_inputs(position, draft=True),
        )
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
