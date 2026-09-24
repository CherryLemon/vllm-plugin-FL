# SPDX-License-Identifier: Apache-2.0
"""Layer-wise speculative verification with a bounded row-level undo journal.

Independent positions share the model's dense/MoE execution. Stateful attention
advances positions in causal order within each layer. Only rows overwritten by
speculation are retained; rejected suffixes never survive the verify graph.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .batched_decode import DecodeBatch
from .models.deepseek_v41.model import set_dtype
from .models.deepseek_v41.ops import decode_dense_batch


class _Position(DecodeBatch):
    def __init__(self, owner, offset):
        self.owner, self.offset = owner, offset
        super().__init__(
            owner.state,
            owner.pages,
            owner.positions[:, offset : offset + 1],
            owner.active[:, offset : offset + 1],
        )

    def write(self, pool, values, positions, active=None):
        # Gather only the overwritten row, before writing it. Keep the original
        # write mask: a non-publishing compressor/index operation has no undo.
        mask = self.active if active is None else active
        old = pool[self.pages[:, None], positions.clamp(0, pool.shape[1] - 1)]
        self.owner.journal.append((self, pool, old, positions, mask))
        super().write(pool, values, positions, mask.contiguous())


class VerifyBatch:
    def __init__(self, state, pages, positions, active):
        self.state, self.pages = state, pages
        self.positions, self.active = positions, active
        self.batch, self.width = positions.shape
        self.journal = []
        self.contexts = [_Position(self, offset) for offset in range(self.width)]

    @contextmanager
    def causal_positions(self):
        # Each attention invocation retains the admitted B x 1 reduction shape.
        with decode_dense_batch(self.batch):
            yield

    def _split(self, x):
        return x.reshape(self.batch, self.width, *x.shape[2:])

    def _merge(self, rows):
        x = torch.cat(rows, dim=1)
        return x.reshape(self.batch * self.width, 1, *x.shape[2:])

    def hash(self, module, input_ids, token_mask=None):
        ids = self._split(input_ids)
        masks = None if token_mask is None else self._split(token_mask)
        return self._merge(
            [
                ctx.hash(
                    module,
                    ids[:, i : i + 1],
                    None if masks is None else masks[:, i : i + 1],
                )
                for i, ctx in enumerate(self.contexts)
            ]
        )

    def attention(self, attn, x):
        x = self._split(x)
        with self.causal_positions():
            return self._merge(
                [
                    ctx.attention(attn, x[:, i : i + 1])
                    for i, ctx in enumerate(self.contexts)
                ]
            )

    def store_draft_context(self, attn, hidden):
        hidden = self._split(hidden)
        with self.causal_positions():
            for i, ctx in enumerate(self.contexts):
                ctx.store_draft_context(attn, hidden[:, i : i + 1])

    def rollback(self, kept):
        # Reverse order is necessary when several speculative positions overwrite
        # the same compressor slot or ring row.
        for ctx, pool, old, positions, mask in reversed(self.journal):
            DecodeBatch.write(
                ctx, pool, old, positions, mask & (kept[:, None] <= ctx.offset)
            )


@dataclass
class VerifyGraph:
    graph: torch.cuda.CUDAGraph
    ids: torch.Tensor
    context: VerifyBatch
    result: tuple
    journal_bytes: int


def _forward(owner, ids, context):
    from .batched_graph import device_routing

    core = owner.model.core
    context.journal = []
    with (
        torch.device(owner.device),
        set_dtype(torch.bfloat16),
        device_routing(core),
        decode_dense_batch(ids.numel()),
    ):
        _, logits, hidden = core(ids.reshape(-1, 1), context)
        if owner.model.speculative_config is not None:
            core.store_spec_context(hidden, context)
        logits = logits.reshape(*ids.shape, -1)
        selected = logits.argmax(-1)
        accepted = (selected[:, :-1] == ids[:, 1:]) & context.active[:, 1:]
        kept = (1 + accepted.long().cumprod(-1).sum(-1)) * context.active[:, 0]
        finite = torch.isfinite(logits).all(-1)
        finite = (finite | ~context.active).all()
        context.rollback(kept)
        if hidden is not None:
            hidden = hidden.reshape(*ids.shape, -1)
            hidden = hidden[
                torch.arange(ids.shape[0], device=ids.device), (kept - 1).clamp_min(0)
            ].unsqueeze(1)
        return selected, kept, hidden, finite


@torch.inference_mode()
def capture_verify(owner, tokens, pages, positions, active):
    ids = tokens.clone()
    context = VerifyBatch(
        owner.state, pages.clone(), positions.clone(), torch.zeros_like(active)
    )
    _forward(owner, ids, context)
    # Warmup's journal must not retain temporary GPU allocations during capture.
    context.journal = []
    torch.cuda.synchronize(owner.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        result = _forward(owner, ids, context)
    journal_bytes = sum(
        old.numel() * old.element_size() for _, _, old, _, _ in context.journal
    )
    return VerifyGraph(graph, ids, context, result, journal_bytes)
