# SPDX-License-Identifier: Apache-2.0
"""One bounded, exact-token prefix snapshot for preparing independent PD requests.

Only incomplete Prefill prefixes are cached. A hit restores the complete
request state, including compressor tails and DSpark history. The final prompt
tokens always run through the model, and the scheduler retains its ordinary
request/page ownership. Total prompt length is part of the key because the
reference projection geometry depends on it.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefixSnapshot:
    tokens: tuple[int, ...]
    prompt_length: int
    page: torch.Tensor


class PrefillPrefixCache:
    def __init__(self):
        self.entry = None
        self.hits = 0
        self.reused_tokens = 0
        self.snapshots = 0

    def lookup(self, tokens, prompt_length):
        entry = self.entry
        if (
            entry is None
            or prompt_length != entry.prompt_length
            or tuple(tokens[: len(entry.tokens)]) != entry.tokens
        ):
            return None
        self.hits += 1
        return entry

    def restore(self, state, block, entry):
        # This blocking copy completes before the model can consume the page.
        # Each request owns its GPU copy; replay never mutates the CPU entry.
        state.storage[block].copy_(entry.page)

    def record(self, tokens, prompt_length, end, state, block):
        if not 0 < end < prompt_length:
            return
        entry = self.entry
        if (
            entry is not None
            and prompt_length == entry.prompt_length
            and end <= len(entry.tokens)
            and tuple(tokens[:end]) == entry.tokens[:end]
        ):
            return
        # Immutable entries also keep an in-flight hit valid after eviction.
        page = state.storage[block].to(device="cpu", copy=True)
        self.entry = PrefixSnapshot(tuple(tokens[:end]), prompt_length, page)
        self.snapshots += 1

    def stats(self):
        return dict(
            hits=self.hits,
            reused_tokens=self.reused_tokens,
            snapshots=self.snapshots,
            cached_tokens=0 if self.entry is None else len(self.entry.tokens),
            page_bytes=0 if self.entry is None else self.entry.page.numel()
            * self.entry.page.element_size(),
        )
