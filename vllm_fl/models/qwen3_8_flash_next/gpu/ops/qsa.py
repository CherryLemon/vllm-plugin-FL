# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.24 import facade for the official QSA kernels.

The implementation deliberately lives in ``vendor.official.qsa`` and is
copied from the Qwen3.8-Flash-Next image. Keeping this module as aliases means
existing vLLM 0.24 model code can retain its import path without silently
selecting the former self-developed FlagOS kernels.
"""

from __future__ import annotations

from ...vendor.official.qsa import (
    expand_qsa_block_indices_cuda,
    qsa_compress_groups_with_ratio,
    qsa_mqa_paged,
    qsa_select_paged_tokens,
    qsa_sparse_paged_attention,
    qsa_store_cache_rows,
)

# The old plugin tests and a few downstream callers used the shorter helper
# name. It is an alias, not a second implementation.
expand_qsa_block_indices = expand_qsa_block_indices_cuda

__all__ = [
    "expand_qsa_block_indices",
    "expand_qsa_block_indices_cuda",
    "qsa_compress_groups_with_ratio",
    "qsa_mqa_paged",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
]
