# SPDX-License-Identifier: Apache-2.0
"""Optional library-owned MHC broadcast specialization."""
from flaggems_vllm import mhc_pre_broadcast_tilelang as _op

from .indexer_backend import INDEXER_BACKEND

if not _op._is_available():
    raise ImportError("The optional MHC broadcast capability is unavailable")


def mhc_pre_broadcast_tilelang(*args, **kwargs):
    return INDEXER_BACKEND._kpool("mhc_pre_broadcast_tilelang", *args, **kwargs)
