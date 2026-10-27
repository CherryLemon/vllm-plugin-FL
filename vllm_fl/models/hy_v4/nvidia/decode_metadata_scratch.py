# SPDX-License-Identifier: Apache-2.0
"""Fixed metadata-r1 slot; real eager callback owns one bounded workspace.

No host read of device values, arithmetic shortcut, capture, collective, I/O or
native output change. The sealed generic operator computes the complete chain.
"""

from dataclasses import dataclass
import threading

import torch

CAPACITY = 1024
SELECTOR_VERSION = 1
VARIANTS = ("baseline", "meta-r1")
# TP, EP, DP, PP, DCP, PCP, noSpec, noLoRA, maximum batched tokens.
EXPECTED_CONFIGURATION = (16, True, 1, 1, 1, 1, True, True, 1024)


@dataclass
class _DecodeMetadataScratch:
    workspace: object
    device: object
    stream_id: int
    thread_id: int
    uses: int = 0
    affinity_fallbacks: int = 0
    last_n: int | None = None
    last_batch: int | None = None
    last_seq_shape: tuple = ()
    last_lengths_dtype: object = None


def maybe_prepare_decode_metadata(indexer, meta, counts, seq_lens, n, device):
    """Return exact owned rows/lengths views, or None for the full old chain.

    Device-value preconditions are the sealed native producer's complete
    counts>=0 / sum(counts)==logical n contract. Wide sequence layouts also need
    enough columns for each count. Nothing infers a count value from a shape.
    """
    variant = getattr(indexer, "_hy4_int8_decode_metadata_variant", "baseline")
    if variant == "baseline":
        return None
    if (
        variant != "meta-r1"
        or getattr(indexer, "_hy4_int8_decode_metadata_selector_version", None)
        != SELECTOR_VERSION
    ):
        raise RuntimeError("unsupported decode metadata selector version/variant")
    if (
        getattr(indexer, "_hy4_int8_decode_metadata_configuration", None)
        != EXPECTED_CONFIGURATION
    ):
        return None
    types = indexer._hy4_int8_decode_metadata_types
    if (
        type(meta) is not types[0]
        or type(meta.decode) is not types[1]
        or meta.decode.decode_lens is not counts
        or meta.decode.seq_lens is not seq_lens
        or meta.num_decode_tokens != n
        or type(n) is not int
        or not 1 <= n <= CAPACITY
        or not isinstance(counts, torch.Tensor)
        or not isinstance(seq_lens, torch.Tensor)
        or counts.ndim != 1
        or not 1 <= counts.shape[0] <= CAPACITY
        or counts.dtype not in (torch.int32, torch.int64)
        or seq_lens.dtype not in (torch.int32, torch.int64)
        or seq_lens.ndim not in (1, 2)
        or seq_lens.shape[0] != counts.shape[0]
        or (seq_lens.ndim == 2 and seq_lens.shape[1] < 1)
        or counts.layout != torch.strided
        or seq_lens.layout != torch.strided
        or device.type != "cuda"
        or counts.device != device
        or seq_lens.device != device
        or any(s < 0 for tensor in (counts, seq_lens) for s in tensor.stride())
    ):
        return None
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    # Native PIECEWISE requests build unpadded attention metadata. FULL/NONE or
    # either capture state must retain the original producer/consumer route.
    if (
        not is_forward_context_available()
        or get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
        or BreakableCUDAGraphCapture.current() is not None
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    stream_id = torch.cuda.current_stream(device).cuda_stream
    thread_id = threading.get_ident()
    state = getattr(indexer, "_hy4_int8_decode_metadata_scratch", None)
    if state is not None and (
        state.device != device
        or state.stream_id != stream_id
        or state.thread_id != thread_id
    ):
        state.affinity_fallbacks += 1
        return None
    if state is None:
        from .decode_metadata import DecodeMetadataWorkspace

        state = _DecodeMetadataScratch(
            workspace=DecodeMetadataWorkspace(CAPACITY, device),
            device=device,
            stream_id=stream_id,
            thread_id=thread_id,
        )
        indexer._hy4_int8_decode_metadata_scratch = state
    rows, lengths = state.workspace.prepare(counts, seq_lens, n)
    state.uses += 1  # Host return/submission only; not completion/numerics.
    state.last_n, state.last_batch = n, counts.shape[0]
    state.last_seq_shape, state.last_lengths_dtype = (
        tuple(seq_lens.shape),
        lengths.dtype,
    )
    return rows, lengths
