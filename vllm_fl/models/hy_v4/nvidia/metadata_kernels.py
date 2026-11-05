# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU producers for framework-owned Hy4 request metadata.

Async packet buffers and index maps are validated by AsyncInputStaging before
submission. Its event lease retains all packet/source owners through gather.
These producers do not own request scheduling or model numerical policy.
"""

from __future__ import annotations

import torch
from flaggems_vllm.utils import libentry, libtuner

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID
from vllm.v1.worker.block_table import SlotMappingMode


@triton.jit(do_not_specialize=["max_num_tokens"])
def _compute_slot_mapping_graph_kernel(
    max_num_tokens,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
    block_table_stride,  # max_num_blocks_per_req
    block_size,
    slot_mapping_ptr,  # [max_num_tokens], int64
    KV_CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_KV_BLOCK: tl.constexpr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    MAP_SLOTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == tl.num_programs(0) - 1:
        num_tokens = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
        # Device boundary changes on every replay, including shrinking batches.
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    if start_idx == end_idx:
        for i in range(0, block_table_stride, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                block_table_ptr + req_idx * block_table_stride + offsets,
                NULL_BLOCK_ID,
                mask=offsets < block_table_stride,
            )
    if not MAP_SLOTS:
        return

    virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        virtual_block_indices = pos // virtual_block_size
        virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        block_indices = (
            virtual_block_indices * BLOCKS_PER_KV_BLOCK
            + local_block_offsets // block_size
        )
        block_numbers = tl.load(
            block_table_ptr + row_offset + block_indices,
            mask=mask & is_local,
            other=0,
        ).to(tl.int64)
        slot_offsets = local_block_offsets % block_size
        slot_ids = block_numbers * block_size + slot_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


@triton.jit
def _compute_num_computed_tokens_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    num_computed_tokens_ptr,
    num_reqs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_reqs
    query_start = tl.load(query_start_loc_ptr + offsets, mask=mask, other=0)
    query_end = tl.load(query_start_loc_ptr + offsets + 1, mask=mask, other=0)
    seq_len = tl.load(seq_lens_ptr + offsets, mask=mask, other=0)
    tl.store(
        num_computed_tokens_ptr + offsets,
        seq_len - (query_end - query_start),
        mask=mask,
    )


def compute_common_attention_metadata(
    block_table, num_reqs, query_start_loc, positions, seq_lens, num_computed_tokens
):
    # Group iteration occurs during eager fallback/warm/capture only.
    for group in block_table.block_tables:
        mode = group.slot_mapping_mode
        if mode not in (SlotMappingMode.TOKEN_TO_KV_SLOT, SlotMappingMode.NONE):
            raise ValueError("unsupported slot mapping mode")
        _compute_slot_mapping_graph_kernel[(num_reqs + 1,)](
            group.max_num_batched_tokens,
            query_start_loc,
            positions,
            group.block_table.gpu,
            group.block_table.gpu.stride(0),
            group.block_size,
            group.slot_mapping.gpu,
            KV_CACHE_BLOCK_SIZE=group.kv_cache_block_size,
            BLOCKS_PER_KV_BLOCK=group.blocks_per_kv_block,
            TOTAL_CP_WORLD_SIZE=group.dcp_world_size,
            TOTAL_CP_RANK=group.dcp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=group.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            NULL_BLOCK_ID=NULL_BLOCK_ID,
            MAP_SLOTS=mode == SlotMappingMode.TOKEN_TO_KV_SLOT,
            BLOCK_SIZE=1024,
        )
    _compute_num_computed_tokens_kernel[(triton.cdiv(num_reqs, 256),)](
        query_start_loc,
        seq_lens,
        num_computed_tokens,
        num_reqs=num_reqs,
        BLOCK_SIZE=256,
    )


@triton.jit
def _prepare_positions_seq_lens_kernel(
    query_start_loc, computed, positions, seq_lens, BLOCK_SIZE: tl.constexpr
):
    req = tl.program_id(0)
    start = tl.load(query_start_loc + req).to(tl.int64)
    end = tl.load(query_start_loc + req + 1).to(tl.int64)
    if start == end:
        tl.store(seq_lens + req, 0)
        return
    base = tl.load(computed + req).to(tl.int64)
    tl.store(seq_lens + req, base + end - start)
    for i in range(start, end, BLOCK_SIZE):
        token = i + tl.arange(0, BLOCK_SIZE)
        tl.store(positions + token, base + token - start, mask=token < end)


def compute_step_common_attention_metadata(
    block_table,
    num_reqs,
    query_start_loc,
    positions,
    seq_lens,
    num_computed_tokens,
    mrope_state=None,
    mrope_output=None,
):
    _prepare_positions_seq_lens_kernel[(num_reqs,)](
        query_start_loc, num_computed_tokens, positions, seq_lens, BLOCK_SIZE=256
    )
    compute_common_attention_metadata(
        block_table, num_reqs, query_start_loc, positions, seq_lens, num_computed_tokens
    )

    if mrope_state is not None:
        decode_mrope_kernel[(num_reqs,)](
            query_start_loc,
            num_computed_tokens,
            mrope_state,
            mrope_output,
            ROW_STRIDE=mrope_output.stride(0),
            COL_STRIDE=mrope_output.stride(1),
            BLOCK_SIZE=256,
        )


@triton.jit
def decode_mrope_kernel(
    query_start_loc,
    computed_tokens,
    state,
    output,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.load(state) == 0:
        return
    req = tl.program_id(0)
    start = tl.load(query_start_loc + req).to(tl.int64)
    end = tl.load(query_start_loc + req + 1).to(tl.int64)
    # Empty padded rows must not read stale request delta/count values.
    if start == end:
        return
    base = tl.load(computed_tokens + req).to(tl.int64) + tl.load(state + 1 + req).to(
        tl.int64
    )
    for i in range(start, end, BLOCK_SIZE):
        token = i + tl.arange(0, BLOCK_SIZE)
        value = base + token - start
        for axis in range(3):
            tl.store(
                output + axis * ROW_STRIDE + token * COL_STRIDE, value, mask=token < end
            )


@triton.jit
def _compute_req_id_per_token_kernel(
    query_start_loc,
    output,
    counts,
    NUM_REQUESTS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(query_start_loc + req).to(tl.int64)
    if req == NUM_REQUESTS:
        # Native fill_(0) semantics for every padded token, including shrink.
        for i in range(start, CAPACITY, BLOCK_SIZE):
            token = i + tl.arange(0, BLOCK_SIZE)
            tl.store(output + token, 0, mask=token < CAPACITY)
        return
    end = tl.load(query_start_loc + req + 1).to(tl.int64)
    tl.store(counts + req, end - start)
    for i in range(start, end, BLOCK_SIZE):
        token = i + tl.arange(0, BLOCK_SIZE)
        tl.store(output + token, req, mask=token < end)


def populate_request_metadata(query_start_loc, request_ids, request_counts):
    """No allocation: fixed extents and current monotonic QSL are caller contracts."""
    num_reqs = request_counts.numel()
    _compute_req_id_per_token_kernel[(num_reqs + 1,)](
        query_start_loc,
        request_ids,
        request_counts,
        NUM_REQUESTS=num_reqs,
        CAPACITY=request_ids.numel(),
        BLOCK_SIZE=256,
    )


@libentry()
@triton.jit(do_not_specialize=["NMAP", "B"])
def _async_input_correct_kernel(
    P,
    NC,
    AC,
    VC,
    PP,
    PD,
    NMAP,
    B,
    CAP: tl.constexpr,
    VC_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.arange(0, BLOCK)
    prev = tl.load(P + 2 * NMAP + r, r < B, -1).to(tl.int32)
    cpu = tl.load(P + 2 * NMAP + B + r, r < B, 0).to(tl.int32)
    g = tl.maximum(prev, 0)
    drafts = tl.load(P + 2 * NMAP + 2 * B + g, (r < B) & (g < CAP), 0).to(tl.int32)
    old = tl.load(NC + g, (r < B) & (g < CAP), 0)
    valid = tl.load(VC + g, (r < B) & (g < VC_SIZE), 0).to(tl.int32)
    accepted = tl.load(AC + r, r < B, 0)
    all_drafts = tl.load(P + 2 * NMAP + 2 * B + r, r < CAP, 0).to(tl.int32)
    # Every old gather completes before any in-place permutation write, also
    # for cycles spanning warps. One CTA is intentional, not an occupancy tune.
    tl.debug_barrier()
    participating = (prev >= 0) & (drafts > 0)
    tl.store(NC + r, tl.where(participating, old + valid, cpu), r < B)
    tl.store(AC + r, tl.where(participating, valid, accepted), r < B)
    tl.store(PP + r, prev, r < B)
    tl.store(PD + r, all_drafts, r < CAP)


@libentry()
@triton.jit(do_not_specialize=["NMAP", "NSAMPLE"])
def _async_input_gather_kernel(
    P,
    OUT,
    SAMPLE,
    DRAFT,
    NMAP,
    NSAMPLE,
    SAMPLE_S0: tl.constexpr,
    DRAFT_S0: tl.constexpr,
    DRAFT_S1: tl.constexpr,
    DRAFT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = i < NMAP
    dst = tl.load(P + i, active, 0).to(tl.int32)
    src = tl.load(P + NMAP + i, active, 0).to(tl.int32)
    a = tl.load(SAMPLE + src * SAMPLE_S0, active & (i < NSAMPLE), 0)
    b = tl.load(
        DRAFT + (src // DRAFT_K) * DRAFT_S0 + (src % DRAFT_K) * DRAFT_S1,
        active & (i >= NSAMPLE),
        0,
    ).to(tl.int32)
    tl.store(OUT + dst, tl.where(i < NSAMPLE, a, b), active)


def async_input_correct(
    packet,
    num_computed,
    num_accepted,
    valid_counts,
    prev_positions,
    prev_drafts,
    *,
    num_maps,
    num_reqs,
    prev_capacity,
):
    """Exact native correction; packet indices must be CPU-validated."""
    _async_input_correct_kernel[(1,)](
        packet,
        num_computed,
        num_accepted,
        valid_counts,
        prev_positions,
        prev_drafts,
        num_maps,
        num_reqs,
        prev_capacity,
        valid_counts.numel(),
        triton.next_power_of_2(prev_capacity),
        num_warps=4,
    )


def async_input_gather(
    packet, out_ids, sampled_ids, draft_ids, *, num_maps, num_sampled
):
    """Sources are disjoint from out_ids; destination indices are unique."""
    if not num_maps:
        return
    _async_input_gather_kernel[(triton.cdiv(num_maps, 128),)](
        packet,
        out_ids,
        sampled_ids,
        draft_ids,
        num_maps,
        num_sampled,
        sampled_ids.stride(0),
        draft_ids.stride(0),
        draft_ids.stride(1),
        draft_ids.shape[1],
        128,
        num_warps=4,
    )


@triton.jit(do_not_specialize=["ACTIVE_B"])
def _mtp_tail_step_kernel(
    P,
    S,
    T,
    Q,
    OP,
    OS,
    SLOT,
    COUNT,
    PSTRIDE: tl.constexpr,
    OPSTRIDE: tl.constexpr,
    TSTRIDE: tl.constexpr,
    NCOLS: tl.constexpr,
    AXES: tl.constexpr,
    ACTIVE_B,
    BLOCK_SIZE: tl.constexpr,
    MAX_LEN: tl.constexpr,
):
    r = tl.program_id(0)
    active = r < ACTIVE_B
    begin = tl.load(Q + r)
    end = tl.load(Q + r + 1)
    active = active & (end - begin == 1)
    # Match eagle_step_slot_mapping_metadata_kernel, including its max-length
    # wrap-to-zero/seq-to-one rule. Padding/capture dummy rows never write KV.
    position = tl.load(P + r)
    new_position = position + 1
    exceeds = new_position >= MAX_LEN
    clamped = tl.where(exceeds, 0, new_position)
    block = tl.minimum(clamped // BLOCK_SIZE, NCOLS - 1)
    block_id = tl.load(T + r * TSTRIDE + block, mask=active, other=0)
    slot = block_id * BLOCK_SIZE + clamped % BLOCK_SIZE
    slot = tl.where(active & ~exceeds, slot, -1)
    seq = tl.load(S + r)
    seq = tl.minimum(tl.where(exceeds, 1, seq + 1), MAX_LEN)
    seq = tl.where(active, seq, 0)
    clamped = tl.where(active, clamped, 0)
    for axis in tl.static_range(AXES):
        # Native M-RoPE tail update copies temporal plane0 into all planes.
        # It does not independently add one to each spatial plane.
        tl.store(OP + axis * OPSTRIDE + r, clamped)
    tl.store(OS + r, seq)
    tl.store(SLOT + r, slot)
    tl.store(COUNT + r, active.to(tl.int32))


def mtp_tail_step(
    positions,
    seq_lens,
    block_table,
    query_start_loc,
    out_positions,
    out_seq_lens,
    out_slots,
    out_counts,
    *,
    active_batch: int,
    block_size: int,
    max_model_len: int,
):
    """All outputs are caller-owned; no allocation, CPU read or fallback."""
    tensors = (
        positions,
        seq_lens,
        block_table,
        query_start_loc,
        out_positions,
        out_seq_lens,
        out_slots,
        out_counts,
    )
    if not all(x.is_cuda and x.device == positions.device for x in tensors):
        raise ValueError("mtp_tail_step requires one CUDA device")
    rows = seq_lens.numel()
    if (
        positions.ndim not in (1, 2)
        or positions.shape[-1] != rows
        or out_positions.shape != positions.shape
        or positions.dtype != torch.int64
        or out_positions.dtype != torch.int64
        or seq_lens.ndim != 1
        or seq_lens.dtype != torch.int32
        or out_seq_lens.dtype != torch.int32
        or out_seq_lens.shape != (rows,)
        or block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.shape[0] != rows
        or block_table.shape[1] < 1
        or query_start_loc.dtype != torch.int32
        or query_start_loc.shape != (rows + 1,)
        or out_slots.dtype != torch.int64
        or out_slots.shape != (rows,)
        or out_counts.dtype != torch.int32
        or out_counts.shape != (rows,)
        or not 0 < active_batch <= rows
        or block_size <= 0
        or max_model_len <= 0
    ):
        raise ValueError("unsupported HY4 tail state layout")
    if (
        positions.stride(-1) != 1
        or out_positions.stride(-1) != 1
        or block_table.stride(1) != 1
        or any(
            not x.is_contiguous()
            for x in (seq_lens, query_start_loc, out_seq_lens, out_slots, out_counts)
        )
    ):
        raise ValueError("unsupported HY4 tail state stride")
    axes = positions.shape[0] if positions.ndim == 2 else 1
    _mtp_tail_step_kernel[(rows,)](
        positions,
        seq_lens,
        block_table,
        query_start_loc,
        out_positions,
        out_seq_lens,
        out_slots,
        out_counts,
        positions.stride(0) if positions.ndim == 2 else rows,
        out_positions.stride(0) if out_positions.ndim == 2 else rows,
        block_table.stride(0),
        block_table.shape[1],
        axes,
        active_batch,
        block_size,
        max_model_len,
        num_warps=1,
    )


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_TOKENS": tokens}, num_warps=warps, num_stages=1)
        for tokens in (16, 32, 64)
        for warps in (4, 8)
    ],
    key=["B", "N", "COLS", "COUNTS_I32", "SEQ_I32", "CS", "SR", "SC", "PADDED"],
    strategy="default",
)
@triton.jit
def _decode_indexer_metadata_kernel(
    Counts,
    Seq,
    Rows,
    Lengths,
    B: tl.constexpr,
    N: tl.constexpr,
    COLS: tl.constexpr,
    CS: tl.constexpr,
    SR: tl.constexpr,
    SC: tl.constexpr,
    COUNTS_I32: tl.constexpr,
    SEQ_I32: tl.constexpr,
    PADDED: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    request = tl.program_id(0)
    requests = tl.arange(0, BLOCK_B)
    counts = tl.load(Counts + requests * CS, requests < B, other=0).to(tl.int64)
    inclusive = tl.cumsum(counts, 0)
    exclusive = inclusive - counts
    start = tl.sum(tl.where(requests == request, exclusive, 0), 0)
    count = tl.load(Counts + request * CS).to(tl.int64)
    offsets = tl.arange(0, BLOCK_TOKENS).to(tl.int64)

    if COLS <= 1:
        sequence = tl.load(Seq + request * SR)
        # Preserve the old subtraction's dtype boundary before the I64 step
        # promotes it. Do not replace this with an all-I64 algebraic shortcut.
        if COUNTS_I32 and SEQ_I32:
            base_length = (
                (sequence.to(tl.int32) - count.to(tl.int32)).to(tl.int32).to(tl.int64)
            )
        else:
            base_length = sequence.to(tl.int64) - count

    # Same complete path for one/many requests and one/many tokens per request.
    # Zero-count requests perform no stores. Prefix intervals are disjoint.
    # Under counts>=0 and sum(counts)==N<=1024 this loop bound fits I32.
    # Prefix and address arithmetic remain I64.
    for first in tl.range(0, count.to(tl.int32), BLOCK_TOKENS):
        step = first + offsets
        token = start + step
        valid = (step < count) & (token < N) & (token >= 0)
        if COLS > 1:
            length = tl.load(
                Seq + request * SR + step * SC, valid & (step < COLS), other=0
            )
        else:
            length = base_length + step + 1
        tl.store(Rows + token, request.to(tl.int64), valid)
        tl.store(Lengths + token, length, valid)

    if PADDED:
        # N is output capacity P in this mode; the logical token count is the
        # GPU sum of counts. Complete prefix math above is unchanged.
        total = tl.sum(counts, 0)
        # Cover all tail rows even for B=1, P=1024. No cross-program overlap:
        # each program owns its striped tail blocks, all beyond the prefix.
        for first in tl.range(request * BLOCK_TOKENS, N, B * BLOCK_TOKENS):
            token = first + offsets
            tail = (token >= total) & (token < N)
            tl.store(Rows + token, 0, tail)
            tl.store(Lengths + token, 0, tail)


def _span(tensor):
    if tensor.numel() == 0:
        return None
    last = sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    first = tensor.data_ptr()
    return first, first + (last + 1) * tensor.element_size()


def _overlap(first, second):
    return (
        first is not None
        and second is not None
        and max(first[0], second[0]) < min(first[1], second[1])
    )


def decode_indexer_metadata(
    counts, seq_lens, rows_out, lengths_out, n: int, *, padded: bool = False
):
    """Exact valid-input equivalent of the frozen adapter metadata chain.

    Device-value contract: counts>=0, sum(counts)==n; wide-2D seq has enough
    columns for every count. No host read/sync checks these device values.
    In padded=True mode n is P, sum(counts)=N may be <=P, and tail N:P
    is fully initialized to safe row0/length0. Existing mode is unchanged.
    Capacity above n is not logical padding and is never written. Outputs
    must not alias one another or either input. No torch compute fallback.
    """
    tensors = (counts, seq_lens, rows_out, lengths_out)
    if not 0 <= n <= 1024:
        raise ValueError("logical decode token count must be in [0,1024]")
    if counts.ndim != 1 or not 0 <= counts.shape[0] <= 1024:
        raise ValueError("counts must be 1D with at most 1024 entries")
    b = counts.shape[0]
    if counts.dtype not in (torch.int32, torch.int64) or seq_lens.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise NotImplementedError(
            "only int32/int64 counts and sequence lengths are supported"
        )
    if (
        seq_lens.ndim not in (1, 2)
        or seq_lens.shape[0] != b
        or (seq_lens.ndim == 2 and seq_lens.shape[1] == 0 and n > 0)
    ):
        raise ValueError("sequence lengths must have shape (B,), (B,1), or (B,C)")
    if counts.device.type != "cuda" or any(
        value.device != counts.device for value in tensors
    ):
        raise NotImplementedError(
            "all metadata and outputs must be on the same CUDA device"
        )
    if any(stride < 0 for value in tensors for stride in value.stride()):
        raise NotImplementedError("negative strides are unsupported")
    cols = seq_lens.shape[1] if seq_lens.ndim == 2 else 1
    expected_length_dtype = seq_lens.dtype if cols > 1 else torch.int64
    if rows_out.dtype != torch.int64 or lengths_out.dtype != expected_length_dtype:
        raise ValueError("output dtypes must match original rows/lengths promotion")
    if any(
        value.ndim != 1 or value.shape[0] < n or value.stride(0) != 1
        for value in (rows_out, lengths_out)
    ):
        raise ValueError(
            "caller-owned outputs must be contiguous 1D buffers with capacity>=n"
        )
    spans = [_span(value) for value in tensors]
    if _overlap(spans[2], spans[3]) or any(
        _overlap(output, source) for output in spans[2:] for source in spans[:2]
    ):
        raise ValueError("outputs must not overlap each other or the inputs")
    if b == 0 and n != 0:
        raise ValueError("empty counts require logical n=0")
    if n:
        with torch.cuda.device(counts.device):
            _decode_indexer_metadata_kernel[(b,)](
                counts,
                seq_lens,
                rows_out,
                lengths_out,
                B=b,
                N=n,
                COLS=cols,
                CS=counts.stride(0),
                SR=seq_lens.stride(0),
                SC=seq_lens.stride(1) if seq_lens.ndim == 2 else 0,
                COUNTS_I32=counts.dtype == torch.int32,
                SEQ_I32=seq_lens.dtype == torch.int32,
                PADDED=padded,
                BLOCK_B=triton.next_power_of_2(b),
            )
    return rows_out[:n], lengths_out[:n]


def _compute_step_common_with_request_ids(
    block_table,
    num_reqs,
    query_start_loc,
    positions,
    seq_lens,
    num_computed_tokens,
    request_ids,
    request_counts,
    *extra_tensors,
):
    compute_step_common_attention_metadata(
        block_table,
        num_reqs,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        *extra_tensors,
    )
    populate_request_metadata(query_start_loc, request_ids, request_counts)
