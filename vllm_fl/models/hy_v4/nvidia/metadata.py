# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Current input facts, async event leases and graph-bound metadata ownership.

GPU producers are imported only when submitting work. Immutable CPU proof
construction and validation do not import FlagGems or initialize its backend.
"""

from __future__ import annotations

import logging
import uuid
import weakref
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CPUSequenceLengthsProof:
    """Immutable packet facts in the current reordered request order."""

    issuer: str
    generation: int
    request_ids: tuple[str, ...]
    query_starts: tuple[int, ...]
    computed: tuple[int, ...]
    prompt_lengths: tuple[int, ...]
    previous_positions: tuple[int, ...]
    previous_drafts: tuple[int, ...]


@dataclass(frozen=True)
class PrefillChunkSequenceBounds:
    """Per-request KV extents for a current paged prefill query slice."""

    step_proof: CPUSequenceLengthsProof
    request_start: int
    request_stop: int
    chunk_start: int
    chunk_end: int
    seq_lens: tuple[int, ...]


_current_cpu_seq_lens_proof: ContextVar[CPUSequenceLengthsProof | None] = ContextVar(
    "current_cpu_seq_lens_proof", default=None
)


def set_current_cpu_seq_lens_proof(proof: CPUSequenceLengthsProof | None) -> None:
    _current_cpu_seq_lens_proof.set(proof)


def get_current_cpu_seq_lens_proof() -> CPUSequenceLengthsProof | None:
    return _current_cpu_seq_lens_proof.get()


def validated_cpu_seq_lens_proof(
    stager,
    request_ids,
    previous_req_map,
    query_starts,
    upper,
    prefilling,
    prompt_lengths,
    num_reqs,
    num_reqs_padded,
    num_tokens,
    num_tokens_padded,
    for_capture,
    ubatch_slices,
):
    """Accept only immutable host facts from the current successful packet."""
    if (
        for_capture
        or ubatch_slices is not None
        or stager is None
        or not stager.in_step
        or stager.plan is None
        or num_reqs_padded != num_reqs
        or num_tokens_padded != num_tokens
        or any(
            x.device.type != "cpu"
            for x in (query_starts, upper, prefilling, prompt_lengths)
        )
        or previous_req_map is None
    ):
        return None
    plan = stager.plan
    proof = plan.cpu_seq_lens_proof
    if (
        proof is None
        or proof.issuer != stager._cpu_proof_issuer
        or proof.generation != stager._cpu_proof_generation
        or plan.num_reqs != num_reqs
        or plan.total_tokens != num_tokens
        or tuple(request_ids) != plan.req_ids
        or tuple(request_ids) != proof.request_ids
        or tuple(int(previous_req_map.get(r, -1)) for r in request_ids)
        != proof.previous_positions
        or len(proof.request_ids) != num_reqs
        or len(proof.query_starts) != num_reqs + 1
        or any(
            len(x) != num_reqs
            for x in (
                proof.computed,
                proof.prompt_lengths,
                proof.previous_positions,
                proof.previous_drafts,
            )
        )
    ):
        return None
    starts = tuple(int(x) for x in query_starts[: num_reqs + 1])
    if starts != proof.query_starts or starts[0] != 0 or starts[-1] != num_tokens:
        return None
    for i in range(num_reqs):
        q = starts[i + 1] - starts[i]
        computed = proof.computed[i]
        prompt = proof.prompt_lengths[i]
        if (
            q < 0
            or computed < 0
            or prompt < 0
            or int(upper[i]) != computed + q
            or int(prompt_lengths[i]) != prompt
            or bool(prefilling[i]) != (computed < prompt)
        ):
            return None
    return proof


def validated_prefill_chunk_bound(
    common,
    chunk,
    req_start,
    req_stop,
    num_decodes,
    decode_threshold,
    compress_ratio,
    dcp_world,
    pcp_world,
):
    """Bind a paged prefill chunk to immutable facts from this input step.

    Per-request lengths bound score columns; their sum bounds flattened
    gather offsets. They intentionally have different capacity contracts.
    """
    proof = common.cpu_seq_lens_proof
    if (
        proof is None
        or compress_ratio != 1
        or dcp_world != 1
        or pcp_world != 1
        or not num_decodes <= req_start < req_stop <= common.num_reqs
        or chunk.num_reqs != req_stop - req_start
        or common.num_reqs != len(proof.request_ids)
        or any(
            x.device.type != "cpu"
            for x in (
                common.query_start_loc_cpu,
                common.seq_lens_cpu_upper_bound,
                common.is_prefilling,
            )
        )
        or tuple(int(x) for x in common.query_start_loc_cpu) != proof.query_starts
    ):
        return None
    starts = proof.query_starts[req_start : req_stop + 1]
    if not (starts[0] <= chunk.token_start < chunk.token_end <= starts[-1]):
        return None
    bounds = []
    for i in range(req_start, req_stop):
        prev, draft = proof.previous_positions[i], proof.previous_drafts[i]
        query_len = proof.query_starts[i + 1] - proof.query_starts[i]
        computed = proof.computed[i]
        bound = computed + query_len
        if (
            query_len <= decode_threshold
            or computed < 0
            or not computed < proof.prompt_lengths[i]
            or not bool(common.is_prefilling[i])
            or not ((prev == -1 and draft == -1) or (prev >= 0 and draft == 0))
            or int(common.seq_lens_cpu_upper_bound[i]) != bound
        ):
            return None
        bounds.append(bound)
    total = sum(bounds)
    if (
        chunk.total_seq_lens != total
        or chunk.local_total_seq_lens != total
        or chunk.num_reqs != chunk.block_table.shape[0]
    ):
        return None
    return PrefillChunkSequenceBounds(
        proof, req_start, req_stop, chunk.token_start, chunk.token_end, tuple(bounds)
    )


def push_current_cpu_seq_lens_proof(proof):
    """Start a nested worker/forward proof lifetime."""
    return _current_cpu_seq_lens_proof.set(proof)


def reset_current_cpu_seq_lens_proof(token):
    """Restore the caller's proof context, including exceptional exits."""
    _current_cpu_seq_lens_proof.reset(token)


def build_input_maps(runner, scheduler_output, num_reqs, total_tokens, cu_tokens):
    """Exact native generic async recipe; normal/unchanged routes stay native."""
    sample_dst, sample_src, draft_dst, draft_src = [], [], [], []
    total_spec, same, maximum = 0, True, -1
    for current in range(num_reqs):
        previous = int(runner.prev_positions.np[current])
        if previous < 0:
            continue
        req_id = runner.input_batch.req_ids[current]
        draft_len = len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
        total_spec += draft_len
        last = int(cu_tokens[current]) - 1
        sample_dst.append(last - draft_len)
        sample_src.append(previous)
        draft_dst.extend(range(last - draft_len + 1, last + 1))
        start = previous * runner.prev_num_spec_tokens
        draft_src.extend(range(start, start + draft_len))
        same &= previous == last
        maximum = max(maximum, last)
    count = len(sample_dst)
    if not count or (same and maximum == count - 1):
        return None
    return SimpleNamespace(
        sample_dst=sample_dst,
        sample_src=sample_src,
        draft_dst=draft_dst,
        draft_src=draft_src,
        num_sampled=count,
        copy_base=count < total_tokens - total_spec,
    )


def pack_host_packet(host, dst, src, prev, cpu_values, previous_drafts):
    """CPU metadata only; its allocation and event lease belong to the runner."""
    n, b, capacity = len(dst), len(prev), len(previous_drafts)
    used = 2 * n + 2 * b + capacity
    host[:n], host[n : 2 * n] = dst, src
    host[2 * n : 2 * n + b] = prev
    host[2 * n + b : 2 * n + 2 * b] = cpu_values
    host[2 * n + 2 * b : used] = previous_drafts
    return used


def _cpu_sequence_proof(
    issuer,
    generation,
    request_ids,
    packet,
    num_maps,
    num_reqs,
    capacity,
    query_ends,
    prompt_lengths,
    previous_req_map,
):
    """Copy only the packet values actually sent for this input-prep lease."""
    ids = tuple(request_ids)
    ends = tuple(int(x) for x in query_ends)
    prompts = tuple(int(x) for x in prompt_lengths)
    if (
        len(ids) != num_reqs
        or len(set(ids)) != num_reqs
        or len(ends) != num_reqs
        or len(prompts) != num_reqs
    ):
        return None
    n, b = num_maps, num_reqs
    prev = tuple(int(x) for x in packet[2 * n : 2 * n + b])
    computed = tuple(int(x) for x in packet[2 * n + b : 2 * n + 2 * b])
    if (
        previous_req_map is None
        or tuple(int(previous_req_map.get(r, -1)) for r in ids) != prev
    ):
        return None
    drafts = tuple(
        (-1 if p == -1 else int(packet[2 * n + 2 * b + p]) if 0 <= p < capacity else -2)
        for p in prev
    )
    starts = (0,) + ends
    if (
        any(p < -1 or p >= capacity for p in prev)
        or any(d < -1 for d in drafts)
        or any(a > b for a, b in zip(starts, starts[1:]))
        or any(v < 0 for v in computed)
        or any(v < 0 for v in prompts)
    ):
        return None
    return CPUSequenceLengthsProof(
        issuer, generation, ids, starts, computed, prompts, prev, drafts
    )


class AsyncInputStaging:
    def __init__(self, runner):
        self.runner = weakref.ref(runner)
        self.capacity = int(runner.max_num_reqs)
        self.map_capacity = int(runner.max_num_tokens)
        size = 2 * self.map_capacity + 3 * self.capacity
        self.cpu = torch.empty(size, dtype=torch.int64, device="cpu", pin_memory=True)
        self.host = self.cpu.numpy()
        self.gpu = torch.empty(size, dtype=torch.int64, device=runner.device)
        self.plan = None
        self.in_step = False
        self._cpu_proof_generation = 0
        self._cpu_proof_issuer = uuid.uuid4().hex

    def begin_step(self):
        # Caller has just synchronized native prepare_inputs_event. Previous
        # H2D and GPU packet consumers are complete; only now overwrite/release.
        if self.in_step:
            raise RuntimeError("overlapping native input preparation")
        self.plan = None
        self._cpu_proof_generation += 1
        self.in_step = True

    def end_step(self):
        # Keep packet and dynamic source owners until NEXT begin after wait.
        self.in_step = False

    def prepare_and_correct(self, scheduler_output, num_reqs, total_tokens, cu_tokens):
        r = self.runner()
        if not self.in_step:
            raise RuntimeError("packet preparation outside native input-prep lease")
        if (
            r._pp_recv_work is not None
            or r.common_attention_metadata_graph is None
            or r.input_batch.prev_sampled_token_ids is None
            or not 1 <= num_reqs <= self.capacity <= 1024
        ):
            return False
        sample = r.input_batch.prev_sampled_token_ids
        draft = r._draft_token_ids
        if (
            not sample.is_cuda
            or sample.dtype != torch.int32
            or sample.ndim != 2
            or sample.shape[1] < 1
            or min(sample.stride()) < 0
        ):
            return False
        m = build_input_maps(r, scheduler_output, num_reqs, total_tokens, cu_tokens)
        if m is None:
            return False
        if m.draft_dst and (
            draft is None
            or draft.ndim != 2
            or draft.dtype not in (torch.int32, torch.int64)
            or not draft.is_cuda
            or draft.shape[1] < 1
            or min(draft.stride()) < 0
        ):
            return False
        if not m.draft_dst:
            draft = sample  # Never dereferenced by a draft lane.
        out = r.input_ids.gpu
        values = (
            out,
            r.num_computed_tokens,
            r.num_accepted_tokens.gpu,
            r.valid_sampled_token_count_gpu,
            r.prev_positions.gpu,
            r.prev_num_draft_tokens.gpu,
        )
        dtypes = (
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int64,
            torch.int32,
        )
        if any(
            not x.is_cuda
            or x.device != out.device
            or x.dtype != dtype
            or x.ndim != 1
            or x.stride(0) != 1
            for x, dtype in zip(values, dtypes)
        ):
            return False
        if sample.device != out.device or draft.device != out.device:
            return False
        if any(
            x.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
            for x in (sample, draft)
        ):
            return False
        dst, src = m.sample_dst + m.draft_dst, m.sample_src + m.draft_src
        n = len(dst)
        if (
            n > min(self.map_capacity, 2**31 - 1)
            or len(set(dst)) != n
            or any(x < 0 or x >= total_tokens for x in dst)
            or any(x >= sample.shape[0] for x in m.sample_src)
            or any(x < 0 or x >= draft.numel() for x in m.draft_src)
        ):
            return False
        if (
            out.numel() < total_tokens
            or sample.numel() > 2**31 - 1
            or draft.numel() > 2**31 - 1
        ):
            return False
        if len({x.untyped_storage().data_ptr() for x in values}) != len(values):
            return False
        prev = r.prev_positions.np[:num_reqs]
        valid = r.valid_sampled_token_count_gpu
        if (
            any(x < -1 or x >= valid.numel() for x in prev)
            or r.num_computed_tokens.numel() < self.capacity
            or r.prev_num_draft_tokens.gpu.numel() != self.capacity
            or r.prev_positions.gpu.numel() < num_reqs
            or r.num_accepted_tokens.gpu.numel() < num_reqs
        ):
            return False
        used = pack_host_packet(
            self.host,
            dst,
            src,
            prev,
            r.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].numpy(),
            r.prev_num_draft_tokens.np,
        )
        self.gpu[:used].copy_(self.cpu[:used], non_blocking=True)
        stream = torch.cuda.current_stream(r.device).cuda_stream
        self.plan = SimpleNamespace(
            num_maps=n,
            num_reqs=num_reqs,
            total_tokens=total_tokens,
            req_ids=tuple(r.input_batch.req_ids),
            copy_base=m.copy_base,
            num_sampled=m.num_sampled,
            sampled=sample,
            draft=draft,
            valid_counts=valid,
            stream=stream,
            original_draft=r._draft_token_ids if m.draft_dst else None,
        )
        from .metadata_kernels import async_input_correct

        async_input_correct(
            self.gpu,
            r.num_computed_tokens,
            r.num_accepted_tokens.gpu,
            valid,
            r.prev_positions.gpu,
            r.prev_num_draft_tokens.gpu,
            num_maps=n,
            num_reqs=num_reqs,
            prev_capacity=self.capacity,
        )
        self.plan.cpu_seq_lens_proof = _cpu_sequence_proof(
            self._cpu_proof_issuer,
            self._cpu_proof_generation,
            r.input_batch.req_ids,
            self.host,
            n,
            num_reqs,
            self.capacity,
            cu_tokens,
            r.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs],
            r.input_batch.prev_req_id_to_index,
        )
        return True

    def apply_if_prepared(self, scheduler_output, num_reqs, total_tokens):
        if self.plan is None:
            return False
        r, p = self.runner(), self.plan
        if (
            not self.in_step
            or p.num_reqs != num_reqs
            or p.total_tokens != total_tokens
            or p.req_ids != tuple(r.input_batch.req_ids)
            or p.sampled is not r.input_batch.prev_sampled_token_ids
            or p.valid_counts is not r.valid_sampled_token_count_gpu
            or (
                p.original_draft is not None
                and p.original_draft is not r._draft_token_ids
            )
            or torch.cuda.current_stream(r.device).cuda_stream != p.stream
        ):
            raise RuntimeError("native packet input-prep lease/stream changed")
        # Preserve original uploads for prompt/mixed new requests. CPU token
        # construction, bookkeeping, PP wait and all noneligible branches remain native.
        if r.enable_prompt_embeds:
            r.is_token_ids.copy_to_gpu(total_tokens)
        if p.copy_base:
            r.input_ids.copy_to_gpu(total_tokens)
            if r.enable_prompt_embeds:
                r.inputs_embeds.copy_to_gpu(total_tokens)
        from .metadata_kernels import async_input_gather

        async_input_gather(
            self.gpu,
            r.input_ids.gpu,
            p.sampled,
            p.draft,
            num_maps=p.num_maps,
            num_sampled=p.num_sampled,
        )
        return True


def make_async_input_staging(runner):
    if not runner.use_async_spec_decode or runner.prepare_inputs_event is None:
        return None
    return AsyncInputStaging(runner)


def _tensor_key(t):
    return (
        t.data_ptr(),
        tuple(t.shape),
        tuple(t.stride()),
        str(t.dtype),
        str(t.device),
    )


def _bindings(block_table, num_reqs, tensors):
    owners = list(tensors)
    groups = []
    for g in block_table.block_tables:
        owners.extend((g.block_table.gpu, g.slot_mapping.gpu))
        groups.append(
            (
                _tensor_key(g.block_table.gpu),
                _tensor_key(g.slot_mapping.gpu),
                g.block_size,
                g.kv_cache_block_size,
                g.blocks_per_kv_block,
                g.max_num_batched_tokens,
                g.dcp_world_size,
                g.dcp_rank,
                g.cp_kv_cache_interleave_size,
                g.slot_mapping_mode.value,
            )
        )
    return (
        (id(block_table), num_reqs, tuple(map(_tensor_key, tensors)), tuple(groups)),
        (block_table, *owners),
    )


class CommonAttentionMetadataGraphRunner:
    def __init__(self):
        from vllm.platforms import current_platform

        self.graphs = {}
        self.graph_pool = current_platform.get_global_graph_pool()
        self._missing_graph_keys = set()
        self._request_id_map = None
        self._request_id_counts = None
        self._request_id_generation = 0
        self._request_id_current = None

    def clear(self):
        # Caller synchronizes before releasing profile-time KV/input owners.
        self.graphs.clear()
        self._missing_graph_keys.clear()
        self._request_id_current = None
        self._request_id_map = None
        self._request_id_counts = None
        self._request_id_generation += 1

    def run(
        self,
        block_table,
        num_reqs,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        *,
        use_graph,
        capture,
        extra_tensors=(),
    ):
        # Expire every prior metadata lease before attempting a new submission.
        self._request_id_current = None
        self._request_id_generation += 1
        if not _request_id_inputs_supported(num_reqs, query_start_loc, positions):
            raise ValueError("unsupported native request-ID metadata layout")
        output = self._request_id_map
        capacity = positions.numel()
        if (
            output is None
            or output.numel() != capacity
            or output.device != positions.device
        ):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("request-ID map must be allocated before capture")
            output = torch.empty(
                (capacity,), dtype=torch.int32, device=positions.device
            )
            self._request_id_map = output
        counts = self._request_id_counts
        if (
            counts is None
            or counts.numel() != num_reqs
            or counts.device != positions.device
        ):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("request counts must be allocated before capture")
            counts = torch.empty(
                (num_reqs,), dtype=torch.int32, device=positions.device
            )
            self._request_id_counts = counts
        result = self._run_bound(
            block_table,
            num_reqs,
            query_start_loc,
            positions,
            seq_lens,
            num_computed_tokens,
            use_graph=use_graph,
            capture=capture,
            extra_tensors=(output, counts, *extra_tensors),
        )
        # Publish only after the exact eager/captured/replay producer succeeded.
        self._request_id_current = (query_start_loc, output, num_reqs, counts)
        return result

    def attach_req_id_per_token(self, common):
        """One native cm_base call, before shallow group copies or slicing.

        The full native QSL suffix must already be current, as required by the
        existing common producer. This method is never called on a sub-batch.
        """
        common._model_request_id_lease = None
        current = self._request_id_current
        if current is None:
            return False
        source, output, extent, counts = current
        qsl = common.query_start_loc
        if (
            qsl.dtype != torch.int32
            or qsl.device != source.device
            or qsl.dim() != 1
            or qsl.stride() != (1,)
            or qsl.data_ptr() != source.data_ptr()
            or qsl.untyped_storage() is not source.untyped_storage()
            or not 0 <= common.num_reqs <= extent
            or qsl.numel() != common.num_reqs + 1
            or not 0 <= common.num_actual_tokens <= output.numel()
            or common.query_start_loc_cpu.numel() != qsl.numel()
        ):
            return False
        common._model_request_id_lease = _RequestIdMapLease(
            self, self._request_id_generation, common, output, counts
        )
        return True

    def _run_bound(
        self,
        block_table,
        num_reqs,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        *,
        use_graph,
        capture,
        extra_tensors=(),
    ):
        from .metadata_kernels import _compute_step_common_with_request_ids

        compute = _compute_step_common_with_request_ids
        args = (
            block_table,
            num_reqs,
            query_start_loc,
            positions,
            seq_lens,
            num_computed_tokens,
            *extra_tensors,
        )
        if not use_graph:
            compute(*args)
            return False
        key, owners = _bindings(block_table, num_reqs, args[2:])
        key = (id(compute), *key)
        owners = (compute, *owners)
        entry = self.graphs.get(key)
        if entry is not None:
            entry[0].replay()
            return True
        if not capture:
            if key not in self._missing_graph_keys:
                logger.warning(
                    "Common metadata binding was not captured; using eager producer"
                )
                self._missing_graph_keys.add(key)
            compute(*args)
            return False
        # Native helper only writes metadata; no KV stores. Compile eagerly,
        # capture, replay once to materialize outputs, then retain all owners.
        compute(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            compute(*args)
        graph.replay()
        self.graphs[key] = (graph, owners)
        return True


def select_decode_deltas(
    req_ids,
    requests,
    scheduled_tokens,
    computed_cpu,
    prompt_length,
    *,
    uses_mrope,
    uses_xdrope_dim=0,
    multimodal_pruning=False,
    scheduled_encoder_inputs=None,
):
    """Return current ordered deltas only when every token is completion.

    prompt_length must be native length_from_prompt_token_ids_or_embeds.
    This makes no tensor/metadata cache and performs no GPU work.
    """
    if (
        not uses_mrope
        or uses_xdrope_dim
        or multimodal_pruning
        or scheduled_encoder_inputs
        or not req_ids
    ):
        return None
    deltas = []
    for i, req_id in enumerate(req_ids):
        req = requests[req_id]
        count = computed_cpu[i]
        scheduled = scheduled_tokens[req_id]
        delta = req.mrope_position_delta
        if (
            req.mrope_positions is None
            or delta is None
            or not -(1 << 63) <= int(delta) < (1 << 63)
            or scheduled <= 0
            or count < prompt_length(req.prompt_token_ids, req.prompt_embeds)
        ):
            return None
        deltas.append(int(delta))
    return tuple(deltas)


class _RequestIdMapLease:
    """Strong tensor owners, weak runner, and one successful submission token."""

    __slots__ = (
        "runner",
        "generation",
        "qsl",
        "qsl_cpu",
        "num_reqs",
        "num_tokens",
        "output",
        "counts",
    )

    def __init__(self, runner, generation, common, output, counts):
        self.runner = weakref.ref(runner)
        self.generation = generation
        self.qsl = common.query_start_loc
        self.qsl_cpu = common.query_start_loc_cpu
        self.num_reqs = common.num_reqs
        self.num_tokens = common.num_actual_tokens
        self.output = output
        self.counts = counts


def _request_id_inputs_supported(num_reqs, query_start_loc, positions):
    return (
        num_reqs > 0
        and query_start_loc.device.type == "cuda"
        and query_start_loc.device == positions.device
        and query_start_loc.dtype == torch.int32
        and query_start_loc.dim() == positions.dim() == 1
        and query_start_loc.numel() == num_reqs + 1
        and query_start_loc.stride() == positions.stride() == (1,)
        and positions.numel() > 0
    )


def _request_id_lease_from_common(common):
    lease = getattr(common, "_model_request_id_lease", None)
    if lease is None:
        return None
    runner = lease.runner()
    if (
        runner is None
        or runner._request_id_current is None
        or runner._request_id_generation != lease.generation
        or runner._request_id_current[1] is not lease.output
        or runner._request_id_current[3] is not lease.counts
        or common.query_start_loc is not lease.qsl
        or common.query_start_loc_cpu is not lease.qsl_cpu
        or common.num_reqs != lease.num_reqs
        or common.num_actual_tokens != lease.num_tokens
    ):
        return None
    return lease


def request_id_per_token_from_common(common):
    lease = _request_id_lease_from_common(common)
    return None if lease is None else lease.output[: lease.num_tokens]


def decode_lens_from_common(common, num_decodes):
    """Current full-batch prefix only; CP/spec/expand eligibility is the caller's."""
    lease = _request_id_lease_from_common(common)
    if lease is None or not 0 <= num_decodes <= lease.num_reqs:
        return None
    return lease.counts[:num_decodes]
