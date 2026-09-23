# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.28 Worker for FL reference Prefill and explicit Decode Graph replay."""

import json
import logging
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from .cache import RequestState
from .collectives import init_tp_collectives, parallel_layout, requested_backend
from .model_loader import FLDeepseekV41Loader
from .pd_connector import DeepseekV41FLConnector
from .sampling import validate_sampling

logger = logging.getLogger(__name__)


@dataclass
class Request:
    tokens: list[int]
    block: int
    computed: int


class ModelRunnerFL028:
    def __init__(self, model, state, device, *, pd_enabled=False):
        self.model = model
        self.state = state
        self.device = device
        self.pd_enabled = pd_enabled
        self.data_parallel = parallel_layout().data_size > 1
        self.requests = {}
        self.drafting_enabled = model.speculative_config is not None
        self.draft_token_ids = None
        self.graph_enabled = os.environ.get("VLLM_FL_DECODE_GRAPH") == "1"
        self.batched_decode_enabled = os.environ.get("VLLM_FL_BATCHED_DECODE") == "1"
        if self.batched_decode_enabled and not self.graph_enabled:
            raise ValueError("batched Decode requires VLLM_FL_DECODE_GRAPH=1")
        self.graphs = None
        self.spec_stats = {
            "draft_steps": 0,
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "verified_steps": 0,
            "target_forward_calls": 0,
            "accepted_prefix_histogram": [0] * 6,
        }

    def take_draft_token_ids(self):
        result, self.draft_token_ids = self.draft_token_ids, None
        return result

    def target_forward(self, tokens, position):
        ids = torch.tensor(tokens, device=self.device, dtype=torch.long)
        if position and self.graph_enabled:
            logits, hidden = self.graphs.target(ids, position)
        else:
            logits, hidden = self.model.forward_with_aux(ids, start_pos=position)
        logits = self.model.compute_logits(logits)
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError("non-finite target logits")
        self.spec_stats["target_forward_calls"] += 1
        if self.drafting_enabled and not (position and self.graph_enabled):
            self.model.store_draft_context(hidden, position)
        return int(logits[0].argmax().item()), hidden

    @staticmethod
    def request_block(block_ids):
        if len(block_ids) != 1 or len(block_ids[0]) != 1:
            raise ValueError(
                "reference profile requires one complete state block per request"
            )
        return block_ids[0][0]

    @torch.inference_mode()
    def execute_model(self, output):
        for req_id in output.finished_req_ids:
            self.requests.pop(req_id, None)
        if output.scheduled_encoder_inputs:
            raise ValueError("multimodal scheduling is not enabled in this profile")
        if output.scheduled_spec_decode_tokens and not self.drafting_enabled:
            raise ValueError("draft tokens scheduled while DSpark is disabled")
        if self.pd_enabled != (
            getattr(output, "kv_connector_metadata", None) is not None
        ):
            raise ValueError("PD connector metadata does not match the runner mode")
        if getattr(output, "kv_cache_block_copies", None):
            raise ValueError("request-state prefix copy is not enabled")
        for item in output.scheduled_new_reqs:
            validate_sampling(item.sampling_params)
            if item.mm_features or item.lora_request or item.prompt_embeds is not None:
                raise ValueError(
                    "FL reference profile accepts token-only requests without LoRA"
                )
            computed = item.num_computed_tokens
            if self.data_parallel and computed == 0:
                raise ValueError(
                    "mixed-axis Decode requires a completed remote Prefill"
                )
            if computed and (
                not self.pd_enabled or computed != len(item.prompt_token_ids) - 1
            ):
                raise ValueError("a new request cannot reuse unvalidated cached state")
            self.requests[item.req_id] = Request(
                list(item.prompt_token_ids),
                self.request_block(item.block_ids),
                computed,
            )
        cached = output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            request = self.requests[req_id]
            if req_id in cached.all_token_ids:
                request.tokens = list(cached.all_token_ids[req_id])
            blocks = cached.new_block_ids[i]
            if req_id in cached.resumed_req_ids:
                if cached.num_computed_tokens[i] != 0:
                    raise ValueError(
                        "resumed request must recompute from position zero"
                    )
                request.block = self.request_block(blocks)
            elif blocks is not None and any(blocks):
                raise ValueError("a request cannot acquire a second state block")
            request.computed = cached.num_computed_tokens[i]

        if (
            self.batched_decode_enabled
            and output.num_scheduled_tokens
            and all(self.requests[r].computed > 0 for r in output.num_scheduled_tokens)
        ):
            return self.execute_decode_batch(output)

        req_ids, samples, next_drafts = [], [], []
        for req_id, count in output.num_scheduled_tokens.items():
            request = self.requests[req_id]
            start = request.computed
            drafts = output.scheduled_spec_decode_tokens.get(req_id, [])
            self.state.bind(request.block, reset=start == 0)
            if start == 0:
                if count != len(request.tokens) or not count or drafts:
                    raise ValueError("chunked prefill is not supported")
                token, hidden = self.target_forward(request.tokens, 0)
                generated = [token]
                request.computed = count
            else:
                if count != 1 + len(drafts) or len(request.tokens) != start + 1:
                    raise ValueError(f"invalid scheduled decode span for {req_id}")
                # Verify greedily in order. Only accepted inputs are committed to
                # Engram/compressor/ring state, so a rejected suffix needs no
                # rollback. This is a correctness baseline, not parallel verify.
                generated = []
                input_token = request.tokens[start]
                for offset in range(count):
                    token, hidden = self.target_forward([input_token], start + offset)
                    generated.append(token)
                    if offset == len(drafts) or token != drafts[offset]:
                        break
                    input_token = token
                request.computed += len(generated)
                if drafts:
                    accepted = len(generated) - 1
                    self.spec_stats["verified_steps"] += 1
                    self.spec_stats["draft_tokens"] += len(drafts)
                    self.spec_stats["accepted_tokens"] += accepted
                    self.spec_stats["accepted_prefix_histogram"][accepted] += 1
            request.tokens.extend(generated)
            draft = []
            if (
                self.drafting_enabled
                and start > 0
                and request.computed + 5 <= self.model.args.max_seq_len
            ):
                last_token = torch.tensor([generated[-1]], device=self.device)
                if self.graph_enabled:
                    result = self.graphs.draft(last_token, hidden, request.computed - 1)
                else:
                    result = self.model.propose_draft(
                        last_token, hidden, request.computed - 1
                    )
                ids, logits, confidence = result
                if not (
                    bool(torch.isfinite(logits).all().item())
                    and bool(torch.isfinite(confidence).all().item())
                ):
                    raise FloatingPointError("non-finite DSpark logits/confidence")
                draft = ids[0, 1:].tolist()
                if len(draft) != 5 or int(ids[0, 0]) != generated[-1]:
                    raise ValueError(
                        "DSpark must return the bonus token and five drafts"
                    )
                self.spec_stats["draft_steps"] += 1
            req_ids.append(req_id)
            samples.append(generated)
            next_drafts.append(draft)
        self.draft_token_ids = (
            DraftTokenIds(req_ids, next_drafts) if self.drafting_enabled else None
        )
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req: i for i, req in enumerate(req_ids)},
            sampled_token_ids=samples,
        )

    def execute_decode_batch(self, output):
        """Verify request prefixes together, masking a lane after rejection.

        Target verification remains sequential across draft positions for now;
        different requests share each model forward and graph. A rejected lane
        never commits its suffix to Engram, compressor, or window state.
        """
        req_ids = list(output.num_scheduled_tokens)
        requests = [self.requests[r] for r in req_ids]
        drafts = [output.scheduled_spec_decode_tokens.get(r, []) for r in req_ids]
        for req_id, request, draft in zip(req_ids, requests, drafts):
            if (
                output.num_scheduled_tokens[req_id] != 1 + len(draft)
                or len(request.tokens) != request.computed + 1
            ):
                raise ValueError(f"invalid scheduled decode span for {req_id}")
        pages = torch.tensor(
            [r.block for r in requests], device=self.device, dtype=torch.long
        )
        starts = torch.tensor(
            [r.computed for r in requests], device=self.device, dtype=torch.long
        )
        inputs = torch.tensor(
            [r.tokens[r.computed] for r in requests],
            device=self.device,
            dtype=torch.long,
        )
        active_host = [True] * len(requests)
        active = torch.ones(len(requests), device=self.device, dtype=torch.bool)
        samples = [[] for _ in requests]
        last_hidden = None
        last_tokens = inputs.clone()
        max_steps = (
            (6 if self.drafting_enabled else 1)
            if self.data_parallel
            else 1 + max(map(len, drafts), default=0)
        )
        for offset in range(max_steps):
            if not self.any_decode_active(any(active_host)):
                break
            positions = torch.where(active, starts + offset, 0)
            logits, hidden = self.graphs.target_batch(inputs, pages, positions, active)
            if not bool(torch.isfinite(logits[active]).all().item()):
                raise FloatingPointError("non-finite batched target logits")
            selected = logits.argmax(-1)
            selected_host = selected.tolist()
            last_tokens = torch.where(active, selected, last_tokens)
            if self.drafting_enabled:
                if last_hidden is None:
                    last_hidden = torch.zeros_like(hidden)
                last_hidden = torch.where(active[:, None, None], hidden, last_hidden)
            self.spec_stats["target_forward_calls"] += sum(active_host)
            for i, enabled in enumerate(active_host):
                if not enabled:
                    continue
                token = selected_host[i]
                samples[i].append(token)
                active_host[i] = offset < len(drafts[i]) and token == drafts[i][offset]
            active.copy_(torch.tensor(active_host, device=self.device))
            inputs = selected
        next_drafts = [[] for _ in requests]
        for request, generated, draft in zip(requests, samples, drafts):
            request.computed += len(generated)
            request.tokens.extend(generated)
            if draft:
                accepted = len(generated) - 1
                self.spec_stats["verified_steps"] += 1
                self.spec_stats["draft_tokens"] += len(draft)
                self.spec_stats["accepted_tokens"] += accepted
                self.spec_stats["accepted_prefix_histogram"][accepted] += 1
        if self.drafting_enabled and last_hidden is not None:
            can_draft = [
                r.computed + 5 <= self.model.args.max_seq_len for r in requests
            ]
            if any(can_draft) or self.data_parallel:
                active.copy_(torch.tensor(can_draft, device=self.device))
                positions = torch.tensor(
                    [
                        r.computed - 1 if enabled else 0
                        for r, enabled in zip(requests, can_draft)
                    ],
                    device=self.device,
                    dtype=torch.long,
                )
                ids, logits, confidence = self.graphs.draft_batch(
                    last_tokens, last_hidden, pages, positions, active
                )
                if not (
                    bool(torch.isfinite(logits[active]).all().item())
                    and bool(torch.isfinite(confidence[active]).all().item())
                ):
                    raise FloatingPointError("non-finite batched DSpark output")
                for i, (enabled, row) in enumerate(zip(can_draft, ids.tolist())):
                    if enabled:
                        if len(row) != 6 or row[0] != samples[i][-1]:
                            raise ValueError(
                                "DSpark must return bonus plus five drafts"
                            )
                        next_drafts[i] = row[1:]
                        self.spec_stats["draft_steps"] += 1
        self.draft_token_ids = (
            DraftTokenIds(req_ids, next_drafts) if self.drafting_enabled else None
        )
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req: i for i, req in enumerate(req_ids)},
            sampled_token_ids=samples,
        )

    def any_decode_active(self, local_active):
        if not self.data_parallel:
            return local_active
        # EngineCore synchronizes DP iterations, but speculative rejection can
        # stop different groups at different positions within one iteration.
        # This control reduction keeps their EP kernel sequence aligned.
        flag = torch.tensor(int(local_active), dtype=torch.int32, device="cpu")
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())


class WorkerFL028(WorkerBase):
    def init_device(self):
        self.pd_connector = None
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Atomic FP32 Split-K changes mHC mixes between identical requests. The
        # small rounding differences amplify through BF16 residuals and routing.
        torch.use_deterministic_algorithms(True)
        parallel = self.parallel_config
        if parallel.data_parallel_size > 1:
            dp_local_rank = parallel.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = parallel.data_parallel_index
            self.local_rank += dp_local_rank * parallel.tensor_parallel_size
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        torch.manual_seed(self.model_config.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        tp_backend = requested_backend()
        world_size = parallel.world_size
        init_method = self.distributed_init_method
        if parallel.data_parallel_size > 1:
            if tp_backend != "flagcx":
                raise ValueError(
                    "mixed-axis Decode currently requires FlagCX and Gloo control"
                )
            from vllm.utils.network_utils import get_distributed_init_method

            self.rank += parallel.data_parallel_rank * world_size
            world_size = parallel.world_size_across_dp
            init_method = get_distributed_init_method(
                parallel.data_parallel_master_ip, parallel.get_next_dp_init_port()
            )
        if world_size > 1:
            pg_kwargs = dict(
                init_method=init_method,
                rank=self.rank,
                world_size=world_size,
            )
            if tp_backend == "flagcx":
                # Gloo carries the FlagCX unique ID; device tensors use FlagCX.
                dist.init_process_group("gloo", **pg_kwargs)
            else:
                dist.init_process_group("nccl", device_id=self.device, **pg_kwargs)
        init_tp_collectives(self.device, tensor_size=parallel.tensor_parallel_size)
        logger.warning("FL rank %d TP collective backend: %s", self.rank, tp_backend)

    def load_model(self, *, load_dummy_weights=False):
        if load_dummy_weights:
            raise ValueError("real-checkpoint admission does not accept dummy weights")
        begin = time.monotonic()
        with torch.cuda.device(self.device):
            model = FLDeepseekV41Loader(self.load_config).load_model(
                self.vllm_config, self.model_config
            )
        state = RequestState(model.core, self.model_config.max_model_len)
        self.model_runner = ModelRunnerFL028(
            model,
            state,
            self.device,
            pd_enabled=self.vllm_config.kv_transfer_config is not None,
        )
        logger.warning(
            "FL rank %d loaded in %.1fs: %s; execution=%s",
            self.rank,
            time.monotonic() - begin,
            json.dumps(model.load_manifest),
            json.dumps(model.profile),
        )

    def get_model(self):
        return self.model_runner.model

    def get_supported_tasks(self):
        return ("generate",)

    def update_max_model_len(self, max_model_len):
        if max_model_len != self.model_config.max_model_len:
            raise ValueError("request-state layout is fixed at model construction")

    def get_kv_connector_handshake_metadata(self):
        return None

    def take_draft_token_ids(self):
        return self.model_runner.take_draft_token_ids()

    def execute_dummy_batch(self):
        if self.model_runner.data_parallel:
            self.model_runner.execute_decode_batch(
                SimpleNamespace(
                    num_scheduled_tokens={}, scheduled_spec_decode_tokens={}
                )
            )
        return None

    def reset_encoder_cache(self):
        return None

    def get_kv_cache_spec(self):
        return {"fl_request_state": self.model_runner.state.spec}

    def determine_available_memory(self):
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        # The serial reference path reserves 2 GiB for activations/collectives.
        # The exact resident state is separately charged through its cache spec.
        available = int(
            free - total * (1 - self.cache_config.gpu_memory_utilization) - (2 << 30)
        )
        if available <= 0:
            raise MemoryError(
                "insufficient device memory after real weights and workspace reserve"
            )
        return available

    def initialize_from_config(self, kv_cache_config):
        self.model_runner.state.allocate(kv_cache_config, self.device)
        if self.model_runner.graph_enabled:
            if self.model_runner.batched_decode_enabled:
                from .batched_graph import BatchedDecodeGraphs

                self.model_runner.graphs = BatchedDecodeGraphs(
                    self.get_model(),
                    self.model_runner.state,
                    self.device,
                    batch_capacity=self.scheduler_config.max_num_seqs
                    if self.model_runner.data_parallel
                    else None,
                )
            else:
                from .decode_graph import DecodeGraphs

                self.model_runner.state.allocate_graph_scratch()
                self.model_runner.graphs = DecodeGraphs(
                    self.get_model(), self.model_runner.state, self.device
                )
        if self.vllm_config.kv_transfer_config is not None:
            # The strict FL Worker owns its TP collectives. vLLM's global KV
            # initializer assumes its own model-parallel TP group exists.
            self.pd_connector = DeepseekV41FLConnector(
                self.vllm_config, KVConnectorRole.WORKER, kv_cache_config
            )
            self.pd_connector.register_kv_caches(
                {"fl_request_state": self.model_runner.state.storage}
            )
        torch.cuda.empty_cache()

    def compile_or_warm_up_model(self):
        # Use the official startup phase, whose RPC is not limited by the
        # request execution timeout. Cold FlagGems autotuning can take minutes.
        begin = time.monotonic()
        logger.warning("FL rank %d starting Eager kernel warmup", self.rank)
        state = self.model_runner.state
        state.bind(0, reset=True)
        limit = self.model_config.max_model_len
        count = min(16, max(1, limit - 1))
        ids = torch.full(
            (count,),
            self.model_config.hf_config.bos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            _, hidden = self.get_model().forward_with_aux(ids, start_pos=0)
            if self.model_runner.drafting_enabled:
                self.get_model().store_draft_context(hidden, 0)
            if count < limit:
                logits, hidden = self.get_model().forward_with_aux(
                    ids[:1], start_pos=count
                )
                if self.model_runner.drafting_enabled and count + 6 <= limit:
                    self.get_model().propose_draft(logits.argmax(-1), hidden, count)
        torch.cuda.synchronize(self.device)
        state.bind(0, reset=True)
        if self.model_runner.data_parallel:
            # Capture one bounded bucket before the API admits PD transfers.
            # No live request state is present; all lanes are masked off.
            count = self.scheduler_config.max_num_seqs
            tokens = torch.zeros(count, device=self.device, dtype=torch.long)
            active = torch.zeros(count, device=self.device, dtype=torch.bool)
            with torch.inference_mode():
                _, hidden = self.model_runner.graphs.target_batch(
                    tokens, tokens, tokens, active
                )
                if self.model_runner.drafting_enabled:
                    self.model_runner.graphs.draft_batch(
                        tokens, hidden, tokens, tokens, active
                    )
            torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - begin
        logger.warning("FL rank %d Eager warmup finished in %.1fs", self.rank, elapsed)
        return CompilationTimes(elapsed, 0.0)

    def get_cache_block_size_bytes(self):
        return self.model_runner.state.page_bytes

    def execute_model(self, scheduler_output):
        connector = self.pd_connector
        if connector is None:
            return self.model_runner.execute_model(scheduler_output)
        connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
        try:
            connector.start_load_kv(None)
            output = self.model_runner.execute_model(scheduler_output)
            connector.wait_for_save()
            sent, received = connector.get_finished(scheduler_output.finished_req_ids)
            output.kv_connector_output = KVConnectorOutput(
                finished_sending=sent,
                finished_recving=received,
            )
            return output
        finally:
            connector.clear_connector_metadata()

    def sample_tokens(self, grammar_output):
        raise RuntimeError("sampling is synchronous in execute_model for this profile")

    def check_health(self):
        torch.cuda.synchronize(self.device)

    def shutdown(self):
        if self.pd_connector is not None:
            self.pd_connector.shutdown()
            self.pd_connector = None
        if dist.is_initialized():
            dist.destroy_process_group()
        self.model_runner = None
