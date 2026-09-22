# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.28 Worker and sequential Eager Runner for the FL reference graph."""

import json
import logging
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from .cache import RequestState
from .model_loader import FLDeepseekV41Loader
from .sampling import validate_sampling

logger = logging.getLogger(__name__)


@dataclass
class Request:
    tokens: list[int]
    block: int
    computed: int


class ModelRunnerFL028:
    def __init__(self, model, state, device):
        self.model = model
        self.state = state
        self.device = device
        self.requests = {}

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
        if output.scheduled_spec_decode_tokens or output.scheduled_encoder_inputs:
            raise ValueError(
                "speculative or multimodal scheduling is not enabled in this profile"
            )
        if getattr(output, "kv_connector_metadata", None) is not None:
            raise ValueError("PD is not enabled in this profile")
        if getattr(output, "kv_cache_block_copies", None):
            raise ValueError("request-state prefix copy is not enabled")
        for item in output.scheduled_new_reqs:
            validate_sampling(item.sampling_params)
            if item.mm_features or item.lora_request or item.prompt_embeds is not None:
                raise ValueError(
                    "FL reference profile accepts token-only requests without LoRA"
                )
            if item.num_computed_tokens:
                raise ValueError("a new request cannot reuse unvalidated cached state")
            self.requests[item.req_id] = Request(
                list(item.prompt_token_ids), self.request_block(item.block_ids), 0
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

        req_ids, samples = [], []
        for req_id, count in output.num_scheduled_tokens.items():
            request = self.requests[req_id]
            start = request.computed
            tokens = request.tokens[start : start + count]
            if len(tokens) != count or not count:
                raise ValueError(f"invalid scheduled token span for {req_id}")
            if (start == 0 and count != len(request.tokens)) or (
                start > 0 and count != 1
            ):
                raise ValueError(
                    "chunked prefill is not supported by this reference profile"
                )
            self.state.bind(request.block, reset=start == 0)
            ids = torch.tensor(tokens, device=self.device, dtype=torch.long)
            logits = self.model.compute_logits(self.model(ids, start_pos=start))
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError(f"non-finite logits for {req_id}")
            token = int(logits[0].argmax().item())
            request.tokens.append(token)
            request.computed += count
            req_ids.append(req_id)
            samples.append([token])
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req: i for i, req in enumerate(req_ids)},
            sampled_token_ids=samples,
        )


class WorkerFL028(WorkerBase):
    def init_device(self):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Atomic FP32 Split-K changes mHC mixes between identical requests. The
        # small rounding differences amplify through BF16 residuals and routing.
        torch.use_deterministic_algorithms(True)
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        torch.manual_seed(self.model_config.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        if self.parallel_config.world_size > 1:
            dist.init_process_group(
                "nccl",
                init_method=self.distributed_init_method,
                rank=self.rank,
                world_size=self.parallel_config.world_size,
                device_id=self.device,
            )

    def load_model(self, *, load_dummy_weights=False):
        if load_dummy_weights:
            raise ValueError("real-checkpoint admission does not accept dummy weights")
        begin = time.monotonic()
        with torch.cuda.device(self.device):
            model = FLDeepseekV41Loader(self.load_config).load_model(
                self.vllm_config, self.model_config
            )
        state = RequestState(model.core, self.model_config.max_model_len)
        self.model_runner = ModelRunnerFL028(model, state, self.device)
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
        return None

    def execute_dummy_batch(self):
        # DP is disabled and there are no unmatched rank-local collectives.
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
            self.get_model()(ids, start_pos=0)
            if count < limit:
                self.get_model()(ids[:1], start_pos=count)
        torch.cuda.synchronize(self.device)
        state.bind(0, reset=True)
        elapsed = time.monotonic() - begin
        logger.warning("FL rank %d Eager warmup finished in %.1fs", self.rank, elapsed)
        return CompilationTimes(elapsed, 0.0)

    def get_cache_block_size_bytes(self):
        return self.model_runner.state.page_bytes

    def execute_model(self, scheduler_output):
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output):
        raise RuntimeError("sampling is synchronous in execute_model for this profile")

    def check_health(self):
        torch.cuda.synchronize(self.device)

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        self.model_runner = None
