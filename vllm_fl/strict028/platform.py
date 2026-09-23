# SPDX-License-Identifier: Apache-2.0
"""Platform for the CUDA-hosted empty-build admission check.

Deriving from CudaPlatform would import vLLM's compiled extension. Keep the
out-of-tree identity so CUDA kernel selection cannot silently bypass FL.
"""

import os
from contextlib import contextmanager

import torch

from vllm.platforms import Platform, PlatformEnum
from vllm.platforms.interface import DeviceCapability


@contextmanager
def _nvml_device(device_id):
    from vllm.utils.import_utils import import_pynvml

    nvml = import_pynvml()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = (
        visible.split(",")[device_id].strip() if visible is not None else str(device_id)
    )
    nvml.nvmlInit()
    try:
        handle = (
            nvml.nvmlDeviceGetHandleByIndex(int(physical))
            if physical.isdecimal()
            else nvml.nvmlDeviceGetHandleByUUID(physical)
        )
        yield nvml, handle
    finally:
        nvml.nvmlShutdown()


class PlatformFL028(Platform):
    _enum = PlatformEnum.OOT
    device_name = "cuda"
    device_type = "cuda"
    dispatch_key = "CUDA"
    dist_backend = (
        "gloo"
        if os.environ.get("VLLM_FL_TP_BACKEND", "nccl").lower() == "flagcx"
        else "nccl"
    )
    ray_device_key = "GPU"
    device_control_env_var = "CUDA_VISIBLE_DEVICES"
    torch_device_fn = torch.cuda

    @classmethod
    def validate_request(cls, processed_inputs, params):
        from vllm.exceptions import VLLMValidationError
        from vllm.sampling_params import SamplingParams

        from .sampling import validate_sampling

        if processed_inputs.get("type") != "token" or not isinstance(
            params, SamplingParams
        ):
            raise VLLMValidationError(
                "FL reference profile accepts text token generation only"
            )
        try:
            validate_sampling(params)
        except ValueError as error:
            # AsyncLLM only propagates VLLMClientError as a request failure;
            # raw ValueError becomes EngineGenerateError and an HTTP 500.
            raise VLLMValidationError(str(error)) from error

    @classmethod
    def register_custom_kv_cache_specs(cls, vllm_config):
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

        from .cache import FLRequestStateSpec

        # Called by the host after built-in specs have been registered.
        KVCacheSpecRegistry.register(FLRequestStateSpec, FullAttentionManager)

    @classmethod
    def check_and_update_config(cls, vllm_config):
        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        scheduler = vllm_config.scheduler_config
        cache = vllm_config.cache_config
        if model.hf_config.architectures != ["DeepseekV41FlashFLForCausalLM"]:
            raise ValueError(
                "strict028 requires the independently registered DeepseekV41FlashFLForCausalLM architecture"
            )
        if model.dtype != torch.bfloat16 or not model.enforce_eager:
            raise ValueError(
                "the FL reference profile requires dtype=bfloat16 and enforce_eager=True"
            )
        if model.max_model_len > 4096 and (
            os.environ.get("VLLM_FL_EXPERIMENTAL_LONG_CONTEXT") != "1"
            or model.max_model_len > 33792
        ):
            raise ValueError(
                "FL long-context admission requires "
                "VLLM_FL_EXPERIMENTAL_LONG_CONTEXT=1 and max_model_len<=33792"
            )
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
        ):
            raise ValueError(
                "the reference profile currently supports homogeneous TP only"
            )
        if parallel.tensor_parallel_size not in (1, 2, 4, 8):
            raise ValueError("TP must divide the checkpoint's eight output groups")
        if parallel.nnodes_within_dp != 1:
            raise ValueError(
                "multi-node execution requires separate communication acceptance"
            )
        if vllm_config.lora_config:
            raise ValueError("LoRA requires separate integration acceptance")
        kv_transfer = vllm_config.kv_transfer_config
        if kv_transfer is not None:
            if (
                kv_transfer.kv_connector != "DeepseekV41FLConnector"
                or kv_transfer.kv_connector_module_path
                != "vllm_fl.strict028.pd_connector"
                or kv_transfer.kv_role not in ("kv_producer", "kv_consumer")
            ):
                raise ValueError(
                    "FL PD requires the DeepseekV41FLConnector external module "
                    "and a producer or consumer role"
                )
            if os.environ.get("VLLM_FL_TP_BACKEND", "nccl").lower() != "flagcx":
                raise ValueError("FL PD requires VLLM_FL_TP_BACKEND=flagcx")
        spec = vllm_config.speculative_config
        if spec is not None:
            if spec.method != "dspark" or spec.num_speculative_tokens != 5:
                raise ValueError(
                    "FL V4.1 MTP requires method=dspark and 5 draft tokens"
                )
            if (
                spec.enable_adaptive_verification
                or spec.dspark_draft_topk is not None
                or spec.draft_sample_method != "greedy"
                or spec.num_speculative_tokens_per_batch_size is not None
            ):
                raise ValueError("FL DSpark uses fixed-length greedy verification")
            if (
                spec.model != model.model
                or spec.draft_parallel_config.tensor_parallel_size
                != parallel.tensor_parallel_size
            ):
                raise ValueError("FL DSpark shares the target checkpoint and TP ranks")
        if scheduler.enable_chunked_prefill or scheduler.async_scheduling:
            raise ValueError(
                "disable chunked prefill and async scheduling for FL Eager reference"
            )
        if cache.enable_prefix_caching:
            raise ValueError(
                "disable prefix caching for the complete request-state layout"
            )
        if cache.cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("reference KV uses rounded BF16 storage")
        if vllm_config.load_config.load_format != "fl_dsv41":
            raise ValueError(
                "original low-precision weights require load_format=fl_dsv41"
            )
        parallel.worker_cls = "vllm_fl.strict028.worker.WorkerFL028"
        parallel.disable_custom_all_reduce = True
        cache.block_size = model.max_model_len
        if cache.num_gpu_blocks_override is None:
            # Plus the scheduler's reserved null block.
            cache.num_gpu_blocks_override = scheduler.max_num_seqs + 1

    @classmethod
    def update_block_size_for_backend(cls, vllm_config):
        # The custom state spec contains a complete sequence, not kernel KV tiles.
        return None

    def is_cuda_alike(self) -> bool:
        return True

    @classmethod
    def import_kernels(cls) -> None:
        # All kernels in this mode are explicitly imported from FlagGems.
        # The empty host intentionally provides no vLLM device extensions.
        return None

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        # Host imports ask about capability before a worker is bound. NVML does
        # not initialize a CUDA context on the parent's/default device.
        with _nvml_device(device_id) as (nvml, handle):
            return DeviceCapability(*nvml.nvmlDeviceGetCudaComputeCapability(handle))

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        with _nvml_device(device_id) as (nvml, handle):
            return nvml.nvmlDeviceGetName(handle)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        with _nvml_device(device_id) as (nvml, handle):
            return nvml.nvmlDeviceGetUUID(handle)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        with _nvml_device(device_id) as (nvml, handle):
            return int(nvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        torch.cuda.set_device(device)

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(f"strict028 has not validated compute dtype {dtype}")
        if dtype == torch.bfloat16 and cls.get_device_capability().major < 8:
            raise ValueError("BF16 compute is unavailable on this device")
