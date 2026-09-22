#!/usr/bin/env bash
# Run with the NVIDIA Container Toolkit on the validated homogeneous 8-GPU host.
# Set FL_MODEL_PATH and FL_RUNTIME_IMAGE explicitly; no checkpoint is downloaded.
set -euo pipefail
model_path="${FL_MODEL_PATH:?set the read-only DeepSeek-V4.1-Flash checkpoint directory}"
runtime_image="${FL_RUNTIME_IMAGE:?set the image ID from the delivery image receipt}"
test -f "$model_path/config.json"
exec docker run --rm --name "${FL_CONTAINER_NAME:-dsv41-fl-serving}" \
  --gpus all --ipc=host \
  -p "127.0.0.1:${FL_API_PORT:-8000}:8000" \
  -v "$model_path:/models/DeepSeek-V4.1-Flash:ro" \
  "$runtime_image" /models/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash-fl \
  --tokenizer-mode fl_deepseek_v41 \
  --hf-overrides '{"architectures":["DeepseekV41FlashFLForCausalLM"]}' \
  --load-format fl_dsv41 --dtype bfloat16 \
  --generation-config vllm --override-generation-config '{"temperature":0}' \
  --tensor-parallel-size 8 --distributed-executor-backend mp \
  --max-model-len 256 --max-num-seqs 2 --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.95 --enforce-eager \
  --no-enable-prefix-caching --no-enable-chunked-prefill --no-async-scheduling
