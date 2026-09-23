#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start one FL vLLM 0.28 PD role with TP=8 and DSpark on an H100 host.
set -euo pipefail

role="${FL_PD_ROLE:?set FL_PD_ROLE to kv_producer or kv_consumer}"
host_ip="${FL_PD_HOST_IP:?set the routable host IP}"
model_path="${FL_MODEL_PATH:?set the checkpoint path}"
image="${FL_RUNTIME_IMAGE:?set the pinned runtime image}"
port="${FL_PD_API_PORT:?set the API port}"
max_model_len="${FL_MAX_MODEL_LEN:-256}"
max_num_seqs="${FL_MAX_NUM_SEQS:-2}"
max_num_batched_tokens="${FL_MAX_NUM_BATCHED_TOKENS:-256}"
case "$role" in
  kv_producer|kv_consumer) ;;
  *) echo "invalid FL_PD_ROLE: $role" >&2; exit 2 ;;
esac
test -f "$model_path/config.json"

docker_args=(
  --rm --name "${FL_CONTAINER_NAME:-dsv41-fl-${role}}"
  --gpus all --ipc=host --network=host
  --device=/dev/infiniband --ulimit memlock=-1:-1
  --cap-add IPC_LOCK --security-opt seccomp=unconfined
  -e "VLLM_HOST_IP=$host_ip"
  -e FLAGCX_SOCKET_IFNAME=bond0
  -e FLAGCX_BOOTSTRAP_PORT=18998
  -e FLAGCX_LIB_PATH=/opt/flagcx/build/lib/libflagcx.so
  -e FL_PD_TRANSFER_TIMEOUT_S=240
  -e OPENBLAS_NUM_THREADS=1
  -e OMP_NUM_THREADS=1
  -v "$model_path:/models/DeepSeek-V4.1-Flash:ro"
)
if [[ "${VLLM_FL_EXPERIMENTAL_LONG_CONTEXT:-0}" == 1 ]]; then
  docker_args+=(-e VLLM_FL_EXPERIMENTAL_LONG_CONTEXT=1)
fi

# The Docker daemon on 10.8.2.68 records --gpus all without injecting devices.
# Explicit mappings mirror the verified GPU preflight on that host.
if [[ "${FL_PD_EXPLICIT_DEVICES:-0}" == 1 ]]; then
  for node in /dev/nvidia{0..7} /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    test -e "$node"
    docker_args+=(--device="$node")
  done
  for node in /dev/nvidia-caps/*; do
    docker_args+=(--device="$node")
  done
  for lib in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
    test -f "/lib/x86_64-linux-gnu/$lib"
    docker_args+=(-v "/lib/x86_64-linux-gnu/$lib:/driver/$lib:ro")
  done
  docker_args+=(-e LD_LIBRARY_PATH=/driver:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda/lib64)
fi

server_args=()
if [[ "${FL_VALIDATION_RPC:-0}" == 1 ]]; then
  docker_args+=(-e VLLM_SERVER_DEV_MODE=1)
  server_args+=(--worker-extension-cls vllm_fl.strict028.validation.ReferenceProbeExtension)
fi

kv_config="$(python3 - "$role" <<'PY'
import json, sys
role = sys.argv[1]
print(json.dumps({
    "kv_connector": "DeepseekV41FLConnector",
    "kv_connector_module_path": "vllm_fl.strict028.pd_connector",
    "kv_role": role,
    "engine_id": "dsv41-fl-" + role,
    "kv_load_failure_policy": "fail",
}, separators=(",", ":")))
PY
)"

exec docker run "${docker_args[@]}" "$image" /models/DeepSeek-V4.1-Flash \
  --host 0.0.0.0 --port "$port" \
  --served-model-name deepseek-v4.1-flash-fl \
  --tokenizer-mode fl_deepseek_v41 \
  --hf-overrides '{"architectures":["DeepseekV41FlashFLForCausalLM"]}' \
  --load-format fl_dsv41 --dtype bfloat16 \
  --generation-config vllm --override-generation-config '{"temperature":0}' \
  --tensor-parallel-size 8 --distributed-executor-backend mp \
  --max-model-len "$max_model_len" --max-num-seqs "$max_num_seqs" \
  --max-num-batched-tokens "$max_num_batched_tokens" \
  --gpu-memory-utilization 0.95 --enforce-eager \
  --no-enable-prefix-caching --no-enable-chunked-prefill --no-async-scheduling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --kv-transfer-config "$kv_config" \
  "${server_args[@]}" "$@"
