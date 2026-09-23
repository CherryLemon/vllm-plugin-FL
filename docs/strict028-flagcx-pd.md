# DeepSeek V4.1 FL / vLLM 0.28 FlagCX PD admission

This profile transfers one complete `fl_request_state` page per request and TP
rank through the external `DeepseekV41FLConnector`. The page contains the
window and compressed attention caches, Indexer state, incomplete compressor
state, Engram history, and other mutable buffers recorded by `RequestState`.
The connector validates the page layout hash, byte size, TP rank, and model /
tokenizer metadata before handoff. It does not patch official vLLM.

On the current H100 nodes, FlagCX GPU memory registration fails with
`ibv_reg_mr_iova2: Bad address`. The connector therefore registers pinned host
pages with FlagCX and copies GPU → host → RDMA → host → GPU. Both hosts need
`/dev/infiniband`, unlimited memlock, and `FLAGCX_SOCKET_IFNAME=bond0`.
The `10.8.2.68` Docker daemon also needs explicit NVIDIA device and driver
mounts; `tools/strict028/serve_pd.sh` provides these when
`FL_PD_EXPLICIT_DEVICES=1`.

Run Prefill on `10.8.2.13` and Decode on `10.8.2.68`, each with the same pinned
runtime image and read-only checkpoint:

```bash
FL_PD_ROLE=kv_producer FL_PD_HOST_IP=10.8.2.13 FL_PD_API_PORT=18031 \
  FL_MODEL_PATH=/public-nvme/models/DeepSeek-V4.1-Flash \
  FL_RUNTIME_IMAGE=<pinned-image> FL_CONTAINER_NAME=dsv41-fl-pd-p13 \
  FL_VALIDATION_RPC=1 bash tools/strict028/serve_pd.sh

FL_PD_ROLE=kv_consumer FL_PD_HOST_IP=10.8.2.68 FL_PD_API_PORT=18032 \
  FL_MODEL_PATH=/public-nvme/models/DeepSeek-V4.1-Flash \
  FL_RUNTIME_IMAGE=<pinned-image> FL_CONTAINER_NAME=dsv41-fl-pd-d68 \
  FL_PD_EXPLICIT_DEVICES=1 FL_VALIDATION_RPC=1 \
  bash tools/strict028/serve_pd.sh
```

The test client first asks Prefill for one greedy output token with
`do_remote_decode=true` and a unique `transfer_id`. Prefill's response contains
the connector's `kv_transfer_params`. Decode receives the original prompt plus
that token with `do_remote_prefill=true`, skipping exactly the original prompt
positions already represented by the transferred state. The client stitches
Prefill's first output token with Decode's output tokens. If Prefill emits EOS,
the client returns immediately without a Decode request.

This is an Eager correctness profile: TP=8, max context 256, at most two live
sequences, DSpark five-token greedy MTP, no prefix caching, chunked prefill,
async scheduling, cross-vendor PD, or performance claim. `fl_pd_stats` reports
per-rank page counts and bytes for admission. A production router, load testing,
failure injection, and longer-context validation remain separate milestones.
