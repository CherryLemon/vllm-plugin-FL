# SPDX-License-Identifier: Apache-2.0
"""FlagCX PD transfer of the complete FL request-state page.

The H100 nodes in this admission cannot register CUDA allocations with the
RDMA device, so each worker registers pinned host pages with FlagCX. The page
is copied GPU -> host -> FlagCX -> host -> GPU. No KV-only layout is assumed.
"""

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.utils.network_utils import get_ip
from vllm.v1.request import RequestStatus

from .cache import FLRequestStateSpec
from .collectives import parallel_layout
from vllm_fl.distributed.kv_transfer.flagcx_connector import (
    FlagCXConnectorMetadata,
    FlagCXConnectorScheduler,
    FLAGCXLibrary,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = logging.getLogger(__name__)


def _state_spec(kv_cache_config: "KVCacheConfig") -> FLRequestStateSpec:
    groups = kv_cache_config.kv_cache_groups
    if len(groups) != 1 or groups[0].layer_names != ["fl_request_state"]:
        raise ValueError("FL PD requires one complete request-state group")
    spec = groups[0].kv_cache_spec
    if not isinstance(spec, FLRequestStateSpec) or not spec.state_layout_hash:
        raise ValueError("FL PD requires a versioned FLRequestStateSpec")
    return spec


def _one_block(groups: list[list[int]]) -> int:
    if len(groups) != 1 or len(groups[0]) != 1:
        raise ValueError(f"FL PD requires one state page: {groups}")
    return groups[0][0]


def source_plan(global_rank, tensor_size, world_size, remote_tp_size):
    """One source copy per D worker; its TP lane owns unused P copies.

    For TP2 x DP4, D ranks [4,5] pull P copies [4,5], then release [0,2,6]
    and [1,3,7], respectively. Every P copy receives exactly one completion.
    """
    if (
        tensor_size < 1
        or world_size % tensor_size
        or remote_tp_size != world_size
        or not 0 <= global_rank < world_size
    ):
        raise ValueError("PD requires matching global worker counts and divisible TP")
    return global_rank, [
        rank
        for rank in range(global_rank % tensor_size, remote_tp_size, tensor_size)
        if rank != global_rank
    ]


def _model_signature(model_path: str) -> str:
    digest = hashlib.sha256()
    root = Path(model_path)
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ):
        path = root / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class DeepseekV41Scheduler(FlagCXConnectorScheduler):
    """Reuse vLLM's FlagCX request lifetime, with full-state token semantics."""

    def __init__(self, vllm_config, engine_id, kv_cache_config):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self.state_spec = _state_spec(kv_cache_config)
        self.model_signature = _model_signature(vllm_config.model_config.model)
        self.data_parallel = vllm_config.parallel_config.data_parallel_size > 1

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params
        if not params or not params.get("do_remote_prefill"):
            return 0, False
        if self.kv_role != "kv_consumer":
            raise ValueError("only the Decode role may load remote state")
        if (
            getattr(self, "data_parallel", False)
            and params.get("state_transfer_protocol") != 2
        ):
            raise ValueError(
                "mixed-axis PD requires a producer with release protocol v2"
            )
        prompt_len = request.num_prompt_tokens
        count = int(params["num_external_tokens"])
        if not 0 < count < prompt_len:
            raise ValueError("Decode prompt must include the Prefill output token")
        if (
            params.get("state_layout_hash") != self.state_spec.state_layout_hash
            or params.get("state_page_bytes") != self.state_spec.state_page_bytes
            or params.get("state_layout_version") != self.state_spec.layout_version
            or params.get("model_signature") != self.model_signature
        ):
            raise ValueError("Prefill and Decode request-state layouts differ")
        additional = count - num_computed_tokens
        return (additional, True) if additional > 0 else (0, False)

    def request_finished(self, request, block_ids):
        delay_free, params = super().request_finished(request, block_ids)
        if params is not None:
            if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
                raise ValueError("Prefill handoff requires a one-token length cap")
            params.update(
                num_external_tokens=request.num_prompt_tokens,
                state_layout_hash=self.state_spec.state_layout_hash,
                state_layout_version=self.state_spec.layout_version,
                state_page_bytes=self.state_spec.state_page_bytes,
                model_id=str(self.vllm_config.model_config.model),
                model_signature=self.model_signature,
                state_transfer_protocol=2,
            )
        return delay_free, params

    def request_finished_all_groups(self, request, block_ids):
        if len(block_ids) != 1:
            raise ValueError("FL PD requires one state group")
        return self.request_finished(request, block_ids)


class FullStateFlagCXWorker:
    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        self.spec = _state_spec(kv_cache_config)
        layout = parallel_layout()
        self.rank = layout.global_rank
        self.tp_size = layout.tensor_size
        self.world_size = layout.world_size
        self.model_signature = _model_signature(vllm_config.model_config.model)
        self.role = vllm_config.kv_transfer_config.kv_role
        if self.role not in ("kv_producer", "kv_consumer"):
            raise ValueError("FL PD supports one Prefill and one Decode engine")
        # Retain completed Prefills independently of scarce GPU request pages.
        # The scheduler may recycle a page only after its host copy completes.
        self.host_snapshot_capacity = int(os.environ.get("FL_PD_HOST_SNAPSHOTS", "0"))
        if self.host_snapshot_capacity < 0 or (
            self.host_snapshot_capacity and self.role != "kv_producer"
        ):
            raise ValueError("host snapshot retention is a producer-only capacity")
        self._free_host_slots = list(range(self.host_snapshot_capacity))
        self.offloaded_pages = 0
        self.host = get_ip()
        self.side_port = int(os.environ.get("FLAGCX_BOOTSTRAP_PORT", "8998"))
        self.timeout = int(os.environ.get("FL_PD_TRANSFER_TIMEOUT_S", "240"))
        lib_path = os.environ.get(
            "FLAGCX_LIB_PATH",
            os.path.join(os.environ["FLAGCX_PATH"], "build/lib/libflagcx.so"),
        )
        self.flagcx = FLAGCXLibrary(lib_path)
        self.engine = self.flagcx.flagcxP2pEngineCreate()
        self.rpc_port = 0
        self.storage: torch.Tensor | None = None
        self.staging: torch.Tensor | None = None
        self.device: torch.device | None = None
        self._context = zmq.Context()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._condition = threading.Condition()
        self._pending_send: dict[str, tuple[str, int]] = {}
        self._sent_ids: set[str] = set()
        self._received_ids: set[str] = set()
        self._seen_transfer_ids: set[str] = set()
        self._fatal_error: str | None = None
        self.sent_pages = 0
        self.received_pages = 0
        self.released_pages = 0
        self.sent_bytes = 0
        self.received_bytes = 0
        self._server_thread: threading.Thread | None = None
        self._receiver_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="fl-pd-recv"
        )
        self._connections: dict[str, Any] = {}

    def register(self, caches: dict[str, torch.Tensor]):
        if set(caches) != {"fl_request_state"}:
            raise ValueError("FL PD must register the complete request-state tensor")
        storage = caches["fl_request_state"]
        if (
            storage.dtype != torch.uint8
            or storage.ndim != 2
            or storage.shape[1] != self.spec.state_page_bytes
            or not storage.is_contiguous()
            or storage.device.type != "cuda"
        ):
            raise ValueError("FL PD state storage differs from the declared layout")
        self.storage = storage
        self.device = storage.device
        shape = (
            (self.host_snapshot_capacity, storage.shape[1])
            if self.host_snapshot_capacity
            else storage.shape
        )
        self.staging = torch.empty(
            shape, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self.flagcx.flagcxP2pRegisterHost(
            self.engine, self.staging.data_ptr(), self.staging.numel()
        )
        self.rpc_port = self.flagcx.flagcxP2pGetRpcPort(self.engine)
        if self.role == "kv_consumer":
            self.flagcx.flagcxP2pStartRpcServer(self.engine)
        else:
            self._server_thread = threading.Thread(
                target=self._serve, name=f"fl-pd-send-{self.rank}", daemon=True
            )
            self._server_thread.start()
            if not self._ready.wait(15):
                raise RuntimeError("FL PD side channel failed to bind")
        logger.warning(
            "FL PD rank %d %s: %d pinned state pages, %d bytes each; side=%s:%d, rpc=%d",
            self.rank,
            self.role,
            self.staging.shape[0],
            self.spec.state_page_bytes,
            self.host,
            self.side_port + self.rank,
            self.rpc_port,
        )

    def _fail(self, error: Exception):
        logger.exception("FL PD rank %d transfer failed: %s", self.rank, error)
        with self._condition:
            self._fatal_error = str(error)
            self._condition.notify_all()

    def _serve(self):
        assert self.device is not None
        torch.cuda.set_device(self.device)
        socket = self._context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(f"tcp://{self.host}:{self.side_port + self.rank}")
            self._ready.set()
            while not self._stop.is_set():
                if not socket.poll(100):
                    continue
                request = json.loads(socket.recv())
                try:
                    self._send_one(request)
                    reply = {"status": "done"}
                except Exception as error:
                    self._fail(error)
                    reply = {"status": "error", "message": str(error)}
                socket.send_json(reply)
        except Exception as error:
            self._fail(error)
            self._ready.set()
        finally:
            socket.close()

    def _send_one(self, request: dict):
        assert self.storage is not None and self.staging is not None
        if (
            request["layout_hash"] != self.spec.state_layout_hash
            or request["layout_version"] != self.spec.layout_version
            or request["page_bytes"] != self.spec.state_page_bytes
            or request["tp_size"] != self.tp_size
            or request["rank"] != self.rank
            or request.get("model_signature") != self.model_signature
        ):
            raise ValueError("PD transfer layout or TP rank mismatch")
        action = request.get("action", "transfer")
        if action not in ("transfer", "release"):
            raise ValueError("unsupported PD state action")
        transfer_id = request["transfer_id"]
        deadline = time.monotonic() + self.timeout
        with self._condition:
            if transfer_id in self._seen_transfer_ids:
                raise ValueError("duplicate PD transfer ID")
            while transfer_id not in self._pending_send and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Prefill state not ready for {transfer_id}")
                self._condition.wait(remaining)
            if transfer_id in self._seen_transfer_ids:
                raise ValueError("duplicate PD transfer ID")
            p_req_id, block = self._pending_send[transfer_id]
            self._seen_transfer_ids.add(transfer_id)
        if action == "release":
            # The D owner sends this only after installing its pulled copy.
            # Official KVOutputAggregator waits for all P workers, including
            # both transferred and released copies, before reusing the page.
            with self._condition:
                self._pending_send.pop(transfer_id)
                self._complete_send(p_req_id, block)
                self.released_pages += 1
            return
        if not self.host_snapshot_capacity:
            self.staging[block].copy_(self.storage[block], non_blocking=False)
            torch.cuda.synchronize(self.device)
        session = f"{request['host']}:{request['rpc_port']}"
        conn = self._connections.get(session)
        if conn is None:
            conn = self.flagcx.flagcxP2pGetConn(self.engine, session)
            self._connections[session] = conn
        self.flagcx.flagcxP2pBatchWriteSync(
            conn,
            [self.staging[block].data_ptr()],
            [int(request["dst_addr"])],
            [self.spec.state_page_bytes],
        )
        with self._condition:
            self._pending_send.pop(transfer_id, None)
            self._complete_send(p_req_id, block)
            self.sent_pages += 1
            self.sent_bytes += self.spec.state_page_bytes

    def _complete_send(self, req_id, slot):
        if self.host_snapshot_capacity:
            self._free_host_slots.append(slot)
        else:
            self._sent_ids.add(req_id)

    def _receive_one(self, req_id: str, meta):
        assert self.storage is not None and self.staging is not None
        assert self.device is not None
        torch.cuda.set_device(self.device)

        def exchange(source_rank, action, block):
            socket = self._context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, self.timeout * 1000)
            socket.setsockopt(zmq.RCVTIMEO, self.timeout * 1000)
            try:
                socket.connect(
                    f"tcp://{meta.remote_host}:{meta.remote_port + source_rank}"
                )
                socket.send_json(
                    {
                        "action": action,
                        "transfer_id": meta.transfer_id,
                        "host": self.host,
                        "rpc_port": self.rpc_port,
                        "dst_addr": self.staging[block].data_ptr(),
                        "rank": source_rank,
                        "tp_size": meta.remote_tp_size,
                        "layout_hash": self.spec.state_layout_hash,
                        "layout_version": self.spec.layout_version,
                        "page_bytes": self.spec.state_page_bytes,
                        "model_signature": self.model_signature,
                    }
                )
                reply = socket.recv_json()
                if reply.get("status") != "done":
                    raise RuntimeError(f"Prefill {action} failed: {reply}")
            finally:
                socket.close()

        try:
            source, releases = source_plan(
                self.rank, self.tp_size, self.world_size, meta.remote_tp_size
            )
            block = _one_block(meta.local_block_ids)
            exchange(source, "transfer", block)
            self.storage[block].copy_(self.staging[block], non_blocking=False)
            torch.cuda.synchronize(self.device)
            for unused in releases:
                exchange(unused, "release", block)
            with self._condition:
                self._received_ids.add(req_id)
                self.received_pages += 1
                self.received_bytes += self.spec.state_page_bytes
        except Exception as error:
            self._fail(error)

    def start(self, metadata: FlagCXConnectorMetadata):
        if self.storage is None:
            raise RuntimeError("PD state storage was not registered")
        if self.role == "kv_producer":
            with self._condition:
                for p_req_id, (transfer_id, groups) in metadata.reqs_to_send.items():
                    if groups:
                        if transfer_id in self._pending_send:
                            raise ValueError("duplicate pending PD transfer")
                        block = _one_block(groups)
                        if self.host_snapshot_capacity:
                            if not self._free_host_slots:
                                raise RuntimeError(
                                    "Prefill host snapshot capacity exhausted"
                                )
                            slot = self._free_host_slots.pop()
                            self.staging[slot].copy_(
                                self.storage[block], non_blocking=False
                            )
                            torch.cuda.synchronize(self.device)
                            self._pending_send[transfer_id] = (p_req_id, slot)
                            self._sent_ids.add(p_req_id)
                            self.offloaded_pages += 1
                        else:
                            self._pending_send[transfer_id] = (p_req_id, block)
                self._condition.notify_all()
        else:
            for req_id, meta in metadata.reqs_to_recv.items():
                self._receiver_pool.submit(self._receive_one, req_id, meta)

    def get_finished(self, finished_req_ids: set[str]):
        with self._condition:
            if self._fatal_error is not None:
                raise RuntimeError(f"FL PD transfer failed: {self._fatal_error}")
            sent, received = self._sent_ids, self._received_ids
            self._sent_ids, self._received_ids = set(), set()
        return sent or None, received or None

    def stats(self) -> dict:
        with self._condition:
            return {
                "rank": self.rank,
                "role": self.role,
                "layout_hash": self.spec.state_layout_hash,
                "page_bytes": self.spec.state_page_bytes,
                "pinned_host_staging": self.staging is not None,
                "sent_pages": self.sent_pages,
                "received_pages": self.received_pages,
                "released_pages": self.released_pages,
                "tp_size": self.tp_size,
                "world_size": self.world_size,
                "sent_bytes": self.sent_bytes,
                "received_bytes": self.received_bytes,
                "pending_sends": len(self._pending_send),
                "host_snapshot_capacity": self.host_snapshot_capacity,
                "offloaded_pages": self.offloaded_pages,
                "fatal_error": self._fatal_error,
            }

    def shutdown(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._receiver_pool.shutdown(wait=False, cancel_futures=True)
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
        self._context.term()
        self.flagcx.flagcxP2pEngineDestroy(self.engine)


class DeepseekV41FLConnector(KVConnectorBase_V1, SupportsHMA):
    """vLLM 0.28 external connector for opaque FL request-state pages."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        config = vllm_config.kv_transfer_config
        assert config is not None
        self.scheduler = (
            DeepseekV41Scheduler(vllm_config, config.engine_id, kv_cache_config)
            if role == KVConnectorRole.SCHEDULER
            else None
        )
        self.worker = (
            FullStateFlagCXWorker(vllm_config, kv_cache_config)
            if role == KVConnectorRole.WORKER
            else None
        )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        assert self.scheduler is not None
        return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        assert self.scheduler is not None
        return self.scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(self, scheduler_output):
        assert self.scheduler is not None
        return self.scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request, block_ids):
        assert self.scheduler is not None
        return self.scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(self, request, block_ids):
        assert self.scheduler is not None
        return self.scheduler.request_finished_all_groups(request, block_ids)

    def register_kv_caches(self, kv_caches):
        assert self.worker is not None
        self.worker.register(kv_caches)

    def start_load_kv(self, forward_context, **kwargs):
        assert self.worker is not None
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, FlagCXConnectorMetadata):
            raise TypeError("FL PD connector metadata type mismatch")
        self.worker.start(metadata)

    def wait_for_layer_load(self, layer_name):
        return None

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return None

    def wait_for_save(self):
        return None

    def get_finished(self, finished_req_ids):
        assert self.worker is not None
        return self.worker.get_finished(finished_req_ids)

    def shutdown(self):
        if self.worker is not None:
            self.worker.shutdown()

    def stats(self):
        return self.worker.stats() if self.worker is not None else None
