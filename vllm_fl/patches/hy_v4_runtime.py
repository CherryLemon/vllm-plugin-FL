# SPDX-License-Identifier: Apache-2.0
"""HY4 runtime capability checks and transactional compatibility installation.

The empty-build adapter changes process-global vLLM/FlagGems symbols. It is
installed only by HY4 construction, under a lock, and rolled back on failure.
It is not an instance-local provider and must not be used to hot-swap models
in an already initialized worker.
"""

from contextlib import contextmanager
from threading import RLock
from types import SimpleNamespace
from vllm.logger import init_logger

logger = init_logger(__name__)

import torch

from vllm_fl.utils import use_flaggems_op

_INSTALL_LOCK = RLock()
_MISSING = object()


def has_device_kernel(name: str, device_type: str) -> bool:
    """A schema or a Meta kernel alone does not implement runtime execution."""
    key = {"cuda": "CUDA", "cpu": "CPU", "xpu": "XPU"}.get(device_type, "PrivateUse1")
    try:
        return any(
            torch._C._dispatch_has_kernel_for_dispatch_key(name, k)
            for k in (
                key,
                "CompositeImplicitAutograd",
                "CompositeExplicitAutograd",
            )
        )
    except RuntimeError:
        return False


_REQUIRED_GEMS = (
    "per_token_group_quant_fp8",
    "indexer_k_quant_and_cache",
    "cp_gather_indexer_k_quant_cache",
    "top_k_per_row_prefill",
    "top_k_per_row_decode",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_paged_mqa_logits",
    "flash_mla_sparse_fwd",
    "concat_and_cache_mla",
    "flash_attn_varlen_func",
)


def require_flaggems_policy() -> None:
    denied = [name for name in _REQUIRED_GEMS if not use_flaggems_op(name)]
    if denied:
        raise RuntimeError(
            "HY4 empty-build provider requires FlagGems operators disabled by "
            "the active backend/whitelist/blacklist policy: " + ", ".join(denied)
        )


def validate_hy4_runtime(vllm_config) -> None:
    """Reject unsupported combinations before allocating/loading model weights."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        raise ValueError("HY4 currently requires an NVIDIA CUDA runtime")
    from vllm.v1.attention.ops import flashmla

    cache_dtype = vllm_config.cache_config.cache_dtype
    # The portable sparse MLA implementation handles BF16 KV only. FP8 KV
    # additionally requires native metadata and decode extensions.
    if str(cache_dtype).startswith("fp8"):
        supported, reason = flashmla.is_flashmla_sparse_supported()
        if not supported:
            raise ValueError(
                f"HY4 FP8 KV requires native FlashMLA metadata/decode: {reason}"
            )
        if not native_hy4_available():
            raise ValueError(
                "HY4 FlagGems empty-build provider supports only auto/bfloat16 KV; FP8 KV is unsupported"
            )
    if not native_hy4_available():
        require_flaggems_policy()


def native_hy4_available() -> bool:
    from vllm.utils.deep_gemm import has_deep_gemm
    from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

    return (
        has_deep_gemm()
        and is_flashmla_sparse_supported()[0]
        and all(
            has_device_kernel(name, "cuda")
            for name in (
                "_C::per_token_group_fp8_quant",
                "_C_cache_ops::concat_and_cache_mla",
                "_C_cache_ops::concat_mla_q",
            )
        )
    )


class PatchTransaction:
    def __init__(self):
        self.saved = []

    def set(self, target, name, value):
        self.saved.append((target, name, getattr(target, name, _MISSING)))
        setattr(target, name, value)

    def rollback(self):
        for target, name, old in reversed(self.saved):
            if old is _MISSING:
                delattr(target, name)
            else:
                setattr(target, name, old)


@contextmanager
def patch_transaction():
    with _INSTALL_LOCK:
        tx = PatchTransaction()
        try:
            yield tx
        except BaseException:
            tx.rollback()
            raise


def install_hy4_flaggems_fallback() -> bool:
    with patch_transaction() as tx:
        return _install_fallback(tx)


def _make_hy4_flaggems_mla_prefill_backend() -> type:
    """Build the MLA prefill adapter without importing FA extensions."""
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

    class FlagGemsMLAPrefillBackend(MLAPrefillBackend):
        @staticmethod
        def get_name() -> str:
            return "HYV4_FLAGGEMS_MLA_PREFILL"

        @classmethod
        def is_available(cls) -> bool:
            try:
                from flag_gems import flash_attn_varlen_func  # noqa: F401

                return True
            except ImportError:
                return False

        def _flash_attn_varlen(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
            max_seqlen_q: int,
            max_seqlen_k: int,
            causal: bool,
            return_softmax_lse: bool,
            out: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            from flag_gems import flash_attn_varlen_func

            # MLA has q/k head dim 576 and value dim 512.  FlagGems accepts
            # different head dimensions through padding, matching vLLM's
            # FlashAttention prefill adapter.
            maybe_padded_v = v
            if v.shape[-1] != q.shape[-1]:
                maybe_padded_v = torch.nn.functional.pad(
                    v, [0, q.shape[-1] - v.shape[-1]], value=0
                )

            result = flash_attn_varlen_func(
                q=q,
                k=k,
                v=maybe_padded_v,
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_k=cu_seqlens_k,
                softmax_scale=self.scale,
                causal=causal,
                return_softmax_lse=return_softmax_lse,
                # ``out`` has the unpadded value width.  Let the Triton
                # wrapper allocate when padding is needed, then copy below.
                out=out if maybe_padded_v is v else None,
            )
            lse = None
            if isinstance(result, tuple):
                result, lse = result[0], result[1]
            if maybe_padded_v is not v:
                result = result[..., : v.shape[-1]]
                if out is not None:
                    out.copy_(result)
                    result = out
            if return_softmax_lse:
                assert lse is not None
                return result, lse
            return result

        def run_prefill_new_tokens(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            return_softmax_lse: bool,
            out: torch.Tensor | None = None,
            output_scale: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            if output_scale is not None:
                raise NotImplementedError(
                    "HY4 FlagGems MLA prefill does not support quantized output"
                )
            metadata = self._prefill_metadata
            return self._flash_attn_varlen(
                q,
                k,
                v,
                cu_seqlens_q=metadata.query_start_loc,
                cu_seqlens_k=metadata.query_start_loc,
                max_seqlen_q=metadata.max_query_len,
                max_seqlen_k=metadata.max_query_len,
                causal=True,
                return_softmax_lse=return_softmax_lse,
                out=out,
            )

        def run_prefill_context_chunk(
            self,
            chunk_idx: int,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            metadata = self._prefill_metadata
            assert metadata.chunked_context is not None
            chunked = metadata.chunked_context
            result = self._flash_attn_varlen(
                q,
                k,
                v,
                cu_seqlens_q=metadata.query_start_loc,
                cu_seqlens_k=chunked.cu_seq_lens[chunk_idx],
                max_seqlen_q=metadata.max_query_len,
                max_seqlen_k=chunked.max_seq_lens[chunk_idx],
                causal=False,
                return_softmax_lse=True,
            )
            assert isinstance(result, tuple)
            return result

    return FlagGemsMLAPrefillBackend


def _install_fallback(tx) -> bool:
    """Install process-wide compatibility for a dedicated HY4 worker."""
    import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer

    if getattr(sparse_indexer, "_hy4_flaggems_fallback", False):
        require_flaggems_policy()
        return True
    if native_hy4_available():
        return False
    require_flaggems_policy()

    try:
        # FlagGems 5.3.3 advertises Triton TLE support for this runtime, but
        # the bundled Triton TLE language module is missing ``cumsum``.  Its
        # TLE top-k kernels therefore fail while Triton is hashing/compiling
        # the kernel, before any device code can run.  Select the portable
        # non-TLE kernels (the module keeps both implementations) for HY4's
        # local fallback.  This changes the process-global FlagGems module; rollback covers it
        # if installation fails. Use a dedicated HY4 worker.
        import importlib

        top_k_prefill_module = importlib.import_module(
            "flag_gems.fused.top_k_per_row_prefill"
        )
        top_k_decode_module = importlib.import_module(
            "flag_gems.fused.top_k_per_row_decode"
        )
        tx.set(top_k_prefill_module, "HAS_TLE", False)
        tx.set(top_k_decode_module, "HAS_TLE", False)
        # Triton's dependency walker still visits the constexpr-disabled TLE
        # branch while compiling the non-TLE kernels.  Give that walker a
        # cache-key-compatible cumsum symbol; the branch is never emitted
        # because HAS_TLE is false, while the non-TLE path uses tl.cumsum.
        import triton.language as tl

        for top_k_module in (top_k_prefill_module, top_k_decode_module):
            tle = getattr(top_k_module, "tle", None)
            if tle is not None and not hasattr(tle, "cumsum"):
                tx.set(tle, "cumsum", tl.cumsum)

        # ``flash_mla_sparse_fwd`` has a separate TLE gate from the top-k
        # helpers above.  FlagGems selects its TLE implementation on this
        # Triton build, but the bundled ``triton.experimental.tle.language``
        # module is missing ``pipe``; the first HY4 request then fails while
        # Triton hashes the kernel.  Force the portable non-TLE implementation
        # for this worker-wide fallback as well.
        flashmla_sparse_module = importlib.import_module(
            "flag_gems.fused.flashmla_sparse"
        )
        tx.set(flashmla_sparse_module, "HAS_TLE_FLASHMLA_SPARSE", False)

        from flag_gems.fused import (
            concat_and_cache_mla as flaggems_concat_and_cache_mla,
            cp_gather_indexer_k_quant_cache,
            flash_mla_sparse_fwd,
            fp8_fp4_mqa_logits,
            fp8_fp4_paged_mqa_logits,
            indexer_k_quant_and_cache,
            top_k_per_row_decode,
            top_k_per_row_prefill,
        )
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "HY4 empty-build runtime has no DeepGEMM and FlagGems DSA kernels "
            "are unavailable; cannot construct sparse attention safely."
        ) from exc

    # ``sparse_attn_indexer`` captured these names when its custom op was
    # registered.  Rebind the module globals so the already-registered op
    # invokes FlagGems' Triton implementation at runtime.
    tx.set(sparse_indexer, "fp8_fp4_mqa_logits", fp8_fp4_mqa_logits)
    tx.set(sparse_indexer, "fp8_fp4_paged_mqa_logits", fp8_fp4_paged_mqa_logits)
    tx.set(
        sparse_indexer,
        "ops",
        SimpleNamespace(
            indexer_k_quant_and_cache=indexer_k_quant_and_cache,
            cp_gather_indexer_k_quant_cache=cp_gather_indexer_k_quant_cache,
            top_k_per_row_prefill=top_k_per_row_prefill,
            top_k_per_row_decode=top_k_per_row_decode,
        ),
    )

    # vLLM's CUDA indexer selects ``torch.ops._C.persistent_topk`` for every
    # decode batch on CUDA (and the cooperative variant for some shapes).
    # Those are part of the omitted vLLM extension, so make only this module's
    # platform view report a non-CUDA execution path for that branch.  The
    # CustomOp dispatcher has already bound ``SparseAttnIndexer.forward_cuda``
    # using its own platform object; this proxy therefore affects only the
    # registered Python implementation and leaves the rest of vLLM unchanged.
    native_sparse_indexer_platform = sparse_indexer.current_platform

    class _HY4FlagGemsIndexerPlatform:
        def is_cuda(self) -> bool:
            return False

        def __getattr__(self, name):
            return getattr(native_sparse_indexer_platform, name)

    tx.set(sparse_indexer, "current_platform", _HY4FlagGemsIndexerPlatform())

    # HY4's sink-capable backend imports these functions by value.  Rebind its
    # globals (and the base module for inherited helper paths) to avoid the
    # native FlashMLA/NV kernel when the wheel is empty.
    import vllm.v1.attention.backends.mla.flashmla_sparse as native_sparse

    from vllm_fl.models import hy_v4_flashmla_sparse as hy4_sparse

    tx.set(native_sparse, "flash_mla_sparse_fwd", flash_mla_sparse_fwd)
    tx.set(hy4_sparse, "flash_mla_sparse_fwd", flash_mla_sparse_fwd)

    # The empty vLLM wheel also omits the generic MLA cache-update extension.
    # AttentionImplBase imports ``vllm._custom_ops`` at call time, so replacing
    # just this module attribute is sufficient and is process-wide within this dedicated HY4 worker.
    import vllm._custom_ops as vllm_custom_ops

    if not has_device_kernel("_C_cache_ops::concat_and_cache_mla", "cuda"):
        tx.set(vllm_custom_ops, "concat_and_cache_mla", flaggems_concat_and_cache_mla)

    # The sparse MLA backend also concatenates the no-pe and RoPE query parts
    # through ``_C_cache_ops.concat_mla_q``.  This tiny operation is absent
    # from empty-build wheels and has no FlagGems equivalent.  Keep the
    # preallocated vLLM buffer and copy each slice directly, avoiding a
    # temporary concatenated tensor.
    if not has_device_kernel("_C_cache_ops::concat_mla_q", "cuda"):

        def concat_hy4_mla_q(
            ql_nope: torch.Tensor,
            q_pe: torch.Tensor,
            q_out: torch.Tensor,
        ) -> None:
            nope_width = ql_nope.shape[-1]
            rope_width = q_pe.shape[-1]
            if q_out.shape[-1] != nope_width + rope_width:
                raise ValueError(
                    "HY4 concat_mla_q output width mismatch: "
                    f"expected {nope_width + rope_width}, "
                    f"got {q_out.shape[-1]}"
                )
            q_out[..., :nope_width].copy_(ql_nope)
            q_out[..., nope_width:].copy_(q_pe)

        tx.set(vllm_custom_ops, "concat_mla_q", concat_hy4_mla_q)

    # On Hopper vLLM 0.24's automatic prefill selector only considers
    # FlashAttention.  The empty wheel intentionally stubs that extension,
    # even though HY4's sparse path uses MQA only.  Select a FlagGems adapter
    # so layer construction remains valid and dense future HY4 layers also
    # have a real Triton prefill implementation.
    import vllm.model_executor.layers.attention.mla_attention as mla_attention
    import vllm.v1.attention.backends.mla.prefill as prefill
    import vllm.v1.attention.backends.mla.prefill.selector as prefill_selector

    def has_callable_flash_attn_varlen() -> bool:
        """Check the function the FlashAttention backend will actually call.

        The empty-build vLLM wheel keeps the extension module importable and
        makes its availability probe return true on CUDA, but the exported
        function is ``None``.  Checking the backend's captured global catches
        that ABI stub before its constructor asserts.
        """
        try:
            from vllm.v1.attention.backends.mla.prefill import flash_attn

            if callable(getattr(flash_attn, "flash_attn_varlen_func", None)):
                return True
        except (ImportError, OSError):
            pass
        try:
            import vllm.vllm_flash_attn as vllm_flash_attn

            return callable(getattr(vllm_flash_attn, "flash_attn_varlen_func", None))
        except (ImportError, OSError):
            return False

    if not getattr(mla_attention, "_hy4_flaggems_prefill_fallback", False):
        native_get_prefill_backend = mla_attention.get_mla_prefill_backend
        flaggems_prefill_backend = _make_hy4_flaggems_mla_prefill_backend()

        def get_hy4_prefill_backend(vllm_config):
            try:
                backend = native_get_prefill_backend(vllm_config)
                if backend.is_available() and (
                    backend.get_name() != "FLASH_ATTN"
                    or has_callable_flash_attn_varlen()
                ):
                    return backend
            except (ImportError, OSError, AssertionError, ValueError):
                pass
            return flaggems_prefill_backend

        tx.set(mla_attention, "get_mla_prefill_backend", get_hy4_prefill_backend)
        tx.set(prefill, "get_mla_prefill_backend", get_hy4_prefill_backend)
        tx.set(prefill_selector, "get_mla_prefill_backend", get_hy4_prefill_backend)
        tx.set(mla_attention, "_hy4_flaggems_prefill_fallback", True)

    logger.warning_once(
        "HY4: vLLM SparseAttnIndexer has no DeepGEMM; using FlagGems Triton "
        "indexer/top-k and sparse MLA kernels."
    )
    tx.set(sparse_indexer, "_hy4_flaggems_fallback", True)
    return True
