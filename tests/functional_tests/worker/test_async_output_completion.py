# SPDX-License-Identifier: Apache-2.0
"""Actual CUDA stream callbacks, D2H visibility and graph replay."""

import importlib
import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from vllm_fl.worker.async_output import (
    _NativeEventfdCompletionPool,
    _wait_for_async_output_event,
)


@pytest.fixture
def pool():
    from vllm.platforms import current_platform

    if not current_platform.is_cuda() or current_platform.is_rocm():
        pytest.skip("native completion requires NVIDIA CUDA")
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("requires NVIDIA CUDA")
    # vLLM's extension must coexist with the plugin extension; it owns _C too.
    name = (
        "vllm._C" if importlib.util.find_spec("vllm._C") else "vllm._C_stable_libtorch"
    )
    importlib.import_module(name)

    original = torch.ops._C.weak_ref_tensor.default
    import vllm_fl._C  # noqa: F401

    assert torch.ops._C.weak_ref_tensor.default is original
    assert torch.ops.vllm_fl.cuda_eventfd_completion_supported()
    instance = _NativeEventfdCompletionPool(
        torch.ops.vllm_fl.enqueue_cuda_eventfd_completion, capacity=2
    )
    yield instance
    torch.cuda.synchronize()
    instance.close()


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("device_index", [0, 1])
def test_callback_observes_fresh_copies_and_reuses_descriptors(
    pool, use_graph, device_index
):
    if torch.cuda.device_count() <= device_index:
        pytest.skip("requires two CUDA devices")
    with torch.cuda.device(device_index):
        stream = torch.cuda.Stream()
        source = torch.zeros(65536, device="cuda", dtype=torch.int64)
        destination = torch.empty_like(source, device="cpu", pin_memory=True)
        graph = None
        if use_graph:
            graph = torch.cuda.CUDAGraph()
            capture_stream = torch.cuda.Stream(device=device_index)
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=capture_stream):
                source.add_(1)
            torch.cuda.current_stream().wait_stream(capture_stream)
            source.zero_()
        descriptors = list(pool._event_fds)
        for value in range(1, 33):
            if graph is not None:
                graph.replay()
            else:
                source.fill_(value)
            destination.fill_(-1)
            with torch.cuda.stream(stream):
                stream.wait_stream(torch.cuda.current_stream())
                destination.copy_(source, non_blocking=True)
                event = torch.cuda.Event()
                event.record()
                completion = pool.enqueue(stream)
            assert completion is not None
            _wait_for_async_output_event(event, completion)
            assert torch.all(destination == value).item()
        assert pool._event_fds == descriptors
        assert len(pool._available) == 2
        assert not pool._quarantined


def test_late_callback_after_shutdown_keeps_descriptor_valid(pool):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(10000000)
        completion = pool.enqueue(stream)
    assert completion is not None
    fd = pool._event_fds[completion.slot]
    pool.close()
    stream.synchronize()
    assert os.eventfd_read(fd) == 1
    # The callback has finished; only the test can now safely reclaim this fd.
    os.close(fd)


def test_cuda_context_error_does_not_hang_or_fallback(pool):
    code = textwrap.dedent("""\
        import torch
        import vllm_fl._C
        from vllm_fl.worker.async_output import (
            _NativeEventfdCompletionPool, _wait_for_async_output_event,
        )
        pool = _NativeEventfdCompletionPool(
            torch.ops.vllm_fl.enqueue_cuda_eventfd_completion, capacity=1,
        )
        stream = torch.cuda.Stream()
        event = torch.cuda.Event()
        try:
            with torch.cuda.stream(stream):
                torch._assert_async(torch.tensor(False, device='cuda'))
                event.record()
                completion = pool.enqueue(stream)
            _wait_for_async_output_event(event, completion)
        except RuntimeError:
            print('CUDA_ERROR_PROPAGATED', flush=True)
        else:
            raise AssertionError('device assertion must propagate')
    """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CUDA_ERROR_PROPAGATED" in result.stdout


@pytest.mark.parametrize("kind", ["tokens", "pooling"])
def test_model_runner_output_matches_event_completion(pool, monkeypatch, kind):
    from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput

    from vllm_fl.worker import async_output
    from vllm_fl.worker.model_runner import (
        AsyncGPUModelRunnerOutput,
        AsyncGPUPoolingModelRunnerOutput,
    )

    results = []
    for native in (False, True):
        monkeypatch.setattr(
            async_output, "_native_completion_pool", pool if native else False
        )
        output = ModelRunnerOutput(req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1})
        stream = torch.cuda.Stream()
        if kind == "tokens":
            tokens = torch.tensor([[17], [23]], device="cuda")
            logprobs = LogprobsTensors(
                logprob_token_ids=tokens.clone(),
                logprobs=torch.tensor([[-1.0], [-2.0]], device="cuda"),
                selected_token_ranks=torch.tensor([1, 2], device="cuda"),
            )
            wrapper = AsyncGPUModelRunnerOutput(
                output, tokens, logprobs, [1], stream, vocab_size=128
            )
        else:
            wrapper = AsyncGPUPoolingModelRunnerOutput(
                output,
                torch.arange(16, device="cuda").reshape(2, 8),
                [True, False],
                stream,
            )
        assert (wrapper._async_copy_completion is not None) == native
        results.append(wrapper.get_output())
    if kind == "tokens":
        assert (
            results[0].sampled_token_ids == results[1].sampled_token_ids == [[17], []]
        )
        for field in ("logprob_token_ids", "logprobs", "sampled_token_ranks"):
            torch.testing.assert_close(
                torch.as_tensor(getattr(results[0].logprobs, field)),
                torch.as_tensor(getattr(results[1].logprobs, field)),
                rtol=0,
                atol=0,
            )
    else:
        assert results[0].pooler_output[1] is results[1].pooler_output[1] is None
        torch.testing.assert_close(
            results[0].pooler_output[0], results[1].pooler_output[0]
        )
        torch.testing.assert_close(results[1].pooler_output[0], torch.arange(8))
