"""Check the real vLLM collective boundary without launching device collectives."""

import ctypes
from types import SimpleNamespace

import torch


def test_all_reduce_custom_op_keeps_ctypes_out_of_dynamo(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state

    calls = []

    def all_reduce(x):
        # Model the untraceable pointer conversion used by PyNCCL. The real
        # GroupCoordinator custom op must defer this until graph execution.
        ctypes.c_void_p(x.data_ptr())
        calls.append(x.clone())
        return x * 2

    group = parallel_state.GroupCoordinator.__new__(parallel_state.GroupCoordinator)
    group.world_size = 2
    group.unique_name = "fl_cpu_compile_regression"
    group.use_custom_op_call = True
    group.device_communicator = SimpleNamespace(all_reduce=all_reduce)
    monkeypatch.setitem(parallel_state._groups, group.unique_name, lambda: group)
    # Vendor builds normally register this op on their accelerator dispatch
    # key. Use the same implementation on CPU solely for this contract test.
    cpu_lib = torch.library.Library("vllm", "IMPL", "CPU")
    if not torch._C._dispatch_has_kernel_for_dispatch_key("vllm::all_reduce", "CPU"):
        cpu_lib.impl("all_reduce", parallel_state.all_reduce)

    graphs = []

    def backend(graph, _example_inputs):
        graphs.append(graph)
        return graph.forward

    try:
        compiled = torch.compile(group.all_reduce, backend=backend, fullgraph=True)
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        torch.testing.assert_close(compiled(x), x * 2)
        updated = x + 7
        torch.testing.assert_close(compiled(updated), updated * 2)
        assert len(calls) == 2
        torch.testing.assert_close(calls[-1], updated)
        assert len(graphs) == 1
        assert any(
            node.target
            in (torch.ops.vllm.all_reduce, torch.ops.vllm.all_reduce.default)
            for node in graphs[0].graph.nodes
        ), graphs[0].code
    finally:
        torch._dynamo.reset()
        cpu_lib._destroy()
