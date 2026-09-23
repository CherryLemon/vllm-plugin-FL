# SPDX-License-Identifier: Apache-2.0
"""Real model compositions with mutable state, arbitrary pages and positions."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_fl.strict028.batch_diagnostics import compare_model_layers
from vllm_fl.strict028.batched_decode import DecodeBatch
from vllm_fl.strict028.batched_graph import BatchedDecodeGraphs
from vllm_fl.strict028.cache import RequestState
from vllm_fl.strict028.models.deepseek_v41.model import (
    ModelArgs,
    Transformer,
    set_dtype,
)
from vllm_fl.strict028.models.deepseek_v41.ops import (
    decode_dense_batch,
    decode_mean,
    decode_sum,
    grouped_output_projection,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_model(num_pages=5, parallel=None):
    torch.manual_seed(20260923)
    args = ModelArgs(
        max_batch_size=1,
        max_seq_len=160,
        temperature=0,
        vocab_size=64,
        dim=64,
        moe_inter_dim=64,
        n_layers=5,
        n_mtp_layers=1,
        n_heads=4,
        n_routed_experts=4 if parallel is None else parallel.world_size,
        n_activated_experts=2,
        q_lora_rank=64,
        head_dim=64,
        rope_head_dim=32,
        o_groups=1 if parallel is None else parallel.tensor_size,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(0, 2, 2, 1, 1, 0),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 2, 3, 4),
        index_n_heads=4,
        index_head_dim=64,
        index_topk=16,
        candidate_source_layer=3,
        candidate_topk_blocks=3,
        candidate_block_size=8,
        dspark_block_size=5,
        dspark_target_layer_ids=(3, 4),
        dspark_markov_rank=32,
        dspark_noise_token_id=2,
        swiglu_limit=10,
    )
    with torch.device("cuda"), set_dtype(torch.bfloat16):
        core = Transformer(args, enable_mtp=True)
    with torch.no_grad():
        for name, param in core.named_parameters():
            if param.dtype == torch.float4_e2m1fn_x2:
                param.view(torch.uint8).random_(0, 256)
            elif param.dtype == torch.float8_e8m0fnu:
                param.view(torch.uint8).fill_(120)  # 2**-7
            elif param.dtype == torch.float8_e4m3fn:
                param.copy_(torch.randn(param.shape, device="cuda") * 0.1)
            elif (
                name.endswith("norm.weight") or "hc_" in name and name.endswith("scale")
            ):
                param.fill_(1)
            else:
                param.normal_(0, 0.02)
        # Include Engram's history/hash contract without a tokenizer download.
        from vllm_fl.strict028.models.deepseek_v41.engram import NgramHashState

        hasher = NgramHashState.__new__(NgramHashState)
        torch.nn.Module.__init__(hasher)
        hasher.layout = SimpleNamespace(max_ngram_size=4)
        hasher.pad_id = 2
        hasher.register_buffer("token_map", torch.arange(64, device="cuda"))
        hasher.register_buffer(
            "cache", torch.zeros(1, 160, device="cuda", dtype=torch.long)
        )
        hasher.register_buffer(
            "multipliers", torch.tensor([[3, 7, 11, 13]], device="cuda")
        )
        hasher.register_buffer(
            "primes", torch.tensor([[[17], [19], [23]]], device="cuda")
        )
        hasher.register_buffer("offsets", torch.tensor([[0, 17, 36]], device="cuda"))
        core.engram_hash = hasher
    core.requires_grad_(False)
    model = SimpleNamespace(core=core, args=args, speculative_config=object())
    state = RequestState(core, args.max_seq_len)
    config = KVCacheConfig(
        num_pages,
        [KVCacheTensor(num_pages * state.page_bytes, ["fl_request_state"])],
        [KVCacheGroupSpec(["fl_request_state"], state.spec)],
    )
    state.allocate(config, "cuda")
    for page in range(num_pages):
        state.bind(page, reset=True)
    return model, state


@torch.inference_mode()
def serial_target(model, state, tokens, pages, positions, active):
    outputs, hiddens = [], []
    for token, page, position, enabled in zip(
        tokens.tolist(), pages.tolist(), positions.tolist(), active.tolist()
    ):
        state.bind(page)
        if enabled:
            with torch.device("cuda"), set_dtype(torch.bfloat16):
                _, logits, hidden = model.core(
                    torch.tensor([[token]], device="cuda"), position
                )
                model.core.store_spec_context(hidden, position)
            outputs.append(logits)
            hiddens.append(hidden)
    return torch.cat(outputs), torch.cat(hiddens)


@torch.inference_mode()
def test_batched_target_draft_graph_with_positions_pages_and_padding():
    model, state = make_model()
    graph = BatchedDecodeGraphs(model, state, torch.device("cuda"))
    # Seed different, genuine prefix states, including both sides of ring and
    # compressor boundaries. No random KV substitutes for model prefill.
    for page, length in [(1, 7), (2, 8), (3, 127)]:
        state.bind(page, reset=True)
        with torch.device("cuda"), set_dtype(torch.bfloat16):
            _, _, hidden = model.core(
                torch.arange(length, device="cuda").remainder(64)[None], 0
            )
            model.core.store_spec_context(hidden, 0)
    for order, positions_list, active_list in (
        ([1, 2, 3], [7, 8, 127], [True, True, True]),
        ([3, 1, 2], [128, 8, 9], [True, True, True]),
        ([2, 3, 1], [10, 129, 9], [True, False, True]),
    ):
        pages = torch.tensor(order, device="cuda")
        positions = torch.tensor(positions_list, device="cuda")
        active = torch.tensor(active_list, device="cuda")
        tokens = (positions * 3 + pages).remainder(64)
        initial = state.storage.clone()
        ref_logits, ref_hidden = serial_target(
            model, state, tokens, pages, positions, active
        )
        expected = state.storage.clone()
        state.storage.copy_(initial)
        logits, hidden = graph.target_batch(tokens, pages, positions, active)
        assert torch.isfinite(logits[active]).all()
        torch.testing.assert_close(logits[active], ref_logits, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(hidden[active], ref_hidden, rtol=0, atol=0)
        for field in state.fields:
            actual_field = (
                state.storage[:, field.offset : field.offset + field.nbytes]
                .contiguous()
                .view(field.dtype)
            )
            expected_field = (
                expected[:, field.offset : field.offset + field.nbytes]
                .contiguous()
                .view(field.dtype)
            )
            torch.testing.assert_close(
                actual_field,
                expected_field,
                rtol=0,
                atol=0,
                msg=lambda msg, field=field: f"{field.module}.{field.name}: {msg}",
            )
        bonus = logits.argmax(-1)
        before_draft = state.storage.clone()
        reference = []
        for i, (page, position) in enumerate(zip(order, positions_list)):
            if active_list[i]:
                state.bind(page)
                with torch.device("cuda"), set_dtype(torch.bfloat16):
                    reference.append(
                        model.core.forward_spec(
                            bonus[i : i + 1], hidden[i : i + 1], position
                        )
                    )
        expected_draft = state.storage.clone()
        state.storage.copy_(before_draft)
        result = graph.draft_batch(bonus, hidden, pages, positions, active)
        for component, actual in enumerate(result):
            reference_component = torch.cat([r[component] for r in reference])
            torch.testing.assert_close(
                actual[active], reference_component, rtol=1e-4, atol=1e-4
            )
        torch.testing.assert_close(state.storage, expected_draft, rtol=0, atol=0)
    assert graph.stats()["target_graphs"] == 1
    assert graph.stats()["draft_graphs"] == 1
    assert graph.stats()["page_copy_bytes"] == 0


@torch.inference_mode()
def test_device_engram_hashes_match_serial_and_do_not_write_inactive_pages():
    model, state = make_model()
    module = model.core.engram_hash
    pages = torch.tensor([2, 4, 1], device="cuda")
    positions = torch.tensor([[1], [7], [128]], device="cuda")
    active = torch.tensor([[True], [True], [False]], device="cuda")
    ids = torch.tensor([[14], [7], [29]], device="cuda")
    pool = state.pool(module, "cache")
    pool.copy_(torch.arange(160, device="cuda").remainder(64))
    original = state.storage.clone()
    expected = []
    for i in range(2):
        state.bind(int(pages[i]))
        expected.append(module(ids[i : i + 1], int(positions[i])))
    final = state.storage.clone()
    state.storage.copy_(original)
    context = DecodeBatch(state, pages, positions, active)
    actual = module(ids, context)
    torch.testing.assert_close(actual[:2], torch.cat(expected), rtol=0, atol=0)
    torch.testing.assert_close(state.storage, final, rtol=0, atol=0)


@torch.inference_mode()
def test_twenty_request_graph_keeps_cache_untouched_during_capture():
    model, state = make_model(21)
    pages = torch.arange(1, 21, device="cuda")
    for page in range(1, 21):
        state.bind(page, reset=True)
        with torch.device("cuda"), set_dtype(torch.bfloat16):
            _, _, hidden = model.core(
                torch.tensor([[page, 1, 2, 3, 4, 5, 6]], device="cuda"), 0
            )
            model.core.store_spec_context(hidden, 0)
    tokens = pages.remainder(64)
    positions = torch.full((20,), 7, device="cuda")
    active = torch.ones(20, dtype=torch.bool, device="cuda")
    initial = state.storage.clone()
    graph = BatchedDecodeGraphs(model, state, torch.device("cuda"))
    record = graph._capture(tokens, pages, positions, active, None)
    torch.testing.assert_close(state.storage, initial, rtol=0, atol=0)
    graph.targets[20] = record
    ref_logits, ref_hidden = serial_target(
        model, state, tokens, pages, positions, active
    )
    expected = state.storage.clone()
    state.storage.copy_(initial)
    logits, hidden = graph.target_batch(tokens, pages, positions, active)
    torch.testing.assert_close(logits, ref_logits, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(hidden, ref_hidden, rtol=0, atol=0)
    torch.testing.assert_close(state.storage, expected, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("draft", [False, True])
def test_layer_probe_restores_state_and_removes_hooks(draft):
    model, state = make_model()
    hiddens = []
    for page, length in ((1, 2), (2, 3), (3, 4)):
        state.bind(page, reset=True)
        with torch.device("cuda"), set_dtype(torch.bfloat16):
            _, _, hidden = model.core(torch.arange(length, device="cuda")[None], 0)
            model.core.store_spec_context(hidden, 0)
            hiddens.append(hidden[:, -1:])
    saved = state.storage.clone()
    runner = SimpleNamespace(
        model=model,
        state=state,
        graphs=BatchedDecodeGraphs(model, state, torch.device("cuda")),
    )
    with torch.device("cuda"), set_dtype(torch.bfloat16):
        report = compare_model_layers(
            runner,
            torch.tensor([4, 5, 6]),
            torch.tensor([1, 2, 3]),
            torch.tensor([2, 3, 4]),
            draft_hidden=torch.cat(hiddens) if draft else None,
        )
    assert report["serial_events"] > 0 and report["batch_events"] > 0
    if not draft:
        assert not report["first_differences"]
    torch.testing.assert_close(state.storage, saved, rtol=0, atol=0)
    assert not any(
        m._forward_hooks or m._forward_pre_hooks for m in model.core.modules()
    )


@pytest.mark.parametrize("width", [512, 1280, 5120, 20480])
def test_decode_statistics_preserve_real_dimension_reduction(width):
    torch.manual_seed(41)
    values = (
        torch.randn(20, 1, width, device="cuda", dtype=torch.bfloat16).float().square()
    )
    reference = torch.cat([row.mean(-1, keepdim=True) for row in values.split(1)])
    with decode_dense_batch(20):
        actual = decode_mean(values, -1, keepdim=True)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [3, 20])
def test_engram_dot_sum_preserves_request_reduction(batch):
    torch.manual_seed(41)
    values = torch.randn(batch, 1, 4, 5120, device="cuda", dtype=torch.bfloat16).float()
    reference = torch.cat([part.sum(-1) for part in values.split(1)])
    with decode_dense_batch(batch):
        actual = decode_sum(values, -1)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("batch", [3, 20])
def test_five_token_woa_preserves_reference_matmul_on_graph_replay(batch):
    torch.manual_seed(17)
    x = torch.randn(batch, 5, 1, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1, 1024, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
    with decode_dense_batch(batch):
        grouped_output_projection(x, weight)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = grouped_output_projection(x, weight)
    for _ in range(3):
        x.normal_()
        reference = torch.cat(
            [torch.einsum("bsgd,grd->bsgr", part, weight) for part in x.split(1)]
        )
        graph.replay()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@torch.inference_mode()
def test_fixed_capacity_graph_masks_padding_and_participates_when_idle():
    model, state = make_model()
    state.bind(1, reset=True)
    with torch.device("cuda"), set_dtype(torch.bfloat16):
        _, _, prefix_hidden = model.core(torch.tensor([[3, 2, 1]], device="cuda"), 0)
        model.core.store_spec_context(prefix_hidden, 0)
    initial = state.storage.clone()
    graph = BatchedDecodeGraphs(model, state, torch.device("cuda"), batch_capacity=3)
    tokens = torch.tensor([2], device="cuda")
    pages = torch.tensor([1], device="cuda")
    positions = torch.tensor([3], device="cuda")
    active = torch.tensor([True], device="cuda")
    expected_logits, expected_hidden = serial_target(
        model, state, tokens, pages, positions, active
    )
    expected_state = state.storage.clone()
    state.storage.copy_(initial)
    logits, hidden = graph.target_batch(tokens, pages, positions, active)
    torch.testing.assert_close(logits, expected_logits, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(hidden, expected_hidden, rtol=0, atol=0)
    torch.testing.assert_close(state.storage, expected_state, rtol=0, atol=0)
    # Empty DP lanes execute the same captured kernels and collectives, but
    # return no samples and cannot modify either null or live request pages.
    empty = tokens[:0]
    idle_logits, idle_hidden = graph.target_batch(empty, empty, empty, active[:0])
    assert idle_logits.shape[0] == idle_hidden.shape[0] == 0
    torch.testing.assert_close(state.storage, expected_state, rtol=0, atol=0)
    graph.draft_batch(empty, hidden[:0], empty, empty, active[:0])
    torch.testing.assert_close(state.storage, expected_state, rtol=0, atol=0)
    assert graph.stats()["target_batch_sizes"] == [3]
    assert graph.stats()["draft_batch_sizes"] == [3]


@torch.inference_mode()
def test_validation_rpc_uses_host_snapshots_and_restores_live_pages():
    from vllm_fl.strict028.validation import ReferenceProbeExtension

    model, state = make_model()
    graphs = BatchedDecodeGraphs(model, state, torch.device("cuda"), batch_capacity=4)
    runner = SimpleNamespace(
        model=model,
        state=state,
        graphs=graphs,
        requests={},
        batched_decode_enabled=True,
    )
    worker = SimpleNamespace(
        model_runner=runner,
        get_model=lambda: model,
        device=torch.device("cuda"),
        global_rank=0,
    )
    original = state.storage.clone()
    report = ReferenceProbeExtension.fl_batched_decode_differential(
        worker, list(range(40))
    )
    assert torch.equal(state.storage[1:4], original[1:4])
    assert all(
        r["logits_equal"]
        and r["hidden_equal"]
        and r["state_equal"]
        and r["draft_state_equal"]
        and all(r["draft_equal"])
        for r in report["passes"]
    )
