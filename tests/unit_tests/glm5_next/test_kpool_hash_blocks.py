# SPDX-License-Identifier: Apache-2.0
"""Exercise real vLLM 0.24 hashing and allocation with a 4-token tail group."""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils, single_type_kv_cache_manager
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinatorNoPrefixCache,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from vllm_fl.models.glm5_next_kpool import KpoolTailManager, KpoolTailSpec
from vllm_fl.patches import glm5_next_kpool as kpool_patch


@pytest.fixture(scope="module")
def engine_hooks():
    kpool_patch.install_glm5_next_kpool()
    # Use the real upstream callable retained by the owned patch to reproduce
    # the regression, rather than reimplementing the upstream fallback.
    return next(
        patch.pristine
        for patch in kpool_patch._EARLY_PATCHES
        if patch.owner is kv_cache_utils
        and patch.attr == "resolve_kv_cache_block_sizes"
    )


@pytest.fixture(autouse=True)
def deterministic_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(
        kv_cache_utils,
        "NONE_HASH",
        getattr(kv_cache_utils, "NONE_HASH", None),
        raising=False,
    )
    kv_cache_utils.init_none_hash(sha256)


def _engine_config(
    *, prefix=True, hash_size=None, engine_block=64, dcp=1, pcp=1, connector=False
):
    # These are the already-resolved config values consumed by the public
    # resolver. Spec, coordinator, manager, Request and hashing stay real.
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=engine_block,
            enable_prefix_caching=prefix,
            hash_block_size=hash_size,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
        kv_transfer_config=SimpleNamespace() if connector else None,
    )


def _mla_spec(*, block_size=512, compress_ratio=1):
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=compress_ratio,
        indexes_kv_by_block_stride=True,
    )


def _mamba_spec(mode="align"):
    return MambaSpec(
        block_size=512,
        shapes=((2, 128), (128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode=mode,
    )


def _tail_cache_config(config, *, mamba_mode="align"):
    single_type_kv_cache_manager.register_all_kvcache_specs(config)
    specs = {
        "layers.0.mla": _mla_spec(),
        "layers.0.indexer.k_cache": _mla_spec(compress_ratio=4),
        "layers.0.indexer.tail_cache": KpoolTailSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=256,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=4,
            indexes_kv_by_block_stride=True,
        ),
        "layers.1.kda": _mamba_spec(mamba_mode),
    }
    groups = kv_cache_utils.get_kv_cache_groups(config, specs)
    # CPU bookkeeping requires groups and a block pool, not GPU cache tensors.
    worker_cache = KVCacheConfig(
        num_blocks=64, kv_cache_tensors=[], kv_cache_groups=groups
    )
    # EngineCore performs this real conversion before resolver/Scheduler:
    # worker UniformType groups become representative scheduler specs while
    # the main MLA and compressed indexer still share their block table.
    return kv_cache_utils.generate_scheduler_kv_cache_config([worker_cache])


def _new_manager(cache, config, *, sizes=None):
    if sizes is None:
        sizes = kv_cache_utils.resolve_kv_cache_block_sizes(cache, config)
    return KVCacheManager(
        kv_cache_config=cache,
        max_model_len=4096,
        scheduler_block_size=sizes[0],
        hash_block_size=sizes[1],
        max_num_batched_tokens=2048,
        enable_caching=config.cache_config.enable_prefix_caching,
    )


def _request(name, prompt, hash_size, *, prefix=True):
    # EngineCore creates a hasher only when prefix caching or a KV connector
    # is enabled. Follow that lifecycle for the disabled-prefix cases too.
    hasher = (
        kv_cache_utils.get_request_block_hasher(hash_size, sha256) if prefix else None
    )
    return Request(
        request_id=name,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        block_hasher=hasher,
    )


def test_upstream_mamba_fallback_reproduces_tail_coordinator_failure(engine_hooks):
    config = _engine_config(prefix=True, engine_block=64)
    cache = _tail_cache_config(config)
    sizes = engine_hooks(cache, config)
    # This is the actual serving shape: both large groups use 512 tokens,
    # whereas the ring uses 4. The upstream fallback ignores this ring.
    assert sizes == (512, 512)
    with pytest.raises(AssertionError, match="block_size must be divisible"):
        _new_manager(cache, config, sizes=sizes)


@pytest.mark.parametrize("prefix", [True, False])
@pytest.mark.parametrize("hash_size", [None, 1, 2, 4])
def test_real_manager_hash_allocate_decode_and_free(engine_hooks, prefix, hash_size):
    config = _engine_config(prefix=prefix, hash_size=hash_size)
    cache = _tail_cache_config(config)
    sizes = kv_cache_utils.resolve_kv_cache_block_sizes(cache, config)
    manager = _new_manager(cache, config, sizes=sizes)
    expected_coordinator = (
        HybridKVCacheCoordinator if prefix else KVCacheCoordinatorNoPrefixCache
    )
    assert isinstance(manager.coordinator, expected_coordinator)
    assert sizes == (512, 4 if hash_size is None else hash_size)
    tail_group, tail_manager = next(
        (index, candidate)
        for index, candidate in enumerate(manager.coordinator.single_type_managers)
        if isinstance(candidate, KpoolTailManager)
    )
    free_before = manager.block_pool.get_num_free_blocks()
    # Equal prompts must still receive distinct tail rings for simultaneous
    # requests; their equal hash chains exercise the strongest reuse case.
    prompt = [100 + index % 257 for index in range(3072)]
    requests = [_request(name, prompt, sizes[1], prefix=prefix) for name in ("a", "b")]
    assert requests[0].block_hashes == requests[1].block_hashes
    for request in requests:
        assert len(request.block_hashes) == (3072 // sizes[1] if prefix else 0)
    rings = {}
    replay = None
    try:
        for chunk in (512, 512, 1024, 1024):
            manager.new_step_starts()
            for request in requests:
                allocated = manager.allocate_slots(request, num_new_tokens=chunk)
                assert allocated is not None
                request.num_computed_tokens += chunk
                blocks = manager.get_blocks(request.request_id).blocks[tail_group]
                assert len(blocks) == 1
                ring = blocks[0]
                assert not ring.is_null and ring.block_hash is None
                assert ring.ref_cnt == 1
                if request.request_id in rings:
                    assert ring is rings[request.request_id]
                    assert not allocated.blocks[tail_group]
                else:
                    rings[request.request_id] = ring
                    assert len(allocated.blocks[tail_group]) == 1
            assert rings["a"] is not rings["b"]
        assert tail_manager.get_num_common_prefix_blocks("a") == 0
        # Real hashing grows through a decode boundary, preserving the prefix
        # chain. The ring remains one unhashed block after this growth.
        manager.new_step_starts()
        for request in requests:
            previous_hashes = request.block_hashes.copy()
            request.append_output_token_ids([700, 701, 702, 703])
            assert request.block_hashes[: len(previous_hashes)] == previous_hashes
            assert len(request.block_hashes) == (3076 // sizes[1] if prefix else 0)
            assert manager.allocate_slots(request, num_new_tokens=4) is not None
            request.num_computed_tokens += 4
            ring = manager.get_blocks(request.request_id).blocks[tail_group]
            assert len(ring) == 1 and ring[0] is rings[request.request_id]
            assert ring[0].block_hash is None
        # Replay through the actual coordinator must not return a tail page.
        manager.new_step_starts()
        replay = _request("replay", prompt, sizes[1], prefix=prefix)
        hit_blocks, hit_tokens = manager.get_computed_blocks(replay)
        assert not hit_blocks.blocks[tail_group]
        if prefix:
            assert hit_tokens > 0 and hit_tokens % sizes[0] == 0
            assert hit_blocks.blocks[0]
        else:
            assert hit_tokens == 0
        allocated = manager.allocate_slots(
            replay,
            num_new_tokens=min(512, replay.num_tokens - hit_tokens),
            num_new_computed_tokens=hit_tokens,
            new_computed_blocks=hit_blocks,
        )
        assert allocated is not None
        assert len(allocated.blocks[tail_group]) == 1
        replay_ring = manager.get_blocks(replay.request_id).blocks[tail_group]
        assert len(replay_ring) == 1
        assert all(replay_ring[0] is not ring for ring in rings.values())
        assert replay_ring[0].block_hash is None
    finally:
        if replay is not None:
            manager.free(replay)
        for request in requests:
            manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert all(ring.ref_cnt == 0 for ring in rings.values())
    assert all(ring.block_hash is None for ring in rings.values())
    # Freeing releases memory, not a reusable prefix entry for raw tail data.
    later = _request("later", prompt, sizes[1], prefix=prefix)
    later_hits, _ = manager.get_computed_blocks(later)
    assert not later_hits.blocks[tail_group]
    manager.new_step_starts()
    allocated = manager.allocate_slots(later, num_new_tokens=512)
    assert allocated is not None and len(allocated.blocks[tail_group]) == 1
    later_ring = manager.get_blocks(later.request_id).blocks[tail_group]
    assert len(later_ring) == 1 and later_ring[0].block_hash is None
    manager.free(later)
    assert manager.block_pool.get_num_free_blocks() == free_before


@pytest.mark.parametrize("mode", ["none", "all"])
@pytest.mark.parametrize("prefix,connector", [(True, False), (False, True)])
def test_kda_mode_without_aligned_snapshots_rejects_prefix_reuse(
    engine_hooks, mode, prefix, connector
):
    config = _engine_config(prefix=prefix, connector=connector)
    cache = _tail_cache_config(config, mamba_mode=mode)
    with pytest.raises(ValueError, match="requires aligned KDA state pages"):
        _new_manager(cache, config)


@pytest.mark.parametrize("prefix", [True, False])
def test_invalid_hash_override_rejected_before_manager_construction(
    engine_hooks, prefix
):
    config = _engine_config(prefix=prefix, hash_size=8)
    cache = _tail_cache_config(config)
    with pytest.raises(ValueError, match="Invalid hash_block_size=8"):
        _new_manager(cache, config)


@pytest.mark.parametrize("prefix", [True, False])
@pytest.mark.parametrize("dcp,pcp", [(2, 1), (1, 2), (2, 2)])
def test_tail_groups_reject_context_parallelism(engine_hooks, prefix, dcp, pcp):
    config = _engine_config(prefix=prefix, dcp=dcp, pcp=pcp)
    cache = _tail_cache_config(config)
    with pytest.raises(ValueError, match="context parallelism"):
        _new_manager(cache, config)


@pytest.mark.parametrize(
    "prefix,hash_size,engine_block,dcp,pcp,shape",
    [
        (True, None, 64, 1, 1, "mamba"),
        (False, 2, 64, 1, 1, "mamba"),
        (True, 8, 512, 1, 1, "mamba"),
        (True, 4, 64, 1, 1, "window"),
        (True, 8, 64, 1, 1, "window"),
        (True, None, 64, 2, 1, "window"),
        (True, None, 64, 1, 2, "single"),
    ],
)
def test_other_cache_layouts_preserve_upstream_resolver_contract(
    engine_hooks, prefix, hash_size, engine_block, dcp, pcp, shape
):
    config = _engine_config(
        prefix=prefix,
        hash_size=hash_size,
        engine_block=engine_block,
        dcp=dcp,
        pcp=pcp,
    )
    groups = [KVCacheGroupSpec(["mla"], _mla_spec())]
    if shape == "mamba":
        groups.append(KVCacheGroupSpec(["ssm"], _mamba_spec()))
    elif shape == "window":
        groups.append(
            KVCacheGroupSpec(
                ["window"],
                SlidingWindowSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=128,
                    dtype=torch.bfloat16,
                    sliding_window=4,
                ),
            )
        )
    cache = KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=groups)
    try:
        expected = engine_hooks(cache, config)
    except ValueError as error:
        with pytest.raises(ValueError) as patched_error:
            kv_cache_utils.resolve_kv_cache_block_sizes(cache, config)
        assert str(patched_error.value) == str(error)
    else:
        assert kv_cache_utils.resolve_kv_cache_block_sizes(cache, config) == expected


def test_imported_engine_alias_is_owned_and_partial_install_rolls_back():
    """Real modules, fresh process, one failure at the setattr install boundary."""
    code = textwrap.dedent(
        """
        import builtins
        import vllm.v1.engine.core as engine_core
        from vllm.v1.core import kv_cache_utils
        from vllm_fl import activation
        from vllm_fl.patches import glm5_next_kpool as kpool

        assert kpool._EARLY_PATCHES is None, "bootstrap already ran in fresh process"
        pristine_resolver = kv_cache_utils.resolve_kv_cache_block_sizes
        pristine_alias = engine_core.resolve_kv_cache_block_sizes
        inventory_before = activation.patch_inventory()

        def fail_alias_install(owner, attr, value):
            if owner is engine_core and attr == "resolve_kv_cache_block_sizes":
                raise RuntimeError("injected engine alias install failure")
            builtins.setattr(owner, attr, value)

        activation.setattr = fail_alias_install
        try:
            try:
                kpool.install_glm5_next_kpool()
            except RuntimeError as error:
                assert str(error) == "injected engine alias install failure"
            else:
                raise AssertionError("engine alias was not staged in the transaction")
        finally:
            del activation.setattr
        assert kv_cache_utils.resolve_kv_cache_block_sizes is pristine_resolver
        assert engine_core.resolve_kv_cache_block_sizes is pristine_alias
        assert activation.patch_inventory() == inventory_before
        assert kpool._EARLY_PATCHES is None

        kpool.install_glm5_next_kpool()
        assert engine_core.resolve_kv_cache_block_sizes is kv_cache_utils.resolve_kv_cache_block_sizes
        targets = {
            record["target"]
            for record in activation.patch_inventory()
            if record["fingerprint"] == "glm5.kpool" and record["phase"] == "engine/config"
        }
        assert any(target.endswith("kv_cache_utils.resolve_kv_cache_block_sizes") for target in targets)
        assert any(target.endswith("engine.core.resolve_kv_cache_block_sizes") for target in targets)
        # Re-registration remains idempotent after a failed first transaction.
        resolver = kv_cache_utils.resolve_kv_cache_block_sizes
        kpool.install_glm5_next_kpool()
        assert kv_cache_utils.resolve_kv_cache_block_sizes is resolver
        assert engine_core.resolve_kv_cache_block_sizes is resolver
        """
    )
    env = dict(os.environ)
    # This process explicitly installs the real engine/config hooks, so keep
    # general-plugin auto-loading from changing the intended import order.
    env["VLLM_PLUGINS"] = ""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
