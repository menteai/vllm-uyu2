from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

_PATCHED = False


def _is_uyu_config(vllm_config: Any) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", ()) or ()
    return "Uyu2ForCausalLM" in architectures


def _group_uyu_specs(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    buckets: OrderedDict[tuple[Any, ...], dict[str, KVCacheSpec]] = OrderedDict()

    for layer_name, spec in kv_cache_specs.items():
        if isinstance(spec, SlidingWindowSpec):
            key = (SlidingWindowSpec, spec.block_size, spec.sliding_window)
        elif isinstance(spec, FullAttentionSpec):
            key = (FullAttentionSpec, spec.block_size)
        else:
            raise TypeError(
                f"Uyu-2 does not support KV cache spec {type(spec).__name__}"
            )
        buckets.setdefault(key, {})[layer_name] = spec

    groups: list[KVCacheGroupSpec] = []
    for specs in buckets.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(specs)
        if uniform_spec is None:
            raise ValueError("Uyu-2 KV cache specs could not be grouped by type")
        groups.append(KVCacheGroupSpec(list(specs), uniform_spec))
    return groups


def _iter_layer_specs(
    groups: Iterable[KVCacheGroupSpec],
) -> Iterable[KVCacheSpec]:
    for group in groups:
        spec = group.kv_cache_spec
        if not isinstance(spec, UniformTypeKVCacheSpecs):
            raise TypeError("Uyu-2 KV cache groups must use per-layer specs")
        yield from spec.kv_cache_specs.values()


def _group_requirements(
    vllm_config: Any,
    groups: Iterable[KVCacheGroupSpec],
) -> list[tuple[int, int]]:
    requirements: list[tuple[int, int]] = []
    for group in groups:
        spec = group.kv_cache_spec
        if not isinstance(spec, UniformTypeKVCacheSpecs):
            raise TypeError("Uyu-2 KV cache groups must use per-layer specs")
        page_bytes = sum(
            layer_spec.page_size_bytes
            for layer_spec in spec.kv_cache_specs.values()
        )
        max_blocks = max(
            (
                layer_spec.max_memory_usage_bytes(vllm_config)
                + layer_spec.page_size_bytes
                - 1
            )
            // layer_spec.page_size_bytes
            for layer_spec in spec.kv_cache_specs.values()
        )
        requirements.append((page_bytes, max_blocks))
    return requirements


def _allocate_uyu_cache(
    vllm_config: Any,
    groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    if getattr(vllm_config.cache_config, "num_gpu_blocks_override", None) is not None:
        raise ValueError("Uyu-2 retained KV cache does not support block override")

    requirements = _group_requirements(vllm_config, groups)
    bytes_per_concurrent_request = sum(
        page_bytes * max_blocks for page_bytes, max_blocks in requirements
    )
    concurrency = available_memory / bytes_per_concurrent_request
    blocks_per_group = [
        max(1, int(concurrency * max_blocks))
        for _, max_blocks in requirements
    ]

    tensors: list[KVCacheTensor] = []
    for group, num_blocks in zip(groups, blocks_per_group):
        spec = group.kv_cache_spec
        assert isinstance(spec, UniformTypeKVCacheSpecs)
        tensors.extend(
            KVCacheTensor(
                size=spec.kv_cache_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in group.layer_names
        )

    config = KVCacheConfig(
        num_blocks=math.gcd(*blocks_per_group),
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )
    config.uyu_num_blocks_per_group = blocks_per_group
    config.uyu_group_page_bytes = [page_bytes for page_bytes, _ in requirements]
    return config


def install_kv_cache_support() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from vllm.v1.core import kv_cache_manager, kv_cache_utils
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinatorNoPrefixCache
    from vllm.v1.core.single_type_kv_cache_manager import (
        CrossAttentionManager,
        get_manager_for_kv_cache_spec,
    )

    original_get_groups = kv_cache_utils.get_kv_cache_groups
    original_max_memory = kv_cache_utils._max_memory_usage_bytes_from_groups
    original_max_concurrency = (
        kv_cache_utils.get_max_concurrency_for_kv_cache_config
    )
    original_get_config = kv_cache_utils.get_kv_cache_config_from_groups
    original_get_coordinator = kv_cache_manager.get_kv_cache_coordinator

    class _CompositeBlockPool:
        def __init__(self, pools: list[BlockPool], weights: list[int]):
            self.pools = pools
            self.weights = weights
            self.pending_demands = [0] * len(pools)
            self.num_gpu_blocks = sum(pool.num_gpu_blocks for pool in pools)

        def set_pending_demands(self, demands: list[int]) -> None:
            self.pending_demands = demands

        def get_num_free_blocks(self) -> int:
            fits = all(
                demand <= pool.get_num_free_blocks()
                for demand, pool in zip(self.pending_demands, self.pools)
            )
            return 1 if fits else 0

        def get_usage(self) -> float:
            total = sum(
                pool.num_gpu_blocks * weight
                for pool, weight in zip(self.pools, self.weights)
            )
            free = sum(
                pool.get_num_free_blocks() * weight
                for pool, weight in zip(self.pools, self.weights)
            )
            return 1.0 - free / total

        def reset_prefix_cache(self) -> bool:
            return all(pool.reset_prefix_cache() for pool in self.pools)

        def evict_blocks(self, block_ids: set[int]) -> None:
            for pool in self.pools:
                valid_ids = {
                    block_id
                    for block_id in block_ids
                    if block_id < pool.num_gpu_blocks
                }
                pool.evict_blocks(valid_ids)

        def take_events(self) -> list[Any]:
            return [event for pool in self.pools for event in pool.take_events()]

    class _UyuKVCacheCoordinator(KVCacheCoordinatorNoPrefixCache):
        def __init__(
            self,
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
            pcp_world_size,
            scheduler_block_size,
            hash_block_size,
            metrics_collector=None,
        ):
            if enable_caching or use_eagle or enable_kv_cache_events:
                raise ValueError(
                    "Uyu-2 retained KV cache requires prefix caching, "
                    "EAGLE, and KV cache events to be disabled"
                )
            self.kv_cache_config = kv_cache_config
            self.max_model_len = max_model_len
            self.enable_caching = False
            self.use_eagle = False
            self.log_stats = False
            self.metrics_collector = metrics_collector
            self.scheduler_block_size = scheduler_block_size
            self.eagle_group_ids = set()
            self.retention_interval = None

            block_counts = kv_cache_config.uyu_num_blocks_per_group
            self.block_pools = [
                BlockPool(
                    num_gpu_blocks=num_blocks,
                    enable_caching=False,
                    hash_block_size=hash_block_size,
                    enable_kv_cache_events=False,
                    metrics_collector=None,
                )
                for num_blocks in block_counts
            ]
            self.block_pool = _CompositeBlockPool(
                self.block_pools, kv_cache_config.uyu_group_page_bytes
            )
            self.single_type_managers = tuple(
                get_manager_for_kv_cache_spec(
                    kv_cache_spec=(
                        next(iter(group.kv_cache_spec.kv_cache_specs.values()))
                        if isinstance(
                            group.kv_cache_spec, UniformTypeKVCacheSpecs
                        )
                        else group.kv_cache_spec
                    ),
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_model_len=max_model_len,
                    block_pool=self.block_pools[index],
                    enable_caching=False,
                    kv_cache_group_id=index,
                    dcp_world_size=dcp_world_size,
                    pcp_world_size=pcp_world_size,
                    scheduler_block_size=scheduler_block_size,
                )
                for index, group in enumerate(kv_cache_config.kv_cache_groups)
            )
            self.num_single_type_manager = len(self.single_type_managers)

        def get_num_blocks_to_allocate(
            self,
            request_id,
            num_tokens,
            new_computed_blocks,
            num_encoder_tokens,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=False,
        ) -> int:
            demands: list[int] = []
            for index, manager in enumerate(self.single_type_managers):
                if isinstance(manager, CrossAttentionManager):
                    demand = manager.get_num_blocks_to_allocate(
                        request_id,
                        num_encoder_tokens,
                        [],
                        0,
                        num_encoder_tokens,
                        apply_admission_cap=apply_admission_cap,
                    )
                else:
                    demand = manager.get_num_blocks_to_allocate(
                        request_id,
                        num_tokens,
                        new_computed_blocks[index],
                        total_computed_tokens,
                        num_tokens_main_model,
                        apply_admission_cap=apply_admission_cap,
                    )
                demands.append(demand)
            self.block_pool.set_pending_demands(demands)
            return int(any(demands))

    def get_kv_cache_groups(vllm_config, kv_cache_specs):
        if not _is_uyu_config(vllm_config):
            return original_get_groups(vllm_config, kv_cache_specs)
        if getattr(vllm_config, "kv_transfer_config", None) is not None:
            raise ValueError("Uyu-2 retained KV cache does not support KV transfer")
        vllm_config.cache_config.enable_prefix_caching = False
        return _group_uyu_specs(kv_cache_specs)

    def get_kv_cache_config(vllm_config, groups, available_memory):
        if not _is_uyu_config(vllm_config):
            return original_get_config(vllm_config, groups, available_memory)
        return _allocate_uyu_cache(vllm_config, groups, available_memory)

    def get_kv_cache_coordinator(kv_cache_config, *args, **kwargs):
        if not hasattr(kv_cache_config, "uyu_num_blocks_per_group"):
            return original_get_coordinator(kv_cache_config, *args, **kwargs)
        return _UyuKVCacheCoordinator(kv_cache_config, *args, **kwargs)

    def max_memory_usage(vllm_config, groups):
        if not _is_uyu_config(vllm_config):
            return original_max_memory(vllm_config, groups)
        return sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in _iter_layer_specs(groups)
        )

    def max_concurrency(vllm_config, kv_cache_config: KVCacheConfig):
        if not _is_uyu_config(vllm_config):
            return original_max_concurrency(vllm_config, kv_cache_config)

        blocks_per_group = kv_cache_config.uyu_num_blocks_per_group
        requirements = _group_requirements(
            vllm_config, kv_cache_config.kv_cache_groups
        )
        return min(
            num_blocks / max_blocks
            for num_blocks, (_, max_blocks) in zip(
                blocks_per_group, requirements
            )
        )

    kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups
    kv_cache_utils.get_kv_cache_config_from_groups = get_kv_cache_config
    kv_cache_utils._max_memory_usage_bytes_from_groups = max_memory_usage
    kv_cache_utils.get_max_concurrency_for_kv_cache_config = max_concurrency
    kv_cache_manager.get_kv_cache_coordinator = get_kv_cache_coordinator
    _PATCHED = True
