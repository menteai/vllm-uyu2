from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import fields
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


def _semantic_key(spec: KVCacheSpec) -> tuple[Any, ...]:
    if isinstance(spec, SlidingWindowSpec):
        return SlidingWindowSpec, spec.block_size, spec.sliding_window
    if isinstance(spec, FullAttentionSpec):
        return FullAttentionSpec, spec.block_size
    raise TypeError(f"Uyu-2 does not support KV cache spec {type(spec).__name__}")


def _pack_specs(
    specs: dict[str, KVCacheSpec], block_stride: int
) -> list[dict[str, KVCacheSpec]]:
    bins: list[dict[str, KVCacheSpec]] = []
    used_bytes: list[int] = []

    for layer_name, spec in sorted(
        specs.items(), key=lambda item: (-item[1].page_size_bytes, item[0])
    ):
        for index, used in enumerate(used_bytes):
            if used + spec.page_size_bytes <= block_stride:
                bins[index][layer_name] = spec
                used_bytes[index] += spec.page_size_bytes
                break
        else:
            bins.append({layer_name: spec})
            used_bytes.append(spec.page_size_bytes)

    return bins


def _group_uyu_specs(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    if not kv_cache_specs:
        return []

    block_stride = max(spec.page_size_bytes for spec in kv_cache_specs.values())
    semantic_buckets: OrderedDict[tuple[Any, ...], dict[str, KVCacheSpec]] = (
        OrderedDict()
    )
    for layer_name, spec in kv_cache_specs.items():
        semantic_buckets.setdefault(_semantic_key(spec), {})[layer_name] = spec

    groups: list[KVCacheGroupSpec] = []
    for specs in semantic_buckets.values():
        for packed_specs in _pack_specs(specs, block_stride):
            uniform_spec = UniformTypeKVCacheSpecs.from_specs(packed_specs)
            if uniform_spec is None:
                raise ValueError("Uyu-2 cache group has incompatible semantics")
            groups.append(KVCacheGroupSpec(list(packed_specs), uniform_spec))

    return groups


def _group_page_bytes(group: KVCacheGroupSpec) -> int:
    spec = group.kv_cache_spec
    if not isinstance(spec, UniformTypeKVCacheSpecs):
        raise TypeError("Uyu-2 KV cache groups must use per-layer specs")
    return sum(layer.page_size_bytes for layer in spec.kv_cache_specs.values())


def _packed_block_stride(groups: Iterable[KVCacheGroupSpec]) -> int:
    return max((_group_page_bytes(group) for group in groups), default=0)


def _group_max_blocks(vllm_config: Any, group: KVCacheGroupSpec) -> int:
    spec = group.kv_cache_spec
    if not isinstance(spec, UniformTypeKVCacheSpecs):
        raise TypeError("Uyu-2 KV cache groups must use per-layer specs")
    return max(
        (
            layer.max_memory_usage_bytes(vllm_config)
            + layer.page_size_bytes
            - 1
        )
        // layer.page_size_bytes
        for layer in spec.kv_cache_specs.values()
    )


def _blocks_per_request(
    vllm_config: Any, groups: Iterable[KVCacheGroupSpec]
) -> int:
    return sum(_group_max_blocks(vllm_config, group) for group in groups)


def _allocate_uyu_cache(
    vllm_config: Any,
    groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    from vllm.v1.core.kv_cache_utils import may_override_num_blocks

    block_stride = _packed_block_stride(groups)
    if block_stride == 0:
        return KVCacheConfig(1, [], groups)

    num_blocks = may_override_num_blocks(
        vllm_config, available_memory // block_stride
    )
    total_size = block_stride * num_blocks
    tensors: list[KVCacheTensor] = []

    for group in groups:
        spec = group.kv_cache_spec
        assert isinstance(spec, UniformTypeKVCacheSpecs)
        offset = 0
        for layer_name in group.layer_names:
            layer_spec = spec.kv_cache_specs[layer_name]
            tensors.append(
                KVCacheTensor(
                    size=total_size,
                    shared_by=[layer_name],
                    offset=offset,
                    block_stride=block_stride,
                )
            )
            offset += layer_spec.page_size_bytes
        if offset > block_stride:
            raise AssertionError("Uyu-2 packed group exceeds its block stride")

    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )
    config.uyu_block_stride = block_stride
    config.uyu_blocks_per_request = _blocks_per_request(vllm_config, groups)
    return config


def install_kv_cache_support() -> None:
    global _PATCHED
    if _PATCHED:
        return

    tensor_fields = {field.name for field in fields(KVCacheTensor)}
    if not {"offset", "block_stride"}.issubset(tensor_fields):
        raise RuntimeError(
            "vllm-uyu2 requires a vLLM build with packed KV cache support"
        )

    from vllm.v1.core import kv_cache_utils

    if getattr(kv_cache_utils, "_uyu2_packed_support_installed", False):
        _PATCHED = True
        return

    original_get_groups = kv_cache_utils.get_kv_cache_groups
    original_get_config = kv_cache_utils.get_kv_cache_config_from_groups
    original_max_memory = kv_cache_utils._max_memory_usage_bytes_from_groups
    original_max_concurrency = (
        kv_cache_utils.get_max_concurrency_for_kv_cache_config
    )
    original_pool_bytes = kv_cache_utils._pool_bytes_per_block

    def get_kv_cache_groups(vllm_config, kv_cache_specs):
        if not _is_uyu_config(vllm_config):
            return original_get_groups(vllm_config, kv_cache_specs)
        parallel = vllm_config.parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
        ):
            raise ValueError("Uyu-2 packed KV cache currently requires TP=1 and PP=1")
        return _group_uyu_specs(kv_cache_specs)

    def get_kv_cache_config(vllm_config, groups, available_memory):
        if not _is_uyu_config(vllm_config):
            return original_get_config(vllm_config, groups, available_memory)
        return _allocate_uyu_cache(vllm_config, groups, available_memory)

    def max_memory_usage(vllm_config, groups):
        if not _is_uyu_config(vllm_config):
            return original_max_memory(vllm_config, groups)
        return _packed_block_stride(groups) * _blocks_per_request(
            vllm_config, groups
        )

    def max_concurrency(vllm_config, kv_cache_config: KVCacheConfig):
        if not _is_uyu_config(vllm_config):
            return original_max_concurrency(vllm_config, kv_cache_config)
        return kv_cache_config.num_blocks / kv_cache_config.uyu_blocks_per_request

    def pool_bytes(vllm_config, groups):
        if not _is_uyu_config(vllm_config):
            return original_pool_bytes(vllm_config, groups)
        return _packed_block_stride(groups)

    kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups
    kv_cache_utils.get_kv_cache_config_from_groups = get_kv_cache_config
    kv_cache_utils._max_memory_usage_bytes_from_groups = max_memory_usage
    kv_cache_utils.get_max_concurrency_for_kv_cache_config = max_concurrency
    kv_cache_utils._pool_bytes_per_block = pool_bytes
    kv_cache_utils._uyu2_packed_support_installed = True
    _PATCHED = True
