from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from uyu2_vllm_plugin.kv_cache import _allocate_uyu_cache, _group_uyu_specs
from uyu2_vllm_plugin.model import Uyu2VllmForCausalLM


class AttentionGeometryTest(unittest.TestCase):
    def test_uses_retained_sliding_kv_heads(self) -> None:
        model = object.__new__(Uyu2VllmForCausalLM)
        torch.nn.Module.__init__(model)
        model.parallel_config = SimpleNamespace(tensor_parallel_size=1)
        model.pp_group = SimpleNamespace(rank_in_group=0, world_size=1)
        model.text_config = SimpleNamespace(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=256,
            global_head_dim=512,
            sliding_window=1024,
            pruned_shapes={
                "layers": {
                    "0": {"num_attention_heads": 10, "num_key_value_heads": 5},
                    "1": {"num_attention_heads": 22, "num_key_value_heads": 4},
                }
            },
        )
        model.cache_config = None
        model.quant_config = None

        calls: list[dict] = []

        def record_attention(**kwargs):
            calls.append(kwargs)
            return kwargs

        with patch("uyu2_vllm_plugin.model.Attention", record_attention):
            instances = model.create_attention_instances()

        self.assertEqual(list(instances), [0, 1])
        self.assertEqual([call["num_heads"] for call in calls], [10, 32])
        self.assertEqual([call["num_kv_heads"] for call in calls], [5, 4])
        self.assertEqual([call["head_size"] for call in calls], [256, 512])

    def test_groups_different_page_sizes_by_attention_semantics(self) -> None:
        specs = {
            "0.attn": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
            ),
            "1.attn": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=4,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
            ),
            "2.attn": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
            ),
            "3.attn": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=4,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
            ),
            "4.attn": FullAttentionSpec(
                block_size=16,
                num_kv_heads=4,
                head_size=512,
                dtype=torch.bfloat16,
            ),
            "5.attn": FullAttentionSpec(
                block_size=16,
                num_kv_heads=4,
                head_size=512,
                dtype=torch.bfloat16,
            ),
        }

        groups = _group_uyu_specs(specs)

        self.assertEqual(len(groups), 2)
        self.assertTrue(
            all(
                isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                for group in groups
            )
        )
        self.assertEqual([len(group.layer_names) for group in groups], [4, 2])
        grouped_names = {name for group in groups for name in group.layer_names}
        self.assertEqual(grouped_names, set(specs))
        self.assertNotEqual(
            specs["0.attn"].page_size_bytes,
            specs["1.attn"].page_size_bytes,
        )

    def test_allocates_one_exact_sized_tensor_per_layer(self) -> None:
        specs = {
            "0.attn": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=5,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
            ),
            "1.attn": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
            ),
            "2.attn": FullAttentionSpec(
                block_size=16,
                num_kv_heads=4,
                head_size=512,
                dtype=torch.bfloat16,
            ),
        }
        groups = _group_uyu_specs(specs)
        config = SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks_override=None),
            model_config=SimpleNamespace(max_model_len=2048),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
        )
        available_memory = sum(
            spec.max_memory_usage_bytes(config) for spec in specs.values()
        ) * 2

        cache_config = _allocate_uyu_cache(config, groups, available_memory)

        self.assertEqual(cache_config.uyu_num_blocks_per_group, [258, 256])
        self.assertEqual(len(cache_config.kv_cache_tensors), len(specs))
        tensor_by_layer = {
            tensor.shared_by[0]: tensor
            for tensor in cache_config.kv_cache_tensors
        }
        group_blocks = {
            layer_name: cache_config.uyu_num_blocks_per_group[group_index]
            for group_index, group in enumerate(groups)
            for layer_name in group.layer_names
        }
        for layer_name, spec in specs.items():
            self.assertEqual(
                tensor_by_layer[layer_name].size,
                spec.page_size_bytes * group_blocks[layer_name],
            )


if __name__ == "__main__":
    unittest.main()
