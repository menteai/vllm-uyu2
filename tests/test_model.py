from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from uyu2_vllm_plugin.kv_cache import (
    _allocate_uyu_cache,
    _group_page_bytes,
    _group_uyu_specs,
    _packed_block_stride,
)
from uyu2_vllm_plugin.model import Uyu2VllmForCausalLM


class AttentionGeometryTest(unittest.TestCase):
    def test_uses_retained_sliding_kv_heads(self) -> None:
        model = object.__new__(Uyu2VllmForCausalLM)
        torch.nn.Module.__init__(model)
        model.parallel_config = SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        )
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
        self.assertTrue(
            all(call["attn_backend"] is TritonAttentionBackend for call in calls)
        )

    def test_packs_real_uyu_geometry_with_minimal_padding(self) -> None:
        retained_sliding_heads = (
            [16] * 34
            + [15] * 2
            + [11]
            + [9] * 2
            + [7]
            + [6]
            + [5] * 3
            + [3] * 3
            + [2]
            + [1] * 2
        )
        specs = {
            f"{index}.attn": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=heads,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
            )
            for index, heads in enumerate(retained_sliding_heads)
        }
        specs.update(
            {
                f"{index}.attn": FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=512,
                    dtype=torch.bfloat16,
                )
                for index in range(50, 60)
            }
        )

        groups = _group_uyu_specs(specs)

        self.assertEqual(len(groups), 46)
        self.assertTrue(
            all(
                isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                for group in groups
            )
        )
        grouped_names = {name for group in groups for name in group.layer_names}
        self.assertEqual(grouped_names, set(specs))
        self.assertEqual(_packed_block_stride(groups), 256 * 1024)
        self.assertEqual(
            sum(_group_page_bytes(group) for group in groups),
            724 * 16 * 1024,
        )
        self.assertEqual(46 * 256 * 1024 - 724 * 16 * 1024, 192 * 1024)

    def test_allocates_one_packed_backing_with_layer_offsets(self) -> None:
        specs = {
            "0.attn": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=16,
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
            "2.attn": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=5,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
            ),
            "3.attn": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=3,
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
        config = SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks_override=None),
            model_config=SimpleNamespace(max_model_len=2048),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            max_in_flight_tokens=2048,
        )
        block_stride = 256 * 1024
        available_memory = block_stride * 100

        cache_config = _allocate_uyu_cache(config, groups, available_memory)

        self.assertEqual(cache_config.num_blocks, 100)
        self.assertGreater(cache_config.uyu_blocks_per_request, 0)
        self.assertEqual(cache_config.uyu_block_stride, block_stride)
        self.assertEqual(len(groups), 3)
        self.assertEqual(len(cache_config.kv_cache_tensors), len(specs))
        self.assertEqual(
            {tensor.size for tensor in cache_config.kv_cache_tensors},
            {available_memory},
        )
        self.assertEqual(
            {tensor.block_stride for tensor in cache_config.kv_cache_tensors},
            {block_stride},
        )
        tensor_by_layer = {
            tensor.shared_by[0]: tensor
            for tensor in cache_config.kv_cache_tensors
        }
        for group in groups:
            intervals = sorted(
                (
                    tensor_by_layer[layer_name].offset,
                    tensor_by_layer[layer_name].offset
                    + specs[layer_name].page_size_bytes,
                )
                for layer_name in group.layer_names
            )
            self.assertLessEqual(intervals[-1][1], block_stride)
            self.assertTrue(
                all(
                    left[1] <= right[0]
                    for left, right in zip(intervals, intervals[1:])
                )
            )


if __name__ == "__main__":
    unittest.main()
