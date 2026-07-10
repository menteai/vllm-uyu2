from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from uyu2_vllm_plugin.model import Uyu2VllmForCausalLM


class AttentionGeometryTest(unittest.TestCase):
    def test_preserves_base_gqa_kv_heads(self) -> None:
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
        self.assertEqual([call["num_heads"] for call in calls], [32, 32])
        self.assertEqual([call["num_kv_heads"] for call in calls], [16, 4])
        self.assertEqual([call["head_size"] for call in calls], [256, 512])


if __name__ == "__main__":
    unittest.main()
