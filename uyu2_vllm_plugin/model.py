from __future__ import annotations

from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.transformers import TransformersForCausalLM
from vllm.v1.attention.backend import AttentionType


class Uyu2VllmForCausalLM(TransformersForCausalLM):
    """Transformers wrapper using a unified 32-head KV cache geometry."""

    def create_attention_instances(self) -> dict[int, Attention]:
        if self.parallel_config.tensor_parallel_size != 1:
            raise ValueError(
                "Uyu2VllmForCausalLM currently supports tensor parallel size 1"
            )

        config = self.text_config
        pp_rank = self.pp_group.rank_in_group
        pp_size = self.pp_group.world_size
        start, end = get_pp_indices(config.num_hidden_layers, pp_rank, pp_size)

        attention_instances: dict[int, Attention] = {}
        for layer_idx in range(start, end):
            is_sliding = config.layer_types[layer_idx] == "sliding_attention"
            head_size = int(
                config.head_dim
                if is_sliding
                else (config.global_head_dim or config.head_dim)
            )

            attention_instances[layer_idx] = Attention(
                num_heads=32,
                head_size=head_size,
                scale=1.0,
                num_kv_heads=32,
                cache_config=self.cache_config,
                quant_config=self.quant_config,
                logits_soft_cap=getattr(config, "attn_logit_softcapping", None),
                per_layer_sliding_window=(
                    int(config.sliding_window) if is_sliding else None
                ),
                prefix=f"{layer_idx}.attn",
                attn_type=AttentionType.DECODER,
            )

        return attention_instances
