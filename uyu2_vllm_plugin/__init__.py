from __future__ import annotations


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Uyu2ForCausalLM",
        "uyu2_vllm_plugin.model:Uyu2VllmForCausalLM",
    )
