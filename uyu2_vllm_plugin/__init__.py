from __future__ import annotations


def register() -> None:
    from vllm import ModelRegistry

    from uyu2_vllm_plugin.kv_cache import install_kv_cache_support

    install_kv_cache_support()

    ModelRegistry.register_model(
        "Uyu2ForCausalLM",
        "uyu2_vllm_plugin.model:Uyu2VllmForCausalLM",
    )
