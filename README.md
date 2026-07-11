# vllm-uyu2

`vllm-uyu2` registers the custom `Uyu2ForCausalLM` architecture with vLLM.
It is the recommended serving integration for
[`mente-ai/uyu-2-28B`](https://huggingface.co/mente-ai/uyu-2-28B), a text-only
Korean and English role-playing model with non-uniform structured pruning.

## Compatibility

- Python 3.10-3.12
- vLLM 0.23.x
- Transformers 5.13.x
- Tensor parallel size 1
- BF16-capable CUDA GPU
- Prefix caching disabled
- EAGLE/speculative cache and KV transfer unsupported

The version bounds are intentional because this plugin uses vLLM's model and
attention extension interfaces, which can change between releases.

## Install

```bash
git clone https://github.com/menteai/vllm-uyu2.git
pip install ./vllm-uyu2
```

For development:

```bash
pip install -e .
```

## Serve uyu-2-28B

```bash
VLLM_PLUGINS=uyu2 vllm serve mente-ai/uyu-2-28B \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --attention-backend TRITON_ATTN \
  --enforce-eager \
  --no-enable-prefix-caching \
  --served-model-name uyu-2-28b
```

The equivalent reusable launcher is available at `scripts/serve_uyu2.sh`.
Override `MODEL`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `HOST`, `PORT`, or
`SERVED_MODEL_NAME` through environment variables.

## How it works

Uyu-2 stores the retained KV heads of each layer directly. Sliding-window
layers keep complete 2Q:1KV pruning groups, so both Q and KV tensors remain
compact. Full-attention layers retain all four shared KV heads; their Q outputs
are restored to the original 32 positions only for the attention operation so
the original Q-to-KV mapping remains valid.

The plugin groups full-attention and sliding-window layers by cache semantics
and gives each group an independently sized block pool. Layers in a group share
logical block IDs and token offsets, while vLLM creates a separate physical KV
tensor sized from each layer's retained `num_kv_heads`.

At a 2,048-token context with a 1,024-token sliding window, the theoretical
combined BF16 K+V payload is approximately 0.785 GiB. Actual capacity also
depends on block allocation, available GPU memory, and serving configuration.

## Verify installation

```bash
VLLM_PLUGINS=uyu2 python -c \
  'from vllm import ModelRegistry; from vllm.plugins import load_general_plugins; load_general_plugins(); print("Uyu2ForCausalLM" in ModelRegistry.get_supported_archs())'
```

The command should print `True`.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
