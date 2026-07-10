# vllm-uyu2

`vllm-uyu2` registers the custom `Uyu2ForCausalLM` architecture with vLLM.
It is the recommended serving integration for
[`mente-ai/uyu-2-28b`](https://huggingface.co/mente-ai/uyu-2-28b), a text-only
Korean and English role-playing model with non-uniform structured pruning.

## Compatibility

- Python 3.10-3.12
- vLLM 0.23.x
- Transformers 5.13.x
- Tensor parallel size 1
- BF16-capable CUDA GPU

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

## Serve uyu-2-28b

```bash
VLLM_PLUGINS=uyu2 vllm serve mente-ai/uyu-2-28b \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --attention-backend TRITON_ATTN \
  --enforce-eager \
  --served-model-name uyu-2-28b
```

The equivalent reusable launcher is available at `scripts/serve_uyu2.sh`.
Override `MODEL`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `HOST`, `PORT`, or
`SERVED_MODEL_NAME` through environment variables.

## How it works

The model contains different retained attention shapes across layers. vLLM
0.23 cannot allocate those heterogeneous KV pages directly, so the remote model
code expands each layer to 32 Q/KV cache slots. This plugin creates matching
vLLM attention instances and preserves vLLM's paged KV cache and continuous
batching behavior.

The padding affects only runtime attention and KV cache tensors. It does not
restore pruned model weights or increase the checkpoint size. A future custom
allocator/backend could store only retained KV heads, but that is outside this
plugin's current scope.

## Verify installation

```bash
VLLM_PLUGINS=uyu2 python -c \
  'from vllm import ModelRegistry; from vllm.plugins import load_general_plugins; load_general_plugins(); print("Uyu2ForCausalLM" in ModelRegistry.get_supported_archs())'
```

The command should print `True`.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
