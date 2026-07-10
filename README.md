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
code restores the original indexed attention envelopes: 32 Q slots in every
layer, 16 KV slots in sliding-window layers, and 4 KV slots in full-attention
layers. Retained projections are scattered into their original slots while
removed slots remain empty. This preserves the base GQA sharing ratios instead
of duplicating K/V for every Q head.

The Q padding is a transient attention tensor. KV cache pages use the 16/4 GQA
envelopes, reducing the theoretical BF16 K+V payload at a 2,048-token context
with a 1,024-token sliding window from approximately 2.813 GiB in plugin 0.1.0
to 0.938 GiB. An ideal heterogeneous retained-KV allocator would use
approximately 0.785 GiB. The padding does not restore pruned model weights or
increase the checkpoint size.

With `--max-model-len 2048` and `--gpu-memory-utilization 0.80` on the release
validation system, vLLM reported 50,212 GPU KV cache tokens and 24.52x maximum
concurrency, compared with 19,440 tokens and 9.49x for plugin 0.1.0. The exact
capacity depends on available GPU memory and serving configuration.

## Verify installation

```bash
VLLM_PLUGINS=uyu2 python -c \
  'from vllm import ModelRegistry; from vllm.plugins import load_general_plugins; load_general_plugins(); print("Uyu2ForCausalLM" in ModelRegistry.get_supported_archs())'
```

The command should print `True`.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
