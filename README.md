# vllm-uyu2

`vllm-uyu2` registers the custom `Uyu2ForCausalLM` architecture with vLLM.
It is the recommended serving integration for
[`mente-ai/uyu-2-28B`](https://huggingface.co/mente-ai/uyu-2-28B), a text-only
Korean and English role-playing model with non-uniform structured pruning.

## Compatibility

- Python 3.10-3.12
- vLLM main/nightly with packed KV cache support (`KVCacheTensor.offset` and
  `block_stride`)
- Transformers 5.13.x
- Tensor parallel size 1 and pipeline parallel size 1
- BF16-capable CUDA GPU
- `TRITON_ATTN` or another block-stride-aware attention backend

The version bounds are intentional because this plugin uses vLLM's model and
attention extension interfaces, which can change between releases.

## Install

```bash
uv pip install -U vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly

git clone https://github.com/menteai/vllm-uyu2.git
uv pip install --no-deps ./vllm-uyu2
```

For development:

```bash
uv pip install --no-deps -e .
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

The plugin packs layers with the same cache lifetime into 256 KiB logical cache
groups. The 50 sliding-window layers form 41 groups and the 10 full-attention
layers form 5 groups. All 46 managers use vLLM's standard KV cache coordinator
and one physical `BlockPool`.

Each logical block is stored in one packed backing allocation. vLLM's
`KVCacheTensor.offset` and `block_stride` fields expose a correctly sized view
to each layer, so retained head counts remain physical rather than transient
runtime masks. The packed layout reserves 736 16-KiB units for 724 units of KV
data, adding 192 KiB per complete set of group pages (approximately 1.66%). At
a 2,048-token context with a 1,024-token sliding window, the combined BF16 K+V
allocation is approximately 0.797 GiB.

The release path was validated with vLLM main commit `04d553f` using CUDA 13
nightly native extensions from commit `0923879`. With a 2,048-token model
length and `--gpu-memory-utilization 0.80`, vLLM reported 61,081 GPU KV cache
tokens and 29.83x maximum concurrency on the validation system. Capacity is
hardware- and configuration-dependent.

## Verify installation

```bash
VLLM_PLUGINS=uyu2 python -c \
  'from vllm import ModelRegistry; from vllm.plugins import load_general_plugins; load_general_plugins(); print("Uyu2ForCausalLM" in ModelRegistry.get_supported_archs())'
```

The command should print `True`.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
