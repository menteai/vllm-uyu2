#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-mente-ai/uyu-2-28B}"

export VLLM_PLUGINS=uyu2

exec vllm serve "$MODEL" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-2048}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --enforce-eager \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8001}" \
  --served-model-name "${SERVED_MODEL_NAME:-uyu-2-28b}"
