#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local model path or Hugging Face model id.}"

export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-EviNurse}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-58003}"
export RAG_BASE_URL="${RAG_BASE_URL:-http://127.0.0.1:50002}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"

uvicorn server.rag_openai_api:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers 1
