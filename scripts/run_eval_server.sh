#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the vLLM OpenAI-compatible server with defaults used by
# the MCQ evaluation scripts. This starts the same model as run_vllm_server.sh.

export MODEL_PATH="${MODEL_PATH:-Agnania/EviNurse-32B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-EviNurse}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export TEMPERATURE="${TEMPERATURE:-0}"

bash "$(dirname "$0")/run_vllm_server.sh"
