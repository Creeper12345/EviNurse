#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
Start any OpenAI-compatible model endpoint, then evaluate with:

python scripts/evaluate_mcq.py \
  --input data/samples/mcq_sample.json \
  --output outputs/mcq_sample_predictions.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model EviNurse \
  --api-key EMPTY

For vLLM, use:

MODEL_PATH=/path/to/model SERVED_MODEL_NAME=EviNurse bash scripts/run_vllm_server.sh
EOF

