# Reproducibility Notes

## Automated MCQ Evaluation

These commands reproduce the public automated MCQ evaluation workflow. They do not reproduce private RAG retrieval, because the evidence knowledge base and retrieval service are not included in this release.

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If serving EviNurse-32B locally with vLLM:

```bash
pip install -r requirements-vllm.txt
```

### 2. Download Data

```bash
python scripts/download_dataset.py \
  --repo-id Agnania/NursData-MCQ \
  --output-dir data/NursData-MCQ
```

Validate the downloaded file:

```bash
python scripts/validate_dataset.py \
  --input data/NursData-MCQ/evinurse_automated_eval_3438.json
```

### 3. Start Model

Start the released model with vLLM:

```bash
MODEL_PATH=Agnania/EviNurse-32B \
SERVED_MODEL_NAME=EviNurse \
TENSOR_PARALLEL_SIZE=4 \
MAX_MODEL_LEN=4096 \
bash scripts/run_eval_server.sh
```

Any OpenAI-compatible endpoint can also be used if it exposes `/v1/chat/completions`.

### 4. Run Sample Evaluation

```bash
python scripts/evaluate_mcq.py \
  --input data/samples/mcq_sample.json \
  --output outputs/mcq_sample_predictions.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model EviNurse \
  --api-key EMPTY
```

### 5. Run Full Benchmark

```bash
python scripts/evaluate_mcq.py \
  --input data/NursData-MCQ/evinurse_automated_eval_3438.json \
  --output outputs/nursdata_mcq_predictions.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model EviNurse \
  --api-key EMPTY
```

### 6. Expected Output

The evaluator prints exact-match accuracy:

```text
Accuracy: <accuracy> (<correct>/3438)
Saved predictions to: outputs/nursdata_mcq_predictions.json
```

The output JSON is a list of records containing the original MCQ fields plus:

- `model_output`: raw model response.
- `predicted_answer`: extracted option letter.
- `is_correct`: exact-match result against `answer`.
- `error`: exception text if the request failed, otherwise `null`.

## Model Availability

The EviNurse model is released separately at:

https://huggingface.co/Agnania/EviNurse-32B

## Recommended Reporting

For each evaluated model, report:

- Model name and version.
- Endpoint framework, such as vLLM or SGLang.
- Prompt template.
- Decoding parameters.
- Correct count, total count, and accuracy.
- Prediction file or checksum.
