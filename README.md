# EviNurse

EviNurse is a domain-specific large language model framework for evidence-based nursing. It was developed to improve access to high-quality nursing evidence and to support evidence-informed nursing practice through supervised fine-tuning and retrieval-augmented generation.

This repository provides reproducible materials for the automated multiple-choice question (MCQ) evaluation and model-serving interfaces reported in the EviNurse manuscript:

- Benchmark access and sample data.
- OpenAI-compatible MCQ evaluation code.
- vLLM model-serving scripts for evaluation and deployment.
- A de-identified OpenAI-compatible RAG API example.
- Environment and dependency files.

## Resources

| Resource | Link |
| --- | --- |
| NursData-MCQ benchmark | [NursData-MCQ](https://huggingface.co/datasets/Agnania/NursData-MCQ) |
| Project repository | [EviNurse](https://github.com/Creeper12345/EviNurse) |
| EviNurse model | [EviNurse-32B](https://huggingface.co/Agnania/EviNurse-32B) |

## Repository Layout

```text
.
├── configs/
│   ├── eval_config.example.yaml
│   └── model_server.example.yaml
├── data/
│   └── samples/
│       └── mcq_sample.json
├── docs/
│   ├── benchmark.md
│   └── reproducibility.md
├── scripts/
│   ├── download_dataset.py
│   ├── evaluate_mcq.py
│   ├── evaluate_mcq_with_context.py
│   ├── validate_dataset.py
│   ├── run_eval_server.sh
│   ├── run_rag_api_server.sh
│   └── run_vllm_server.sh
├── server/
│   └── rag_openai_api.py
├── src/
│   └── evinurse_eval/
│       ├── __init__.py
│       ├── answer_extraction.py
│       ├── io.py
│       └── metrics.py
├── LICENSE
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download the Benchmark

Download the complete benchmark from Hugging Face:

```bash
python scripts/download_dataset.py \
  --repo-id Agnania/NursData-MCQ \
  --output-dir data/NursData-MCQ
```

Alternatively:

```bash
hf download Agnania/NursData-MCQ \
  --repo-type dataset \
  --local-dir data/NursData-MCQ
```

The repository includes only a small placeholder sample at `data/samples/mcq_sample.json`. Use the Hugging Face dataset for full evaluation.

Validate the downloaded benchmark:

```bash
python scripts/validate_dataset.py \
  --input data/NursData-MCQ/evinurse_automated_eval_3438.json
```

## Run MCQ Evaluation

The evaluator expects an OpenAI-compatible chat completion endpoint, such as vLLM, SGLang, or another local/remote model server.

Start the vLLM server with evaluation defaults:

```bash
MODEL_PATH=Agnania/EviNurse-32B \
SERVED_MODEL_NAME=EviNurse \
bash scripts/run_eval_server.sh
```

Run on the sample file:

```bash
python scripts/evaluate_mcq.py \
  --input data/samples/mcq_sample.json \
  --output outputs/mcq_sample_predictions.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model EviNurse \
  --api-key EMPTY
```

Run on the full Hugging Face benchmark after download:

```bash
python scripts/evaluate_mcq.py \
  --input data/NursData-MCQ/evinurse_automated_eval_3438.json \
  --output outputs/nursdata_mcq_predictions.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model EviNurse \
  --api-key EMPTY
```

The script reports exact-match MCQ accuracy based on the final extracted option letter.

For benchmark records that include retrieved evidence in a `context` field, use the context-aware evaluator. It follows the same local evaluation flow used in the project: select the top-k context snippets, include them in the prompt, call an OpenAI-compatible endpoint, extract the option letter, and write incremental JSON results.

```bash
python scripts/evaluate_mcq_with_context.py \
  --input data/samples/mcq_sample.json \
  --output outputs/mcq_sample_context_predictions.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model EviNurse \
  --api-key EMPTY \
  --top-k-context 5
```

## Start a Model Server

For a standard vLLM OpenAI-compatible server:

```bash
MODEL_PATH=Agnania/EviNurse-32B \
SERVED_MODEL_NAME=EviNurse \
bash scripts/run_vllm_server.sh
```

`scripts/run_eval_server.sh` is a thin wrapper around the same vLLM server with defaults used by the MCQ evaluation scripts. It is not a separate model implementation.

## RAG API

The manuscript system also used retrieval-augmented generation for evidence-based nursing responses. This repository includes a de-identified OpenAI-compatible RAG API example at `server/rag_openai_api.py`.

The RAG API keeps the public serving interface and prompt construction logic while externalizing deployment-specific details:

- `MODEL_PATH`: local model path or Hugging Face model id.
- `RAG_BASE_URL`: URL of a retrieval service returning a list of objects with `content` and optional `doc_name`.
- `RAG_ENDPOINT`: retrieval endpoint path, default `/getReference`.

Start the RAG API:

```bash
MODEL_PATH=Agnania/EviNurse-32B \
RAG_BASE_URL=http://127.0.0.1:50002 \
SERVED_MODEL_NAME=EviNurse \
bash scripts/run_rag_api_server.sh
```

The underlying knowledge base, private documents, server addresses, and deployment credentials are not included in this release.

## Why This Release

Nursing-domain AI remains understudied, and reproducible benchmarks are needed to make model comparisons useful to the field. Releasing NursData-MCQ and the evaluation code enables readers to:

- Reproduce the automated evaluation setting.
- Compare general-purpose and nursing-domain models on the same MCQ benchmark.
- Test future nursing AI systems under a transparent evaluation protocol.

## License

This code repository is released under the Apache License 2.0. See [LICENSE](LICENSE).
