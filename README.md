# EviNurse

EviNurse is a domain-specific large language model system designed to support evidence-informed nursing practice. It was developed on Qwen3-32B using supervised fine-tuning and retrieval-augmented generation, with high-quality nursing evidence organized according to the 5S evidence pyramid.

This repository provides reproducible materials for the automated multiple-choice question (MCQ) evaluation and model-serving interfaces reported in the EviNurse manuscript:

- Benchmark access and sample data.
- OpenAI-compatible MCQ evaluation code.
- vLLM model-serving scripts for evaluation and deployment.
- A de-identified OpenAI-compatible RAG API example that preserves the source-level and passage-level retrieval interface.
- Environment and dependency files.

The included RAG API is a de-identified interface example. It preserves the request/response shape and retrieval-augmented generation flow, but cannot reproduce the full RAG system unless connected to a compatible external retrieval service and knowledge base.

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
├── .env.example
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── requirements-vllm.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For vLLM serving or the optional RAG API server, install the GPU serving dependencies in the same environment:

```bash
pip install -r requirements-vllm.txt
```

Copy `.env.example` if you want to manage model-serving settings through environment variables:

```bash
cp .env.example .env
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

### Hardware Notes

EviNurse-32B is a 32B-parameter model. Full-precision or bfloat16 serving generally requires multi-GPU inference or another memory-saving deployment strategy. The example scripts support tensor parallelism through `TENSOR_PARALLEL_SIZE`.

Recommended starting points:

- Use multiple high-memory CUDA GPUs for the full 32B model.
- Increase `TENSOR_PARALLEL_SIZE` to match the number of GPUs used by vLLM.
- Reduce `MAX_MODEL_LEN` if GPU memory is insufficient.
- Quantized or otherwise optimized deployments may have different memory requirements and are not configured in this repository.

Example:

```bash
MODEL_PATH=Agnania/EviNurse-32B \
SERVED_MODEL_NAME=EviNurse \
TENSOR_PARALLEL_SIZE=4 \
MAX_MODEL_LEN=4096 \
bash scripts/run_vllm_server.sh
```

## RAG API

The manuscript system used retrieval-augmented generation for evidence-based nursing responses. User queries were rewritten into retrieval-oriented representations, then evidence was retrieved through a dual-stage strategy: source-level retrieval followed by passage-level retrieval within shortlisted sources. Retrieved evidence was filtered using semantic relevance, source characteristics, and suitability for the target question, with higher-level evidence prioritized according to the 5S evidence pyramid.

This repository includes a de-identified OpenAI-compatible RAG API example at `server/rag_openai_api.py`. It is provided to document and test the serving interface.

The RAG API keeps the public serving interface and prompt construction logic while externalizing deployment-specific details:

- `MODEL_PATH`: local model path or Hugging Face model id.
- `RAG_BASE_URL`: URL of a retrieval service.
- `RAG_MODE`: `dual` or `single`, default `dual`.
- `RAG_SOURCE_ENDPOINT`: optional source-level retrieval endpoint, default `/retrieve_sources`.
- `RAG_PASSAGE_ENDPOINT`: optional passage-level retrieval endpoint, default `/retrieve_passages`.
- `RAG_ENDPOINT`: single-stage fallback endpoint, default `/getReference`.

Start the RAG API:

```bash
MODEL_PATH=Agnania/EviNurse-32B \
RAG_BASE_URL=http://127.0.0.1:50002 \
RAG_MODE=dual \
SERVED_MODEL_NAME=EviNurse \
bash scripts/run_rag_api_server.sh
```

The underlying knowledge base, private documents, server addresses, and deployment credentials are not included in this release. If a deployment only exposes a single retrieval endpoint, set `RAG_MODE=single`.

## License

This code repository is released under the Apache License 2.0. See [LICENSE](LICENSE).
