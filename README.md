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
│   ├── run_dual_stage_retrieval_server.sh
│   ├── run_rag_api_server.sh
│   └── run_vllm_server.sh
├── server/
│   ├── dual_stage_retrieval_api.py
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

The manuscript system used retrieval-augmented generation for evidence-based nursing responses. User queries were rewritten into retrieval-oriented representations, then evidence was retrieved through a three-step strategy: summary-level source retrieval, chunk-level passage retrieval within shortlisted sources, and evidence-type supplementation. Retrieved evidence was filtered using semantic relevance, source characteristics, evidence category, publication year, and suitability for the target question, with higher-level evidence prioritized according to the 5S evidence pyramid.

This repository includes two de-identified RAG-related API examples:

- `server/dual_stage_retrieval_api.py`: a retrieval service example that implements summary-level source retrieval, chunk-level passage retrieval within shortlisted sources, metadata-aware scoring, reranking, and evidence-type supplementation.
- `server/rag_openai_api.py`: an OpenAI-compatible generation API that rewrites user queries, calls a retrieval service, constructs the evidence-grounded prompt, and returns generated answers with context metadata.

The generation API keeps the public serving interface and prompt construction logic while externalizing deployment-specific details:

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

To run the dual-stage retrieval service example, connect it to compatible summary-level and chunk-level vector collections built from evidence sources to which you have access:

```bash
RAG_EMBEDDING_MODEL=BAAI/bge-m3 \
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
MILVUS_HOST=127.0.0.1 \
MILVUS_PORT=19530 \
MILVUS_DB_NAME=nursingdb \
SUMMARY_COLLECTION=nursing_summary \
CHUNK_COLLECTION=nursing_article \
bash scripts/run_dual_stage_retrieval_server.sh
```

The summary-level collection should contain `embedding_vector`, `source_id` or `doc_name`, `doc_name`, `summary_text`, and `domain_category` fields. The chunk-level collection should contain `embedding_vector`, `source_id` or `doc_name`, `doc_name`, `chunk_text`, and `domain_category` fields. Equivalent field names can be configured through environment variables. The reported manuscript experiments used document preprocessing, recursive chunking, source-level summarization, BGE-M3 embeddings, vector retrieval, BGE-reranker-v2-m3 reranking, metadata-aware scoring, and evidence-type supplementation based on the 5S evidence pyramid. If only a chunk-level collection is available, the service can still be used with `strategy=single_chunk`; summary-level retrieval is required only for `dual_v1`, `dual_v2`, and `dual_v3`.

The retrieval service supports the following strategies:

| Strategy | Description |
| --- | --- |
| `single_chunk` | Baseline retrieval over the chunk-level knowledge base only, followed by optional reranking and top-k selection. This does not use summary-level source retrieval or evidence-type supplementation. |
| `dual_v1` | Summary-level source retrieval followed by chunk-level retrieval within selected sources. Uses smaller source/chunk candidate pools. |
| `dual_v2` | Same two-stage structure with larger candidate pools and optional preferred-evidence handling for higher-level evidence categories. |
| `dual_v3` | Same as `dual_v2`, with optional temporal metadata handling when an examination year or query year is available. |

The public example documents the retrieval structure and exposes configuration hooks for category and temporal scoring, but does not disclose the study-specific priority weights. By default, the released service ranks candidates primarily by reranker score and leaves category-priority and temporal bonuses disabled unless users configure them for their own corpora.

Example request for the initial non-dual chunk-level baseline:

```bash
curl -X POST http://127.0.0.1:50002/getReference \
  -H "Content-Type: application/json" \
  -d '{"request":"pressure injury prevention in older adults","strategy":"single_chunk","top_k":5}'
```

The underlying knowledge base, private documents, server addresses, and deployment credentials are not included in this release because the evidence sources include copyrighted or licensed materials. If a deployment only exposes a single retrieval endpoint, set `RAG_MODE=single`.

## License

This code repository is released under the Apache License 2.0. See [LICENSE](LICENSE).
