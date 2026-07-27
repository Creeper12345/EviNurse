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

## Retrieval-Augmented Generation Reproducibility

The released repository documents the RAG serving and evaluation workflow, but
does not include the full retrieval knowledge base, evidence documents, or the
deployed retrieval service. These components are not redistributed because the
retrieval corpus contains evidence sources that may be subject to copyright,
database licensing, or platform access restrictions.

The available RAG-related components are:

- `server/dual_stage_retrieval_api.py`: a de-identified dual-stage retrieval
  service example.
- `server/rag_openai_api.py`: an OpenAI-compatible generation API that calls a
  retrieval service and constructs the evidence-grounded prompt.
- `scripts/evaluate_mcq_with_context.py`: an evaluator for benchmark records
  that already include retrieved context snippets.
- Configuration through environment variables for model paths, retrieval
  endpoints, vector collection identifiers, and field names.

To reproduce a local RAG workflow, researchers need access to the same or
equivalent evidence sources and should:

1. Extract and clean evidence documents.
2. Segment documents into recursive text chunks with a maximum chunk
   length of 1,024 characters and an overlap of 256 characters.
3. Build a summary-level knowledge base by storing source identifiers, document
   titles, source categories, publication years when available, source
   summaries, and summary embeddings.
4. Build a chunk-level knowledge base by storing source identifiers, document
   titles, source categories, publication years when available, chunk text, and
   chunk embeddings.
5. Encode summaries and chunks with BGE-M3.
6. Use vector retrieval with L2 distance and rerank candidates with
   BGE-reranker-v2-m3.
7. Run metadata-aware dual-stage retrieval: summary-level source selection
   followed by chunk-level passage selection within shortlisted sources.
8. Apply evidence-type supplementation so that the final evidence context is
   not dominated by one source type and can include higher-level evidence, such
   as guidelines, evidence summaries, and systematic reviews, when available.
9. Use the self-deployed llm or the OpenAI-compatible RAG API to reproduce
   the answer-generation and MCQ-evaluation workflow.

Example retrieval service command:

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

The service exposes `/getReference`, which can be consumed by
`server/rag_openai_api.py` through `RAG_ENDPOINT=/getReference`.
If only a chunk-level vector collection is available, the same service can run
the `single_chunk` baseline without a summary-level collection. Summary-level
collection access is required only for `dual_v1`, `dual_v2`, and `dual_v3`.

Retrieval strategies:

| Strategy | Reproducibility role |
| --- | --- |
| `single_chunk` | Initial chunk-level retrieval baseline. It searches only the chunk-level knowledge base, optionally reranks candidates, and returns the final top-k passages. |
| `dual_v1` | Two-stage retrieval with summary-level source screening followed by chunk-level retrieval within selected sources. It uses smaller candidate pools. |
| `dual_v2` | Two-stage retrieval with larger candidate pools and optional preferred-evidence handling for higher-level evidence categories. |
| `dual_v3` | `dual_v2` plus optional temporal metadata handling when an examination year or query year is available. |

The released example documents the retrieval structure and exposes configuration
hooks for category and temporal scoring, but does not disclose the
study-specific priority weights. By default, the public service ranks candidates
primarily by reranker score and leaves category-priority and temporal bonuses
disabled unless users configure them for their own corpora.

Example request for non-dual chunk-level retrieval:

```bash
curl -X POST http://127.0.0.1:50002/getReference \
  -H "Content-Type: application/json" \
  -d '{"request":"pressure injury prevention in older adults","strategy":"single_chunk","top_k":5}'
```

## Recommended Reporting

For each evaluated model, report:

- Model name and version.
- Endpoint framework.
- Prompt template.
- Decoding parameters.
- Correct count, total count, and accuracy.
- Prediction file or checksum.
