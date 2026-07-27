#!/usr/bin/env bash
set -euo pipefail

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-50002}"
export RAG_EMBEDDING_MODEL="${RAG_EMBEDDING_MODEL:-BAAI/bge-m3}"
export RAG_RERANKER_MODEL="${RAG_RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
export REQUIRE_SUMMARY_COLLECTION="${REQUIRE_SUMMARY_COLLECTION:-false}"
export CATEGORY_PRIORITY_JSON="${CATEGORY_PRIORITY_JSON:-{}}"
export PREFERRED_EVIDENCE_BONUS="${PREFERRED_EVIDENCE_BONUS:-0}"
export ENABLE_TEMPORAL_BONUS="${ENABLE_TEMPORAL_BONUS:-false}"
export MILVUS_HOST="${MILVUS_HOST:-127.0.0.1}"
export MILVUS_PORT="${MILVUS_PORT:-19530}"
export MILVUS_DB_NAME="${MILVUS_DB_NAME:-nursingdb}"
export SUMMARY_COLLECTION="${SUMMARY_COLLECTION:-nursing_summary}"
export CHUNK_COLLECTION="${CHUNK_COLLECTION:-nursing_article}"
export VECTOR_FIELD="${VECTOR_FIELD:-embedding_vector}"
export SOURCE_ID_FIELD="${SOURCE_ID_FIELD:-doc_name}"
export SUMMARY_TEXT_FIELD="${SUMMARY_TEXT_FIELD:-summary_text}"
export CHUNK_TEXT_FIELD="${CHUNK_TEXT_FIELD:-chunk_text}"
export DOC_FIELD="${DOC_FIELD:-doc_name}"
export CATEGORY_FIELD="${CATEGORY_FIELD:-domain_category}"

uvicorn server.dual_stage_retrieval_api:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers 1
