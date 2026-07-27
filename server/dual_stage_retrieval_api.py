#!/usr/bin/env python3
"""Dual-stage retrieval API example for EviNurse.

This de-identified release file documents the retrieval service expected by
``server/rag_openai_api.py``. It follows the manuscript architecture:

1. Summary-level retrieval over a summary-level knowledge base.
2. Chunk-level retrieval restricted to sources selected in stage 1.
3. Evidence-type supplementation according to the 5S evidence pyramid.

The script does not include the EviNurse evidence corpus. To run it, users need
to build compatible vector collections:

Summary-level collection:
- ``embedding_vector``: dense vector field for source summaries
- ``source_id`` or ``doc_name``: stable source identifier
- ``doc_name``: source title
- ``summary_text``: source-level summary text
- ``domain_category``: source/evidence category

Chunk-level collection:
- ``embedding_vector``: dense vector field for chunks
- ``source_id`` or ``doc_name``: stable source identifier matching summaries
- ``doc_name``: source title
- ``chunk_text``: indexed passage text
- ``domain_category``: source/evidence category

Endpoint:

POST /getReference
{
  "request": "query text",
  "exam_year": 2024,
  "strategy": "dual_v3",
  "top_k": 5,
  "debug": false
}
"""

from __future__ import annotations

import os
import re
import time
import json
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from FlagEmbedding import FlagReranker
    from pymilvus import Collection, connections
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - import guard for documentation use
    raise SystemExit(
        "Missing retrieval dependencies. Install requirements-vllm.txt or install "
        "pymilvus, sentence-transformers, and FlagEmbedding."
    ) from exc


EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "nursingdb")

SUMMARY_COLLECTION = os.getenv("SUMMARY_COLLECTION", "nursing_summary")
CHUNK_COLLECTION = os.getenv("CHUNK_COLLECTION", os.getenv("MILVUS_COLLECTION", "nursing_article"))

VECTOR_FIELD = os.getenv("VECTOR_FIELD", "embedding_vector")
SOURCE_ID_FIELD = os.getenv("SOURCE_ID_FIELD", "doc_name")
DOC_FIELD = os.getenv("DOC_FIELD", "doc_name")
CATEGORY_FIELD = os.getenv("CATEGORY_FIELD", "domain_category")
SUMMARY_TEXT_FIELD = os.getenv("SUMMARY_TEXT_FIELD", "summary_text")
CHUNK_TEXT_FIELD = os.getenv("CHUNK_TEXT_FIELD", "chunk_text")

USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() not in {"0", "false", "no"}
REQUIRE_SUMMARY_COLLECTION = os.getenv("REQUIRE_SUMMARY_COLLECTION", "false").lower() in {"1", "true", "yes"}

CATEGORY_PRIORITY = json.loads(os.getenv("CATEGORY_PRIORITY_JSON", "{}"))
ENABLE_TEMPORAL_BONUS = os.getenv("ENABLE_TEMPORAL_BONUS", "false").lower() in {"1", "true", "yes"}

PREFERRED_EVIDENCE_TYPES = [
    item.strip()
    for item in os.getenv(
        "PREFERRED_EVIDENCE_TYPES",
        "guideline,evidence_summary_cn,evidence_summary_en,systematic_review",
    ).split(",")
    if item.strip()
]

STRATEGY_DEFAULTS = {
    "single_chunk": {"chunk_search_limit": 100, "final_top_k": 5},
    "dual_v1": {"summary_search_limit": 40, "stage1_source_limit": 6, "chunk_search_limit": 80, "stage2_chunk_limit": 18, "final_top_k": 5},
    "dual_v2": {"summary_search_limit": 60, "stage1_source_limit": 8, "chunk_search_limit": 100, "stage2_chunk_limit": 24, "final_top_k": 5},
    "dual_v3": {"summary_search_limit": 60, "stage1_source_limit": 8, "chunk_search_limit": 100, "stage2_chunk_limit": 24, "final_top_k": 5},
}


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_year(text: str) -> int | None:
    for year_text in re.findall(r"(?:19|20)\d{2}", text or ""):
        year = int(year_text)
        if 1980 <= year <= 2035:
            return year
    return None


def parse_exam_year_from_text(text: str) -> int | None:
    match = re.search(r"(20\d{2}|19\d{2})年", text or "")
    return int(match.group(1)) if match else None


def compute_year_bonus(exam_year: int | None, publication_year: int | None) -> float:
    if not ENABLE_TEMPORAL_BONUS or exam_year is None or publication_year is None:
        return 0.0
    # Private deployments may replace this placeholder with the study-specific
    # temporal scoring rule. The public release does not disclose those weights.
    return 0.0


def map_source_category(domain_category: str | None, doc_name: str, text: str) -> str:
    category = normalize_text(domain_category)
    return category or "other"


def milvus_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_source_filter(source_ids: list[str]) -> str:
    values = ", ".join(milvus_string(source_id) for source_id in source_ids)
    return f"{SOURCE_ID_FIELD} in [{values}]"


def unique_fields(fields: list[str]) -> list[str]:
    unique = []
    for field in fields:
        if field not in unique:
            unique.append(field)
    return unique


class ModelManager:
    def __init__(self) -> None:
        self.embed_model: SentenceTransformer | None = None
        self.reranker: FlagReranker | None = None
        self.summary_collection: Collection | None = None
        self.chunk_collection: Collection | None = None

    async def load(self) -> None:
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = FlagReranker(RERANKER_MODEL, use_fp16=True) if USE_RERANKER else None
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, db_name=MILVUS_DB_NAME)
        self.chunk_collection = Collection(CHUNK_COLLECTION)
        try:
            self.summary_collection = Collection(SUMMARY_COLLECTION)
        except Exception:
            if REQUIRE_SUMMARY_COLLECTION:
                raise
            self.summary_collection = None


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_manager.load()
    yield


app = FastAPI(title="EviNurse dual-stage retrieval API example", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestData(BaseModel):
    request: str
    exam_year: int | None = None
    strategy: str = "dual_v3"
    top_k: int = 5
    summary_search_limit: int | None = None
    stage1_source_limit: int | None = None
    chunk_search_limit: int | None = None
    stage2_chunk_limit: int | None = None
    debug: bool = False


class Context(BaseModel):
    doc_name: str | None = None
    content: str
    source_category: str | None = None
    domain_category: str | None = None
    publication_year: int | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
    stage1_source_score: float | None = None
    stage2_score: float | None = None
    retrieval_strategy: str | None = None
    retrieval_stage: str | None = None


class RetrievalResponse(BaseModel):
    context: list[Context]
    retrieval_debug: dict[str, Any] | None = None


async def vector_search(
    *,
    query: str,
    collection: Collection,
    search_limit: int,
    output_fields: list[str],
    expr: str | None = None,
) -> list[dict[str, Any]]:
    if model_manager.embed_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")

    query_embedding = model_manager.embed_model.encode(query, normalize_embeddings=True)
    results = collection.search(
        data=[query_embedding],
        anns_field=VECTOR_FIELD,
        param={"metric_type": "L2", "offset": 0, "ignore_growing": False, "params": {"nprobe": 32}},
        limit=search_limit,
        expr=expr,
        output_fields=output_fields,
    )

    items = []
    for rank, hit in enumerate(results[0], start=1):
        distance = safe_float(hit.distance)
        dense_score = 1.0 / (1.0 + distance) if distance >= 0 else 0.0
        items.append(
            {
                "entity": hit.entity,
                "dense_distance": distance,
                "dense_score": dense_score,
                "vector_rank": rank,
                "score": dense_score,
            }
        )
    return items


async def rerank(query: str, search_results: list[dict[str, Any]], text_field: str) -> list[dict[str, Any]]:
    if not search_results or model_manager.reranker is None:
        return search_results
    pairs = [[query, item["entity"].get(text_field)] for item in search_results]
    scores = model_manager.reranker.compute_score(pairs)
    reranked = [{**item, "rerank_score": safe_float(scores[i]), "score": safe_float(scores[i])} for i, item in enumerate(search_results)]
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked


def build_source_item(item: dict[str, Any], exam_year: int | None, strategy: str) -> dict[str, Any]:
    entity = item["entity"]
    source_id = normalize_text(entity.get(SOURCE_ID_FIELD) or entity.get(DOC_FIELD))
    doc_name = normalize_text(entity.get(DOC_FIELD) or source_id)
    summary_text = normalize_text(entity.get(SUMMARY_TEXT_FIELD))
    domain_category = normalize_text(entity.get(CATEGORY_FIELD))
    source_category = map_source_category(domain_category, doc_name, summary_text)
    publication_year = extract_year(doc_name) or extract_year(summary_text[:240])
    rerank_score = safe_float(item.get("rerank_score", item.get("score")))
    dense_score = safe_float(item.get("dense_score"))
    category_bonus = CATEGORY_PRIORITY.get(source_category, 0.0)
    year_bonus = compute_year_bonus(exam_year, publication_year)
    stage1_source_score = rerank_score + category_bonus
    if strategy == "dual_v3":
        stage1_source_score += year_bonus
    return {
        "source_id": source_id,
        "doc_name": doc_name,
        "summary_text": summary_text,
        "domain_category": domain_category,
        "source_category": source_category,
        "publication_year": publication_year,
        "dense_score": dense_score,
        "dense_distance": safe_float(item.get("dense_distance")),
        "rerank_score": rerank_score,
        "stage1_source_score": stage1_source_score,
        "category_bonus": category_bonus,
        "year_bonus": year_bonus,
    }


def build_chunk_item(item: dict[str, Any], source_map: dict[str, dict[str, Any]], exam_year: int | None, strategy: str) -> dict[str, Any]:
    entity = item["entity"]
    source_id = normalize_text(entity.get(SOURCE_ID_FIELD) or entity.get(DOC_FIELD))
    source = source_map.get(source_id, {})
    doc_name = normalize_text(entity.get(DOC_FIELD) or source.get("doc_name") or source_id)
    chunk_text = normalize_text(entity.get(CHUNK_TEXT_FIELD))
    domain_category = normalize_text(entity.get(CATEGORY_FIELD) or source.get("domain_category"))
    source_category = map_source_category(domain_category, doc_name, chunk_text)
    publication_year = extract_year(doc_name) or extract_year(chunk_text[:240]) or source.get("publication_year")
    dense_score = safe_float(item.get("dense_score"))
    rerank_score = safe_float(item.get("rerank_score", item.get("score")))
    category_bonus = CATEGORY_PRIORITY.get(source_category, 0.0)
    year_bonus = compute_year_bonus(exam_year, publication_year)
    stage1_source_score = safe_float(source.get("stage1_source_score"))
    stage2_score = rerank_score + category_bonus
    if strategy in {"dual_v2", "dual_v3"} and source_category in PREFERRED_EVIDENCE_TYPES:
        stage2_score += safe_float(os.getenv("PREFERRED_EVIDENCE_BONUS", "0"))
    if strategy == "dual_v3":
        stage2_score += year_bonus
    return {
        "source_id": source_id,
        "doc_name": doc_name,
        "chunk_text": chunk_text,
        "domain_category": domain_category,
        "source_category": source_category,
        "publication_year": publication_year,
        "dense_score": dense_score,
        "dense_distance": safe_float(item.get("dense_distance")),
        "rerank_score": rerank_score,
        "stage1_source_score": stage1_source_score,
        "stage2_score": stage2_score,
        "category_bonus": category_bonus,
        "year_bonus": year_bonus,
    }


def unique_by_source_and_text(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for item in items:
        key = (item["source_id"], item["chunk_text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def supplement_evidence_types(selected: list[dict[str, Any]], pool: list[dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Supplement final context with higher-level evidence categories.

    This implements the third step in Figure 1: after summary-level retrieval
    and chunk-level retrieval, the final evidence set is checked for evidence
    type balance. If the selected context is dominated by one category, the
    function attempts to add missing high-level evidence types from the
    shortlisted candidate pool.
    """

    selected = unique_by_source_and_text(selected)
    category_counts: dict[str, int] = {}
    for item in selected:
        category_counts[item["source_category"]] = category_counts.get(item["source_category"], 0) + 1
    dominant_count = max(category_counts.values(), default=0)
    should_supplement = dominant_count >= max(2, len(selected) - 1)
    added_categories = []

    if not should_supplement and any(item["source_category"] in PREFERRED_EVIDENCE_TYPES for item in selected):
        return selected[:top_k], added_categories

    existing = {item["source_category"] for item in selected}
    for category in PREFERRED_EVIDENCE_TYPES:
        if category in existing:
            continue
        for item in pool:
            if item["source_category"] != category:
                continue
            selected.append(item)
            existing.add(category)
            added_categories.append(category)
            break
        if len(selected) >= top_k:
            break

    selected = unique_by_source_and_text(selected)
    selected.sort(key=lambda x: x["stage2_score"], reverse=True)
    return selected[:top_k], added_categories


async def retrieve_dual_stage(
    *,
    query: str,
    strategy: str,
    exam_year: int | None,
    summary_search_limit: int,
    stage1_source_limit: int,
    chunk_search_limit: int,
    stage2_chunk_limit: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if model_manager.summary_collection is None or model_manager.chunk_collection is None:
        raise HTTPException(status_code=500, detail="Vector collections not loaded")

    summary_hits = await vector_search(
        query=query,
        collection=model_manager.summary_collection,
        search_limit=summary_search_limit,
        output_fields=unique_fields([SOURCE_ID_FIELD, DOC_FIELD, SUMMARY_TEXT_FIELD, CATEGORY_FIELD]),
    )
    summary_hits = await rerank(query, summary_hits, SUMMARY_TEXT_FIELD)
    source_candidates = [build_source_item(item, exam_year, strategy) for item in summary_hits]
    source_candidates.sort(key=lambda x: x["stage1_source_score"], reverse=True)
    selected_sources = source_candidates[:stage1_source_limit]
    selected_source_ids = [item["source_id"] for item in selected_sources if item["source_id"]]
    source_map = {item["source_id"]: item for item in selected_sources}

    if not selected_source_ids:
        return [], {"strategy": strategy, "stage": "no_sources"}

    chunk_hits = await vector_search(
        query=query,
        collection=model_manager.chunk_collection,
        search_limit=chunk_search_limit,
        output_fields=unique_fields([SOURCE_ID_FIELD, DOC_FIELD, CHUNK_TEXT_FIELD, CATEGORY_FIELD]),
        expr=build_source_filter(selected_source_ids),
    )
    chunk_hits = await rerank(query, chunk_hits, CHUNK_TEXT_FIELD)
    chunk_candidates = [build_chunk_item(item, source_map, exam_year, strategy) for item in chunk_hits]
    chunk_candidates.sort(key=lambda x: x["stage2_score"], reverse=True)
    narrowed_pool = unique_by_source_and_text(chunk_candidates)[:stage2_chunk_limit]
    selected = narrowed_pool[:top_k]
    selected, added_categories = supplement_evidence_types(selected, narrowed_pool, top_k)

    debug = {
        "strategy": strategy,
        "exam_year": exam_year,
        "summary_search_limit": summary_search_limit,
        "stage1_source_limit": stage1_source_limit,
        "chunk_search_limit": chunk_search_limit,
        "stage2_chunk_limit": stage2_chunk_limit,
        "selected_source_ids": selected_source_ids,
        "source_candidates": source_candidates[:20],
        "selected_sources": selected_sources,
        "candidate_chunks": narrowed_pool[:20],
        "supplemented_evidence_types": added_categories,
    }
    return selected, debug


async def retrieve_single_chunk(
    *,
    query: str,
    exam_year: int | None,
    chunk_search_limit: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if model_manager.chunk_collection is None:
        raise HTTPException(status_code=500, detail="Chunk-level vector collection not loaded")

    chunk_hits = await vector_search(
        query=query,
        collection=model_manager.chunk_collection,
        search_limit=chunk_search_limit,
        output_fields=unique_fields([SOURCE_ID_FIELD, DOC_FIELD, CHUNK_TEXT_FIELD, CATEGORY_FIELD]),
    )
    chunk_hits = await rerank(query, chunk_hits, CHUNK_TEXT_FIELD)
    chunk_candidates = [build_chunk_item(item, {}, exam_year, "single_chunk") for item in chunk_hits]
    chunk_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    selected = unique_by_source_and_text(chunk_candidates)[:top_k]
    debug = {
        "strategy": "single_chunk",
        "exam_year": exam_year,
        "chunk_search_limit": chunk_search_limit,
        "candidate_chunks": chunk_candidates[:20],
    }
    return selected, debug


def build_context_models(selected: list[dict[str, Any]], strategy: str) -> list[Context]:
    retrieval_stage = "final_single_chunk" if strategy == "single_chunk" else "final_after_evidence_type_supplement"
    return [
        Context(
            doc_name=item["doc_name"],
            content=item["chunk_text"],
            source_category=item["source_category"],
            domain_category=item["domain_category"],
            publication_year=item["publication_year"],
            dense_score=round(item["dense_score"], 6),
            rerank_score=round(item["rerank_score"], 6),
            stage1_source_score=round(item.get("stage1_source_score", 0.0), 6),
            stage2_score=round(item.get("stage2_score", 0.0), 6),
            retrieval_strategy=strategy,
            retrieval_stage=retrieval_stage,
        )
        for item in selected
    ]


@app.post("/getReference")
async def get_reference(data: RequestData):
    start_time = time.perf_counter()
    try:
        strategy = data.strategy if data.strategy in STRATEGY_DEFAULTS else "dual_v3"
        cfg = STRATEGY_DEFAULTS[strategy]
        exam_year = data.exam_year or parse_exam_year_from_text(data.request)
        if strategy == "single_chunk":
            selected, debug = await retrieve_single_chunk(
                query=data.request,
                exam_year=exam_year,
                chunk_search_limit=data.chunk_search_limit or cfg["chunk_search_limit"],
                top_k=data.top_k or cfg["final_top_k"],
            )
        else:
            selected, debug = await retrieve_dual_stage(
                query=data.request,
                strategy=strategy,
                exam_year=exam_year,
                summary_search_limit=data.summary_search_limit or cfg["summary_search_limit"],
                stage1_source_limit=data.stage1_source_limit or cfg["stage1_source_limit"],
                chunk_search_limit=data.chunk_search_limit or cfg["chunk_search_limit"],
                stage2_chunk_limit=data.stage2_chunk_limit or cfg["stage2_chunk_limit"],
                top_k=data.top_k or cfg["final_top_k"],
            )
        contexts = build_context_models(selected, strategy)
        if data.debug:
            debug["elapsed_seconds"] = round(time.perf_counter() - start_time, 4)
            return RetrievalResponse(context=contexts, retrieval_debug=debug)
        return contexts
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG failed: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run("dual_stage_retrieval_api:app", host="0.0.0.0", port=50002, workers=1)
