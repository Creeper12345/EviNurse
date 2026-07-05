#!/usr/bin/env python3
"""OpenAI-compatible EviNurse RAG API.

This is a de-identified release version of the project RAG serving code. It
keeps the public API shape, query-rewrite step, evidence prompt construction,
and streaming/non-streaming response behavior, while moving deployment-specific
paths and service endpoints to environment variables.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer


MODEL_PATH = os.getenv("MODEL_PATH", "Agnania/EviNurse-32B")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "EviNurse")
RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://127.0.0.1:50002")
RAG_MODE = os.getenv("RAG_MODE", "dual").lower()
RAG_SOURCE_ENDPOINT = os.getenv("RAG_SOURCE_ENDPOINT", "/retrieve_sources")
RAG_PASSAGE_ENDPOINT = os.getenv("RAG_PASSAGE_ENDPOINT", "/retrieve_passages")
RAG_ENDPOINT = os.getenv("RAG_ENDPOINT", "/getReference")
RAG_TOP_K_SOURCES = int(os.getenv("RAG_TOP_K_SOURCES", "5"))
RAG_TOP_K_PASSAGES = int(os.getenv("RAG_TOP_K_PASSAGES", "5"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.80"))


class MedLLM:
    def __init__(self, model_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
        )
        self.async_model = AsyncLLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            top_p=float(os.getenv("TOP_P", "0.8")),
            repetition_penalty=float(os.getenv("REPETITION_PENALTY", "1.15")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
        )

    async def complete(self, messages: list[dict[str, str]]) -> str:
        chunks = []
        async for chunk in self.generate(messages):
            chunks.append(chunk)
        return "".join(chunks)

    async def generate(self, messages: list[dict[str, str]]):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        request_id = str(time.time_ns())
        generator = self.async_model.generate(text, self.sampling_params, request_id=request_id)
        cursor = 0
        async for request_output in generator:
            output_text = request_output.outputs[0].text
            yield output_text[cursor:]
            cursor = len(output_text)


class ModelManager:
    def __init__(self) -> None:
        self.llm: MedLLM | None = None

    async def load_model(self) -> None:
        self.llm = MedLLM(model_path=MODEL_PATH)

    async def release_resources(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_manager.load_model()
    yield
    await model_manager.release_resources()


app = FastAPI(title="EviNurse RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "EviNurse"
    root: str | None = None
    parent: str | None = None
    permission: list | None = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class Context(BaseModel):
    doc_name: str | None = None
    content: str


class DeltaMessage(BaseModel):
    role: Literal["user", "assistant", "system"] | None = None
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    enable_rag: bool = True


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    context: list[Context] | None = None
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    context: list[Context] | None = None
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: list[ChatCompletionResponseChoice | ChatCompletionResponseStreamChoice]
    created: int = Field(default_factory=lambda: int(time.time()))


async def rewrite_query(messages: list[dict[str, str]]) -> str:
    if model_manager.llm is None:
        raise HTTPException(status_code=500, detail="Chat model not loaded")

    history = []
    for message in messages:
        if message["role"] == "user":
            history.append("[用户query]：" + message["content"])
        elif message["role"] == "assistant":
            history.append("[模型response]：" + message["content"])

    prompt = (
        "历史对话记录：\n"
        + "\n".join(history)
        + "\n\n根据上述历史对话的最后一个[用户query]的内容，生成一个检索语句，"
        "用于检索查找相关护理和医学证据信息。要求补充完整被省略的上下文关键信息，"
        "解决指代模糊问题。\n\n按如下格式输出：\n[检索语句]："
    )
    rewritten = await model_manager.llm.complete([{"role": "user", "content": prompt}])
    rewritten = (
        rewritten.lstrip("[检索语句]：")
        .lstrip("查询")
        .lstrip("检索")
        .lstrip("搜索")
        .lstrip("查找")
        .lstrip("关于")
        .strip()
    )
    return rewritten or messages[-1]["content"]


async def post_retrieval(endpoint: str, payload: dict) -> list[dict]:
    """Call a retrieval endpoint and normalize list-like responses."""

    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    url = f"{RAG_BASE_URL}{endpoint}"
    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
    result = response.json()
    if isinstance(result, dict):
        for key in ("data", "results", "references", "sources", "passages"):
            if isinstance(result.get(key), list):
                result = result[key]
                break
    if not isinstance(result, list):
        return []
    return [item for item in result if isinstance(item, dict)]


def extract_source_ids(sources: list[dict]) -> list[str]:
    ids = []
    for item in sources[:RAG_TOP_K_SOURCES]:
        source_id = item.get("source_id") or item.get("id") or item.get("doc_id") or item.get("doc_name")
        if source_id is not None:
            ids.append(str(source_id))
    return ids


def normalize_reference_items(items: list[dict]) -> list[dict]:
    normalized = []
    for item in items:
        content = item.get("content") or item.get("text") or item.get("passage")
        if not content:
            continue
        normalized.append(
            {
                "doc_name": item.get("doc_name") or item.get("source") or item.get("title"),
                "content": content,
                "source_type": item.get("source_type") or item.get("category"),
                "year": item.get("year") or item.get("publication_year"),
                "score": item.get("score"),
            }
        )
    return normalized[:RAG_TOP_K_PASSAGES]


async def get_reference(query: str) -> list[dict]:
    if RAG_MODE == "single":
        items = await post_retrieval(
            RAG_ENDPOINT,
            {"request": query, "query": query, "top_k": RAG_TOP_K_PASSAGES},
        )
        return normalize_reference_items(items)

    try:
        sources = await post_retrieval(
            RAG_SOURCE_ENDPOINT,
            {"request": query, "query": query, "top_k": RAG_TOP_K_SOURCES},
        )
        source_ids = extract_source_ids(sources)
        passages = await post_retrieval(
            RAG_PASSAGE_ENDPOINT,
            {
                "request": query,
                "query": query,
                "source_ids": source_ids,
                "sources": sources[:RAG_TOP_K_SOURCES],
                "top_k": RAG_TOP_K_PASSAGES,
            },
        )
        references = normalize_reference_items(passages)
        if references:
            return references
    except httpx.HTTPError:
        pass

    payload = {"request": query}
    return normalize_reference_items(await post_retrieval(RAG_ENDPOINT, payload))


def build_rag_prompt(question: str, references: list[dict]) -> str:
    evidence_blocks = []
    for item in references:
        doc_name = item.get("doc_name") or item.get("source") or "Untitled source"
        content = item.get("content", "")
        source_type = item.get("source_type")
        year = item.get("year")
        meta = "；".join(str(x) for x in [source_type, year] if x)
        header = f"[参考文献名称]:{doc_name}"
        if meta:
            header += f"\n[来源元数据]:{meta}"
        evidence_blocks.append(f"{header}\n\n[参考文献内容]:{content}")
    evidence = "\n\n\n".join(evidence_blocks)

    return f"""
你是一名循证护理专家，熟悉证据金字塔和临床护理决策，能够基于高质量医学证据给出规范、可执行的护理建议。

【证据使用规则】
1. 证据优先级从高到低：临床实践指南、证据总结、系统评价/Meta分析、专家共识或原始研究。
2. 当高等级证据已形成明确结论时，不得使用低等级证据替代或削弱结论。
3. 可综合多条高等级证据形成判断，不逐条罗列文献内容。
4. 不得引入证据之外的事实、数据、阈值或推断性结论。
5. 当证据不足或结论不明确时，应保守作答。
6. 与护理问题无关的证据无需提及。

【回答结构要求】
1. 优先给出最关键、可执行的护理建议。
2. 多条建议按重要性排序。
3. 简要注明所依据的证据信息来源。
4. 不得提及模型、检索或生成过程。

护理问题：
{question}

可用证据：
{evidence}
""".strip()


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelCard(id=SERVED_MODEL_NAME)])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if model_manager.llm is None:
        raise HTTPException(status_code=500, detail="Chat model not loaded")
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user.")

    current_messages = [message.model_dump() for message in request.messages]
    context_list = None

    if request.enable_rag:
        try:
            query = await rewrite_query(current_messages)
            references = await get_reference(query)
            context_list = [
                Context(doc_name=item.get("doc_name") or item.get("source"), content=item["content"])
                for item in references
            ]
            if references:
                current_messages[-1]["content"] = build_rag_prompt(
                    question=current_messages[-1]["content"],
                    references=references,
                )
        except Exception:
            context_list = None

    if request.stream:
        return EventSourceResponse(stream_response(current_messages, request.model, context_list))

    response = await model_manager.llm.complete(current_messages)
    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        context=context_list,
        finish_reason="stop",
    )
    return ChatCompletionResponse(model=request.model, choices=[choice], object="chat.completion")


async def stream_response(
    messages: list[dict[str, str]],
    model_id: str,
    context: list[Context] | None = None,
):
    first_choice = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None,
    )
    first_chunk = ChatCompletionResponse(
        model=model_id,
        choices=[first_choice],
        object="chat.completion.chunk",
    )
    yield first_chunk.model_dump_json(exclude_unset=True)

    index = 1
    async for new_text in model_manager.llm.generate(messages):  # type: ignore[union-attr]
        choice = ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(content=new_text),
            finish_reason=None,
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice], object="chat.completion.chunk")
        yield chunk.model_dump_json(exclude_unset=True)
        index += 1

    final_choice = ChatCompletionResponseStreamChoice(
        index=index,
        delta=DeltaMessage(),
        context=context,
        finish_reason="stop",
    )
    final_chunk = ChatCompletionResponse(model=model_id, choices=[final_choice], object="chat.completion.chunk")
    yield final_chunk.model_dump_json(exclude_unset=True) + "\n"
    yield "[DONE]\n"
