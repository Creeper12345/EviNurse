#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible chat model on MCQs with optional context.

This script mirrors the local evaluation flow used in the project: it reads a
JSON benchmark, keeps the top-k context entries when present, prompts an
OpenAI-compatible endpoint, extracts a final option letter, and writes a
resumable JSON result file.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def call_openai_api(
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content or ""
        end_time = time.time()
        return {
            "response": answer,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "error": None,
        }
    except Exception as exc:
        end_time = time.time()
        return {
            "response": "",
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "error": repr(exc),
        }


def extract_answer(response: str) -> str | None:
    final_patterns = [
        r"(?:答案|最终答案|Answer|Final answer)\s*[:：]?\s*([A-E])",
        r"^\s*([A-E])\s*$",
    ]
    for pattern in final_patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    match = re.search(r"[A-E]", response)
    return match.group(0).upper() if match else None


def format_context(contexts: list[dict], top_k: int) -> str:
    selected_contexts = contexts[:top_k]
    blocks = []
    for idx, ctx in enumerate(selected_contexts, start=1):
        doc_name = ctx.get("doc_name") or ctx.get("source") or ""
        content = ctx.get("content") or ctx.get("text") or ""
        blocks.append(f"[{idx}] {doc_name}\n{content}".strip())
    return "\n\n".join(blocks)


def format_prompt(item: dict, top_k_context: int) -> tuple[str, int]:
    question = item["question"]
    options = item["options"]
    options_str = "\n".join(f"{key}. {value}" for key, value in options.items())
    contexts = item.get("context") or []
    context_str = format_context(contexts, top_k_context)

    if context_str:
        prompt = (
            "你是一名医学护理领域的专业助手，请根据参考资料回答问题。\n\n"
            "【参考资料】\n"
            f"{context_str}\n\n"
            "【问题】\n"
            f"{question}\n"
            "选项：\n"
            f"{options_str}\n\n"
            "请仅回答最合适的选项字母（如 A、B、C、D、E）。"
        )
    else:
        prompt = (
            "你是一名医学护理领域的专业助手，请回答以下选择题。\n\n"
            "【问题】\n"
            f"{question}\n"
            "选项：\n"
            f"{options_str}\n\n"
            "请仅回答最合适的选项字母（如 A、B、C、D、E）。"
        )
    return prompt, min(len(contexts), top_k_context)


def load_json(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of MCQ records.")
    return data


def write_results(path: str | Path, records: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to benchmark JSON.")
    parser.add_argument("--output", required=True, help="Path to prediction JSON.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="EviNurse")
    parser.add_argument("--top-k-context", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    if args.limit:
        data = data[: args.limit]

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    results = []

    for item in tqdm(data, desc="Evaluating MCQs"):
        prompt, used_context_num = format_prompt(item, args.top_k_context)
        messages = [{"role": "user", "content": prompt}]
        api_result = call_openai_api(
            client=client,
            messages=messages,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        predicted_answer = extract_answer(api_result["response"])
        answer_gt = item.get("answer")

        result = {
            "id": item.get("id"),
            "question": item.get("question"),
            "options": item.get("options"),
            "answer": answer_gt,
            "used_context_num": used_context_num,
            "model_output": api_result["response"],
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == answer_gt,
            "response_time": api_result["duration"],
            "error": api_result["error"],
        }
        results.append(result)
        write_results(args.output, results)

        if args.request_sleep > 0:
            time.sleep(args.request_sleep)

    correct_count = sum(1 for item in results if item["is_correct"])
    accuracy = correct_count / len(results) if results else 0
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
