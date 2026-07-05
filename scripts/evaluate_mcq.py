#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible chat model on a nursing MCQ benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evinurse_eval.answer_extraction import extract_choice
from evinurse_eval.io import load_json_records, save_json_records
from evinurse_eval.metrics import accuracy


SYSTEM_PROMPT = (
    "You are a nursing examination assistant. Answer the multiple-choice "
    "question by selecting one final option letter from A to E. "
    "Return the final answer in the form: 答案：A"
)


def format_question(item: dict) -> str:
    lines = [f"问题：{item['question']}", "选项："]
    for key in sorted(item["options"]):
        lines.append(f"{key}. {item['options'][key]}")
    lines.append("请只给出最终答案。")
    return "\n".join(lines)


def call_model(client: OpenAI, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to benchmark JSON.")
    parser.add_argument("--output", required=True, help="Path to prediction JSON.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_records(args.input)
    if args.limit:
        records = records[: args.limit]

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    outputs = []

    for item in tqdm(records, desc="Evaluating"):
        prompt = format_question(item)
        try:
            model_output = call_model(
                client=client,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            predicted = extract_choice(model_output)
            error = None
        except Exception as exc:  # Keep batch evaluation resumable/debuggable.
            model_output = ""
            predicted = None
            error = repr(exc)

        result = dict(item)
        result.update(
            {
                "model_output": model_output,
                "predicted_answer": predicted,
                "is_correct": predicted == item.get("answer"),
                "error": error,
            }
        )
        outputs.append(result)

    save_json_records(args.output, outputs)
    correct, total, score = accuracy(outputs)
    print(f"Accuracy: {score:.4%} ({correct}/{total})")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()

