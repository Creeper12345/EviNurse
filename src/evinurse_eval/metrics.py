"""Evaluation metrics."""

from __future__ import annotations


def accuracy(records: list[dict]) -> tuple[int, int, float]:
    total = len(records)
    correct = sum(1 for item in records if item.get("is_correct") is True)
    return correct, total, correct / total if total else 0.0

