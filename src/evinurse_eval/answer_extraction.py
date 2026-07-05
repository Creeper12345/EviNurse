"""Answer extraction helpers for multiple-choice model outputs."""

from __future__ import annotations

import re


EXPLICIT_PATTERNS = [
    re.compile(r"(?:最终答案|答案|Answer|Final answer)\s*[:：]?\s*([A-E])\b", re.IGNORECASE),
    re.compile(r"^\s*([A-E])\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"选\s*([A-E])(?:\b|[。．.、，,\s])", re.IGNORECASE),
]


def extract_choice(text: str) -> str | None:
    """Extract a final option letter from model output.

    The extraction deliberately avoids taking the first arbitrary A-E in the
    response, because explanation text often contains option letters.
    """

    if not text:
        return None
    for pattern in EXPLICIT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    stripped = text.strip()
    if len(stripped) == 1 and stripped.upper() in {"A", "B", "C", "D", "E"}:
        return stripped.upper()
    return None
