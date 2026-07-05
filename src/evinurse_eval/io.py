"""Input and output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def save_json_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

