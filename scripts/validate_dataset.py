#!/usr/bin/env python3
"""Validate the basic schema of a NursData-MCQ JSON file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evinurse_eval.io import load_json_records


REQUIRED_FIELDS = {"id", "question", "options", "answer"}
VALID_OPTIONS = {"A", "B", "C", "D", "E"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    records = load_json_records(args.input)
    ids = []
    errors = []

    for index, item in enumerate(records):
        missing = REQUIRED_FIELDS - set(item)
        if missing:
            errors.append(f"row {index}: missing fields {sorted(missing)}")
            continue
        ids.append(item["id"])
        if not isinstance(item["options"], dict) or not item["options"]:
            errors.append(f"row {index}: options must be a non-empty object")
        if item["answer"] not in VALID_OPTIONS:
            errors.append(f"row {index}: invalid answer {item['answer']!r}")
        if item["answer"] not in item["options"]:
            errors.append(f"row {index}: answer not found in options")

    duplicate_count = len(ids) - len(set(ids))
    if duplicate_count:
        errors.append(f"duplicate ids: {duplicate_count}")

    if errors:
        for error in errors[:50]:
            print(error)
        if len(errors) > 50:
            print(f"... {len(errors) - 50} more errors")
        raise SystemExit(1)

    print(f"validated_records={len(records)}")
    print("schema_ok=true")


if __name__ == "__main__":
    main()

