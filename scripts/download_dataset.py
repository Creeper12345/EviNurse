#!/usr/bin/env python3
"""Download the NursData-MCQ benchmark from Hugging Face."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Agnania/NursData-MCQ")
    parser.add_argument("--output-dir", default="data/NursData-MCQ")
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.output_dir,
    )
    print(path)


if __name__ == "__main__":
    main()

