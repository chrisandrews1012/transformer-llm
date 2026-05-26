#!/usr/bin/env python3
"""
Download a small English training corpus in JSONL format.

Default dataset: TinyStories (English text, small-model friendly, produces
cleaner generations than a noisy web corpus).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    """
    Parse arguments, download the chosen English dataset, and write it to JSONL.
    """
    parser = argparse.ArgumentParser(description="Download an English dataset in JSONL format")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["tinystories"],
        default="tinystories",
        help="Which English dataset preset to download",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=50000,
        help="How many examples to save",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/raw/tinystories_train_50k.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if args.dataset == "tinystories":
        dataset_name = "roneneldan/TinyStories"
        text_field = "text"
    else:
        raise ValueError(f"Unsupported dataset preset: {args.dataset}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Downloading English Dataset")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Split: {args.split}")
    print(f"Max examples: {args.max_examples}")
    print(f"Output path: {output_path}")
    print("=" * 80)

    split = f"{args.split}[:{args.max_examples}]"
    dataset = load_dataset(dataset_name, split=split)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, example in enumerate(dataset):
            text = example[text_field].strip()
            if not text:
                continue

            record = {
                "text": text,
                "source": args.dataset,
                "example_id": idx,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    main()
