#!/usr/bin/env python3
"""
Download a small teaching-friendly SFT dataset subset.

Current recommended source:
    databricks/databricks-dolly-15k

Why this source?
- public instruction/response format
- small enough to inspect
- easy to sample down for small-scale experiments

This script intentionally keeps the output format simple:
{"instruction": ..., "input": ..., "response": ..., ...}
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

import sys
sys.path.append(str(Path(__file__).parent.parent))

from transformer_llm.data.sft_dataset import format_sft_prompt
from transformer_llm.tokenizer.loading import load_tokenizer


def sample_dolly_subset(
    examples_per_category: int = 16,
    seed: int = 42,
    tokenizer_path: str | None = None,
    max_seq_len: int | None = None,
):
    """
    Sample a small balanced subset from Dolly 15k.

    Dolly has 8 categories, so the default gives 128 examples total.
    """
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    tokenizer = None
    if tokenizer_path is not None:
        tokenizer = load_tokenizer(tokenizer_path)

    grouped = defaultdict(list)
    for idx, example in enumerate(dataset):
        if tokenizer is not None and max_seq_len is not None:
            prompt = format_sft_prompt(
                example["instruction"],
                example.get("context", ""),
            )
            full_text = f"{prompt} {example['response'].strip()}"
            token_ids = tokenizer.encode(full_text, add_special_tokens=False)
            token_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]
            if len(token_ids) > max_seq_len:
                continue
        grouped[example["category"]].append((idx, example))

    rng = random.Random(seed)
    sampled_records = []

    for category in sorted(grouped):
        pool = grouped[category]
        rng.shuffle(pool)
        for source_index, example in pool[:examples_per_category]:
            sampled_records.append(
                {
                    "instruction": example["instruction"].strip(),
                    "input": example.get("context", "").strip(),
                    "response": example["response"].strip(),
                    "category": category,
                    "source_dataset": "databricks/databricks-dolly-15k",
                    "source_index": source_index,
                }
            )

    rng.shuffle(sampled_records)
    return sampled_records


def write_jsonl(records, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_readme(
    readme_path: Path,
    output_filename: str,
    num_records: int,
    examples_per_category: int,
    tokenizer_path: str | None,
    max_seq_len: int | None,
):
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme = f"""# Small SFT dataset for teaching

This folder stores a small sampled subset of:
- `databricks/databricks-dolly-15k`

Recommended citation/source page:
- https://huggingface.co/datasets/databricks/databricks-dolly-15k

Generation details for this local subset:
- output file: `{output_filename}`
- total examples: {num_records}
- sampling strategy: {examples_per_category} examples per category
- random seed: 42
"""
    if tokenizer_path is not None and max_seq_len is not None:
        readme += f"""- filtered to fit tokenizer: `{tokenizer_path}`
- max sequence length used for filtering: {max_seq_len}
"""

    readme += """

This subset is meant for small-scale SFT experiments, not benchmark-quality
instruction tuning.
"""
    readme_path.write_text(readme, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Download a small SFT dataset subset")
    parser.add_argument(
        "--source",
        choices=["dolly"],
        default="dolly",
        help="Which SFT source dataset to sample from",
    )
    parser.add_argument(
        "--examples_per_category",
        type=int,
        default=16,
        help="Number of examples to sample from each Dolly category",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/raw/dolly_15k_small/dolly_15k_128.jsonl",
        help="Where to save the sampled JSONL file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="references/tokenizers/english_bytebpe_8k.json",
        help="Tokenizer used to filter examples by encoded sequence length",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum encoded sequence length allowed in the output subset",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Downloading small SFT dataset")
    print("=" * 80)
    print(f"Source: {args.source}")
    print(f"Examples per category: {args.examples_per_category}")
    print(f"Output path: {args.output_path}")
    print(f"Seed: {args.seed}")
    print(f"Tokenizer filter: {args.tokenizer_path}")
    print(f"Max sequence length: {args.max_seq_len}")
    print("=" * 80)

    if args.source != "dolly":
        raise ValueError(f"Unsupported source: {args.source}")

    records = sample_dolly_subset(
        examples_per_category=args.examples_per_category,
        seed=args.seed,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
    )

    output_path = Path(args.output_path)
    write_jsonl(records, output_path)
    write_readme(
        output_path.parent / "README.md",
        output_path.name,
        len(records),
        args.examples_per_category,
        args.tokenizer_path,
        args.max_seq_len,
    )

    print(f"\nSaved {len(records)} records to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
