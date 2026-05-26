#!/usr/bin/env python3
"""
Prepare a low-memory packed language-modeling dataset.

This script tokenizes a JSONL file once and saves:
- one flat token file
- one offset array
- one length array
- shuffled train/val/test index arrays

Training can then memory-map these files instead of rebuilding millions of
Python token lists every run.
"""

from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path

import numpy as np

import sys

sys.path.append(str(Path(__file__).parent.parent))

from transformer_llm.tokenizer.loading import load_tokenizer


def choose_token_dtype(vocab_size: int) -> np.dtype:
    """Pick a compact integer dtype for token IDs."""
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def save_split_indices(
    output_dir: Path,
    num_examples: int,
    split_ratio: tuple[float, float, float],
    split_seed: int | None,
):
    """Create one deterministic shuffled split and save the indices."""
    indices = np.arange(num_examples, dtype=np.int32)
    if split_seed is not None:
        rng = np.random.default_rng(split_seed)
        rng.shuffle(indices)

    train_size = int(num_examples * split_ratio[0])
    val_size = int(num_examples * split_ratio[1])
    val_end = train_size + val_size

    np.save(output_dir / "train_idx.npy", indices[:train_size])
    np.save(output_dir / "val_idx.npy", indices[train_size:val_end])
    np.save(output_dir / "test_idx.npy", indices[val_end:])


def main():
    parser = argparse.ArgumentParser(description="Prepare a packed LM dataset")
    parser.add_argument("--input_path", type=str, required=True, help="Raw JSONL file with a text field")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer JSON path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for packed output")
    parser.add_argument("--max_seq_len", type=int, required=True, help="Skip examples longer than this")
    parser.add_argument("--max_examples", type=int, default=None, help="Optional cap for quick presets")
    parser.add_argument("--text_field", type=str, default="text", help="JSON field that contains the text")
    parser.add_argument(
        "--split_ratio",
        type=float,
        nargs=3,
        default=(0.995, 0.005, 0.0),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios",
    )
    parser.add_argument("--split_seed", type=int, default=42, help="Seed for split shuffling")
    parser.add_argument(
        "--no_add_special_tokens",
        action="store_true",
        help="Do not wrap each example with BOS/EOS during tokenization",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Preparing Packed Dataset")
    print("=" * 80)
    print(f"Input path: {args.input_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Output dir: {output_dir}")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"Max examples: {args.max_examples}")
    print("=" * 80)

    tokenizer = load_tokenizer(args.tokenizer_path)
    token_dtype = choose_token_dtype(tokenizer.vocab_size)
    add_special_tokens = not args.no_add_special_tokens

    offsets = array("Q")
    lengths = array("I")
    total_tokens = 0
    kept_examples = 0
    skipped_too_long = 0
    skipped_too_short = 0

    tokens_path = output_dir / "tokens.bin"
    with open(args.input_path, "r", encoding="utf-8") as f_in, open(tokens_path, "wb") as f_out:
        for line_idx, line in enumerate(f_in):
            if args.max_examples is not None and line_idx >= args.max_examples:
                break

            record = json.loads(line)
            text = record.get(args.text_field, "").strip()
            if not text:
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)

            if len(token_ids) < 2:
                skipped_too_short += 1
                continue
            if len(token_ids) > args.max_seq_len:
                skipped_too_long += 1
                continue

            token_array = np.asarray(token_ids, dtype=token_dtype)
            f_out.write(token_array.tobytes())

            offsets.append(total_tokens)
            lengths.append(len(token_array))
            total_tokens += len(token_array)
            kept_examples += 1

            if kept_examples % 10000 == 0:
                print(
                    f"Kept {kept_examples:,} examples | "
                    f"tokens={total_tokens:,} | "
                    f"skipped_long={skipped_too_long:,}"
                )

    np.save(output_dir / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(output_dir / "lengths.npy", np.asarray(lengths, dtype=np.int32))
    save_split_indices(
        output_dir=output_dir,
        num_examples=kept_examples,
        split_ratio=tuple(args.split_ratio),
        split_seed=args.split_seed,
    )

    metadata = {
        "format": "packed_lm_v1",
        "input_path": args.input_path,
        "tokenizer_path": args.tokenizer_path,
        "text_field": args.text_field,
        "vocab_size": tokenizer.vocab_size,
        "token_dtype": token_dtype.name,
        "max_seq_len": args.max_seq_len,
        "add_special_tokens": add_special_tokens,
        "num_examples": kept_examples,
        "total_tokens": total_tokens,
        "split_ratio": list(args.split_ratio),
        "split_seed": args.split_seed,
        "skipped_too_short": skipped_too_short,
        "skipped_too_long": skipped_too_long,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
