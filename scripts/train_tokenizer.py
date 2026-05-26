#!/usr/bin/env python3
"""
Train Tokenizer (BPE or Character-level)

This script trains a tokenizer on the provided dataset.

Usage:
    python scripts/train_tokenizer.py --data_path /path/to/data.jsonl --vocab_size 8000 --tokenizer_type bpe
    python scripts/train_tokenizer.py --data_path /path/to/data.jsonl --tokenizer_type char
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from transformer_llm.tokenizer.base import CharacterTokenizer
from transformer_llm.tokenizer.bpe import BPETokenizer
from transformer_llm.tokenizer.byte_bpe import ByteLevelBPETokenizer
from tqdm import tqdm


def iter_texts_from_jsonl(data_path: str, max_examples: int = None):
    """
    Yield texts from JSONL one by one.

    This is the low-memory path. It is especially useful when reusing a
    real-world dataset but keeping tokenizer training simple.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break

            data = json.loads(line)
            text = data.get("text", "")
            if text:
                yield text


def load_texts_from_jsonl(data_path: str, max_examples: int = None):
    """
    Load texts from JSONL file.

    Args:
        data_path: Path to JSONL file
        max_examples: Maximum number of examples to load

    Returns:
        List of text strings
    """
    return list(
        tqdm(
            iter_texts_from_jsonl(data_path, max_examples),
            desc="Loading texts",
            unit="texts",
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer (BPE or Character-level)")

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data (JSONL format)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to use for training",
    )

    # Tokenizer type
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["bpe", "byte_bpe", "char"],
        default="bpe",
        help="Type of tokenizer to train: 'bpe', 'byte_bpe', or 'char'",
    )

    # Tokenizer arguments
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8000,
        help="Target vocabulary size (only used for BPE)",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for a pair to be merged (only used for BPE)",
    )

    # Output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/raw/tokenizer.json",
        help="Path to save trained tokenizer",
    )

    args = parser.parse_args()

    print("=" * 80)
    print(f"Training {args.tokenizer_type.upper()} Tokenizer")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Tokenizer type: {args.tokenizer_type}")
    if args.tokenizer_type in {"bpe", "byte_bpe"}:
        print(f"Vocab size: {args.vocab_size}")
        print(f"Min frequency: {args.min_frequency}")
    print(f"Output path: {args.output_path}")
    print("=" * 80)

    # Train tokenizer based on type
    print("\nTraining tokenizer...")
    if args.tokenizer_type == "bpe":
        print("Loading data into memory for the teaching BPE tokenizer...")
        texts = load_texts_from_jsonl(args.data_path, args.max_examples)
        print(f"✓ Loaded {len(texts)} texts")
        tokenizer = BPETokenizer()
        tokenizer.train(
            texts=texts,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
    elif args.tokenizer_type == "byte_bpe":
        tokenizer = ByteLevelBPETokenizer()
        texts = iter_texts_from_jsonl(args.data_path, args.max_examples)
        tokenizer.train(
            texts=texts,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
    elif args.tokenizer_type == "char":
        tokenizer = CharacterTokenizer()
        texts = iter_texts_from_jsonl(args.data_path, args.max_examples)
        tokenizer.train(texts=texts)
    else:
        raise ValueError(f"Unknown tokenizer type: {args.tokenizer_type}")

    print(f"✓ Training complete. Final vocab size: {tokenizer.vocab_size}")

    # Save tokenizer
    print(f"\nSaving tokenizer to {args.output_path}...")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output_path)
    print(f"✓ Tokenizer saved")

    # Load from saved tokenizer to verify
    print(f"\nVerifying saved tokenizer...")
    if args.tokenizer_type == "bpe":
        tokenizer = BPETokenizer()
    elif args.tokenizer_type == "byte_bpe":
        tokenizer = ByteLevelBPETokenizer()
    else:
        tokenizer = CharacterTokenizer()
    tokenizer.load(args.output_path)
    print(f"✓ Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Test tokenizer
    print("\n" + "=" * 80)
    print("Testing tokenizer:")
    print("=" * 80)
    test_text = "Hello world, this is a test."
    token_ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(token_ids)

    print(f"Test text:    {test_text}")
    print(f"Token IDs:    {token_ids}")
    print(f"Decoded:      {decoded}")
    print(f"Vocab size:   {tokenizer.vocab_size}")
    print(f"Num tokens:   {len(token_ids)}")

    print("\n" + "=" * 80)
    print("✓ Tokenizer training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
