#!/usr/bin/env python3
"""
Test Tokenizer (BPE or Character-level)

This script loads a trained tokenizer and tests it with various inputs.
Automatically detects the tokenizer type from the saved file.

Usage:
    python scripts/test_tokenizer.py --tokenizer_path references/tokenizers/english_bytebpe_8k.json
    python scripts/test_tokenizer.py --tokenizer_path references/tokenizers/english_bytebpe_8k.json --text "Your custom text here"
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from transformer_llm.tokenizer.loading import detect_tokenizer_type, load_tokenizer


def run_tokenizer_test(tokenizer, text: str):
    """Run a quick encode/decode check and print results."""
    print(f"\nInput text: '{text}'")
    print("-" * 80)

    # Encode
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    print(f"Token IDs (with special tokens): {token_ids}")

    token_ids_no_special = tokenizer.encode(text, add_special_tokens=False)
    print(f"Token IDs (without special):     {token_ids_no_special}")

    # Decode
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"Decoded text: '{decoded}'")

    # Show tokens
    if hasattr(tokenizer, 'id_to_token'):
        tokens = [tokenizer.id_to_token.get(tid, "<UNK>") for tid in token_ids_no_special]
    elif hasattr(tokenizer, 'id_to_char'):
        tokens = [tokenizer.id_to_char.get(tid, "<UNK>") for tid in token_ids_no_special]
    else:
        tokens = []
    print(f"Tokens: {tokens}")

    print(f"Number of tokens: {len(token_ids_no_special)}")
    print(f"Compression ratio: {len(text.split())}/{len(token_ids_no_special)} = {len(text.split())/max(len(token_ids_no_special), 1):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Test tokenizer (BPE or Character-level)")

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="references/tokenizers/english_bytebpe_8k.json",
        help="Path to tokenizer JSON (defaults to the provided English baseline tokenizer)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom text to test (optional)",
    )

    args = parser.parse_args()

    # Detect tokenizer type
    print("=" * 80)
    print("Testing Tokenizer")
    print("=" * 80)
    print(f"Tokenizer path: {args.tokenizer_path}")

    tokenizer_type = detect_tokenizer_type(args.tokenizer_path)
    print(f"Detected type: {tokenizer_type.upper()}")
    print("=" * 80)

    # Load appropriate tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    print(f"✓ Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Test with default examples or custom text
    if args.text:
        run_tokenizer_test(tokenizer, args.text)
    else:
        # Default test cases
        test_cases = [
            "Hello world, this is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating!",
            "Tokenization works by splitting text into tokens.",
            "Testing with numbers: 123 and symbols: @#$%",
        ]

        print("\n" + "=" * 80)
        print("Running default test cases:")
        print("=" * 80)

        for text in test_cases:
            run_tokenizer_test(tokenizer, text)

    # Show vocabulary statistics
    print("\n" + "=" * 80)
    print("Vocabulary Statistics:")
    print("=" * 80)
    print(f"Total vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: {list(tokenizer.special_tokens.keys())}")

    # Show some example tokens
    print("\nSample tokens from vocabulary (first 20):")
    if hasattr(tokenizer, 'id_to_token'):
        vocab_dict = tokenizer.id_to_token
    elif hasattr(tokenizer, 'id_to_char'):
        vocab_dict = tokenizer.id_to_char
    else:
        vocab_dict = {}

    for i in range(min(20, len(vocab_dict))):
        if i in vocab_dict:
            print(f"  {i:3d}: '{vocab_dict[i]}'")

    print("\n" + "=" * 80)
    print("✓ Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
