#!/usr/bin/env python3
"""
Generate Text

This script generates text using a trained language model.

Usage:
    python scripts/generate_text.py \
        --checkpoint models/best_model.pt \
        --tokenizer references/tokenizers/english_bytebpe_8k.json \
        --prompt "Once upon a time"
"""

import argparse
from pathlib import Path

import torch

import sys
sys.path.append(str(Path(__file__).parent.parent))

from transformer_llm.generation.generator import BeamSearchGenerator, TextGenerator, compare_sampling_strategies
from transformer_llm.model.config import ModelConfig, get_small_config
from transformer_llm.model.language_model import TransformerLanguageModel
from transformer_llm.tokenizer.loading import load_tokenizer


def maybe_strip_module_prefix(state_dict: dict) -> dict:
    """
    Make a checkpoint saved from DataParallel loadable in a plain model.

    `nn.DataParallel` saves parameter names like `module.layers.0...`, while the
    single-GPU / CPU model expects names without the `module.` prefix.
    """
    if not state_dict:
        return state_dict

    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained model")

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare different sampling strategies",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=1,
        help="Use beam search when beam_width > 1",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="Length penalty for beam search",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Text Generation")
    print("=" * 80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get("config", {}).get("model_config")
    if model_config is None:
        print("Warning: Model config not found in checkpoint. Using default config.")
        model_config = get_small_config()
        model_config.vocab_size = tokenizer.vocab_size
    elif isinstance(model_config, dict):
        model_config = ModelConfig.from_dict(model_config)

    # Create model
    model = TransformerLanguageModel(model_config)
    state_dict = checkpoint["model_state_dict"]
    state_dict = maybe_strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}, step {checkpoint.get('global_step', 'unknown')}")

    # Create generator
    generator = TextGenerator(model, tokenizer, device)

    print("\n" + "=" * 80)

    # Compare strategies or generate
    if args.compare:
        print("Comparing sampling strategies...")
        print("=" * 80)
        compare_sampling_strategies(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        if args.beam_width > 1 and args.num_samples != 1:
            parser.error("--beam_width > 1 currently supports only --num_samples 1")

        # Generate text
        print(f"Prompt: {args.prompt}")
        print("=" * 80)

        if args.beam_width > 1:
            beam_generator = BeamSearchGenerator(model, tokenizer, device, beam_width=args.beam_width)
            generated_texts = [
                beam_generator.generate(
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    length_penalty=args.length_penalty,
                )
            ]
        else:
            generated_texts = generator.generate(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=not args.greedy,
                num_return_sequences=args.num_samples,
            )

        for i, text in enumerate(generated_texts, 1):
            print(f"\nSample {i}:")
            print("-" * 80)
            print(text)
            print("-" * 80)

    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
