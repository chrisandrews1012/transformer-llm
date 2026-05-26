#!/usr/bin/env python3
"""
Train Language Model

This script trains a transformer language model on the provided dataset.

Usage:
    python scripts/train_model.py --config configs/tiny.yaml
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from transformer_llm.data.dataloader import create_bucketed_dataloader, create_dataloader
from transformer_llm.data.dataset import LanguageModelingDataset
from transformer_llm.data.packed_dataset import PackedTokenDataset, is_packed_dataset_dir
from transformer_llm.model.config import ModelConfig
from transformer_llm.model.language_model import TransformerLanguageModel
from transformer_llm.tokenizer.loading import load_tokenizer
from transformer_llm.training.trainer import create_trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_wrap_data_parallel(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Use all visible GPUs by default when more than one CUDA device is available.

    On a multi-GPU machine, this allows running the same command with automatic speedup.
    """
    if device.type != "cuda":
        return model

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return model

    print(f"Using DataParallel across {gpu_count} GPUs")
    return torch.nn.DataParallel(model)


def create_lm_dataset(config: dict, tokenizer, split: str, seed: int):
    """
    Create either a raw-text dataset or a packed low-memory dataset.

    If `data_path` points to a packed dataset directory, we reuse the prepared
    token files. Otherwise we fall back to the original JSONL path.
    """
    data_path = config["data"]["data_path"]

    if is_packed_dataset_dir(data_path):
        return PackedTokenDataset(data_dir=data_path, split=split)

    return LanguageModelingDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=config["model"]["max_seq_len"],
        split=split,
        split_ratio=tuple(config["data"]["split_ratio"]),
        split_seed=config["data"].get("split_seed", seed),
        add_special_tokens=config["data"].get("add_special_tokens", True),
    )

def main():
    parser = argparse.ArgumentParser(description="Train language model")

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name inside configs/optional/experiment_configs.yaml",
    )

    # Override arguments
    parser.add_argument("--data_path", type=str, help="Override data path")
    parser.add_argument("--tokenizer_path", type=str, help="Override tokenizer path")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--num_epochs", type=int, help="Override number of epochs")
    parser.add_argument("--checkpoint_dir", type=str, help="Override checkpoint directory")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible training")
    parser.add_argument("--resume_from", type=str, help="Resume training from checkpoint")
    parser.add_argument(
        "--bucketed",
        action="store_true",
        help="Use bucketed batching for the training dataloader",
    )

    args = parser.parse_args()

    # Load configuration
    print("=" * 80)
    print("Training Transformer Language Model")
    print("=" * 80)

    config = load_config(args.config)

    if args.experiment is not None:
        if "base" not in config or "experiments" not in config:
            raise ValueError("--experiment requires a config file with 'base' and 'experiments' sections")
        if args.experiment not in config["experiments"]:
            raise ValueError(f"Unknown experiment: {args.experiment}")

        merged = json.loads(json.dumps(config["base"]))
        experiment_cfg = config["experiments"][args.experiment]
        for section, values in experiment_cfg.items():
            if isinstance(values, dict) and isinstance(merged.get(section), dict):
                merged[section].update(values)
            else:
                merged[section] = values
        config = merged

    # Override config with command line arguments
    if args.data_path:
        config["data"]["data_path"] = args.data_path
    if args.tokenizer_path:
        config["data"]["tokenizer_path"] = args.tokenizer_path
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.checkpoint_dir:
        config["training"]["checkpoint_dir"] = args.checkpoint_dir
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.bucketed:
        config["training"]["use_bucketed_dataloader"] = True

    print("\nConfiguration:")
    print(json.dumps(config, indent=2))
    print("=" * 80)

    seed = config["training"].get("seed", 42)
    set_seed(seed)
    print(f"\nRandom seed: {seed}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Visible GPUs: {torch.cuda.device_count()}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(config["data"]["tokenizer_path"])
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create model configuration
    print("\nCreating model...")
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=config["model"]["dropout"],
        attention_type=config["model"].get("attention_type", "mha"),
        num_kv_heads=config["model"].get("num_kv_heads"),
        pos_encoding_type=config["model"].get("pos_encoding_type", "sinusoidal"),
        norm_type=config["model"].get("norm_type", "layernorm"),
        norm_position=config["model"].get("norm_position", "pre"),
        ffn_type=config["model"].get("ffn_type", "standard"),
        activation=config["model"].get("activation", "gelu"),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Create model
    model = TransformerLanguageModel(model_config)
    model = model.to(device)
    model = maybe_wrap_data_parallel(model, device)

    model_for_stats = model.module if isinstance(model, torch.nn.DataParallel) else model
    print(f"Model parameters: {model_for_stats.get_num_params():,}")
    print(f"Non-embedding parameters: {model_for_stats.get_num_params(non_embedding=True):,}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = create_lm_dataset(config, tokenizer, split="train", seed=seed)
    val_dataset = create_lm_dataset(config, tokenizer, split="val", seed=seed)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader_fn = (
        create_bucketed_dataloader
        if config["training"].get("use_bucketed_dataloader", False)
        else create_dataloader
    )
    train_dataloader = train_loader_fn(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 0),
        pad_token_id=tokenizer.pad_token_id,
    )

    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        pad_token_id=tokenizer.pad_token_id,
    )

    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")
    print(f"Bucketed train loader: {config['training'].get('use_bucketed_dataloader', False)}")

    # Create trainer
    print("\nCreating trainer...")
    train_config = {
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "max_grad_norm": config["training"]["max_grad_norm"],
        "gradient_accumulation_steps": config["training"].get("gradient_accumulation_steps", 1),
        "use_amp": config["training"].get("use_amp", False),
        "checkpoint_dir": config["training"]["checkpoint_dir"],
        "log_interval": config["training"].get("log_interval", 10),
        "scheduler_type": config["training"]["scheduler_type"],
        "warmup_steps": config["training"]["warmup_steps"],
        "total_steps": len(train_dataloader) * config["training"]["num_epochs"],
        "pad_token_id": tokenizer.pad_token_id,
        "model_config": model_config.to_dict(),
    }

    trainer = create_trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=train_config,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    trainer.train(num_epochs=config["training"]["num_epochs"])

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
