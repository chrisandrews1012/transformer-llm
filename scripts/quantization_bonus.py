#!/usr/bin/env python3
"""
Compare one trained checkpoint under different inference formats.

This script is intended for the optional hardware-aware bonus. It compares the
same checkpoint under multiple numerical formats and reports simple efficiency
and output-side evidence.

Examples:
    python scripts/quantization_bonus.py \
        --checkpoint models/tiny/best_model.pt \
        --tokenizer references/tokenizers/english_bytebpe_8k.json \
        --prompt "Once upon a time," \
        --formats fp32 int8

    python scripts/quantization_bonus.py \
        --checkpoint models/tiny/best_model.pt \
        --tokenizer references/tokenizers/english_bytebpe_8k.json \
        --prompt "Once upon a time," \
        --formats fp32 fp16 \
        --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent))

from transformer_llm.model.config import ModelConfig, get_small_config
from transformer_llm.model.language_model import TransformerLanguageModel
from transformer_llm.tokenizer.loading import load_tokenizer


def load_checkpoint(path: str, map_location: str | torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if not all(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def build_model_config(checkpoint: dict[str, Any], tokenizer) -> ModelConfig:
    checkpoint_cfg = checkpoint.get("config", {})

    if isinstance(checkpoint_cfg, dict) and "model_config" in checkpoint_cfg:
        model_cfg = checkpoint_cfg["model_config"]
        if isinstance(model_cfg, ModelConfig):
            cfg = model_cfg
        else:
            cfg = ModelConfig.from_dict(model_cfg)
        cfg.vocab_size = tokenizer.vocab_size
        return cfg

    if isinstance(checkpoint_cfg, dict) and "model" in checkpoint_cfg:
        model_section = dict(checkpoint_cfg["model"])
        model_section.setdefault("vocab_size", tokenizer.vocab_size)
        model_section.setdefault("pad_token_id", tokenizer.pad_token_id)
        model_section.setdefault("bos_token_id", tokenizer.bos_token_id)
        model_section.setdefault("eos_token_id", tokenizer.eos_token_id)
        return ModelConfig.from_dict(model_section)

    cfg = get_small_config()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.bos_token_id = tokenizer.bos_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    return cfg


def create_base_model(checkpoint: dict[str, Any], tokenizer, device: torch.device) -> nn.Module:
    config = build_model_config(checkpoint, tokenizer)
    model = TransformerLanguageModel(config)
    state_dict = strip_module_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


def get_model_disk_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return Path(f.name).stat().st_size / (1024 * 1024)


def prepare_variant(model: nn.Module, fmt: str, device: torch.device) -> tuple[nn.Module, torch.dtype | None, torch.device]:
    fmt = fmt.lower()
    if fmt == "fp32":
        return model, torch.float32, device

    if fmt == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 comparison is only supported on CUDA in this bonus script")
        variant = copy.deepcopy(model).half().to(device).eval()
        return variant, torch.float16, device

    if fmt == "bf16":
        if device.type != "cuda":
            raise ValueError("bf16 comparison is only supported on CUDA in this bonus script")
        variant = copy.deepcopy(model).to(dtype=torch.bfloat16, device=device).eval()
        return variant, torch.bfloat16, device

    if fmt == "int8":
        cpu_model = copy.deepcopy(model).to("cpu").eval()
        variant = torch.quantization.quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)
        return variant, None, torch.device("cpu")

    raise ValueError(f"Unsupported format: {fmt}")


def benchmark_forward(model: nn.Module, input_ids: torch.Tensor, warmup: int, runs: int) -> tuple[float, torch.Tensor]:
    with torch.no_grad():
        for _ in range(warmup):
            logits, _ = model(input_ids)

        if input_ids.device.type == "cuda":
            torch.cuda.synchronize(input_ids.device)
        start = time.perf_counter()
        for _ in range(runs):
            logits, _ = model(input_ids)
        if input_ids.device.type == "cuda":
            torch.cuda.synchronize(input_ids.device)
        elapsed = time.perf_counter() - start

    return elapsed / runs, logits[:, -1, :]


def maybe_peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def top_tokens(tokenizer, logits: torch.Tensor, k: int = 5) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
    result = []
    for score, idx in zip(values[0].tolist(), indices[0].tolist()):
        try:
            decoded = tokenizer.decode([idx], skip_special_tokens=False)
        except Exception:
            decoded = str(idx)
        result.append({"token_id": idx, "token": decoded, "prob": score})
    return result


def hardware_summary(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_memory_gb": round(props.total_memory / 1e9, 2),
        }
    return {"device": device.type}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional quantization / low-precision inference comparison")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer JSON")
    parser.add_argument("--prompt", type=str, default="Once upon a time,", help="Prompt used for comparison")
    parser.add_argument("--formats", nargs="+", default=["fp32", "int8"], help="Formats to compare: fp32 fp16 bf16 int8")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cpu or cuda")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup forward passes per format")
    parser.add_argument("--runs", type=int, default=10, help="Measured forward passes per format")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save JSON results")
    args = parser.parse_args()

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_device = torch.device(requested_device)

    tokenizer = load_tokenizer(args.tokenizer)
    checkpoint = load_checkpoint(args.checkpoint, map_location=base_device)
    base_model = create_base_model(checkpoint, tokenizer, device=base_device)

    encoded = tokenizer.encode(args.prompt, add_special_tokens=True)

    results = {
        "prompt": args.prompt,
        "hardware": hardware_summary(base_device),
        "formats": {},
    }

    print("=" * 80)
    print("Optional quantization / low-precision comparison")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tokenizer:  {args.tokenizer}")
    print(f"Prompt:     {args.prompt}")
    print(f"Base device:{base_device}")
    print(f"Formats:    {', '.join(args.formats)}")

    for fmt in args.formats:
        print("\n" + "-" * 80)
        print(f"Format: {fmt}")
        try:
            if base_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(base_device)
            model, _, run_device = prepare_variant(base_model, fmt, base_device)
            input_ids = torch.tensor([encoded], device=run_device)
            avg_latency_s, final_logits = benchmark_forward(model, input_ids, warmup=args.warmup, runs=args.runs)
            peak_memory_mb = maybe_peak_memory_mb(run_device)
            size_mb = get_model_disk_size_mb(model)
            top5 = top_tokens(tokenizer, final_logits, k=5)

            entry = {
                "format": fmt,
                "run_device": str(run_device),
                "avg_forward_latency_ms": round(avg_latency_s * 1000, 3),
                "model_state_size_mb": round(size_mb, 3),
                "peak_memory_mb": None if peak_memory_mb is None else round(peak_memory_mb, 3),
                "top5_next_tokens": top5,
            }
            results["formats"][fmt] = entry

            print(f"run device:          {entry['run_device']}")
            print(f"avg latency (ms):    {entry['avg_forward_latency_ms']}")
            print(f"state size (MB):     {entry['model_state_size_mb']}")
            if entry["peak_memory_mb"] is not None:
                print(f"peak memory (MB):    {entry['peak_memory_mb']}")
            print("top-5 next tokens:")
            for tok in top5:
                print(f"  id={tok['token_id']:<6} prob={tok['prob']:.4f} token={tok['token']!r}")
        except Exception as exc:
            results["formats"][fmt] = {"format": fmt, "error": str(exc)}
            print(f"failed: {exc}")

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
