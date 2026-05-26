# Transformer LLM from Scratch

![Last Commit](https://img.shields.io/github/last-commit/chrisandrews1012/transformer-from-scratch)
![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-37%20passed-brightgreen)

This project started as one of my Master's assignments and has since been refactored into a clean, standalone implementation. The goal was to build a decoder-only transformer language model entirely from scratch in PyTorch, with no Hugging Face model weights and no pre-built transformer blocks, and run controlled experiments on architecture choices.

---

## Problem Statement

Large language models are typically used through high-level APIs that abstract away the internals. This project takes the opposite approach: implement every component from scratch and study how individual architectural decisions affect model quality and training cost. Specifically, the experiments measure the effect of model size, normalization placement, positional encoding type, and normalization type on validation loss and training time, all on the same dataset and training budget.

---

## Approach

The model is a standard decoder-only transformer trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset using a byte-level BPE tokenizer with an 8k vocabulary. Each experiment changes exactly one variable from the baseline and holds everything else fixed.

### Architecture

- Causal self-attention with a lower-triangular mask
- Attention variants: Multi-Head Attention (MHA), Grouped Query Attention (GQA)
- Positional encoding: Sinusoidal (baseline), Rotary (RoPE)
- Normalization: LayerNorm (baseline), RMSNorm; configurable pre-norm or post-norm placement
- Feed-forward variants: standard (baseline), SwiGLU, GeGLU, Mixture of Experts (MoE)
- Decoding strategies: greedy, top-k, top-p nucleus sampling, beam search

### Training Setup

- Optimizer: AdamW with a warmup-cosine learning rate schedule
- Mixed-precision (AMP) and gradient accumulation supported
- Hardware: NVIDIA T4 (Google Colab), FP32

---

## Results

Four experiments against the `tiny` baseline (4 layers, d=256, 4 heads, sinusoidal positional encoding, LayerNorm, pre-norm, val loss 3.978):

| Experiment | Variable changed | Baseline val loss | Variant val loss | Notes |
|---|---|---|---|---|
| 1: Model size | tiny → small (6L, d=384, 6H) | 3.978 | 3.427 | 2.6x params, 2.4x train time |
| 2: Norm placement | pre-norm → post-norm | 3.978 | 4.119 | stable but higher loss |
| 3: Positional encoding | sinusoidal → RoPE | 3.978 | 3.018 | no measurable added training time |
| 4: Normalization type | LayerNorm → RMSNorm | 3.978 | 4.009 | no time savings at this scale |

RoPE produced the largest quality gain with no measurable overhead. Full writeup: [reports/experiments.md](reports/experiments.md).

---

## Repository Structure

```text
.
├── configs/
│   ├── tiny.yaml
│   ├── small.yaml
│   ├── small_plus.yaml
│   ├── tiny_postnorm.yaml
│   ├── tiny_rmsnorm.yaml
│   └── tiny_rope.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── docs/
│   └── architecture-notes.md
├── models/
├── notebooks/
├── references/
│   └── tokenizers/
│       └── english_bytebpe_8k.json
├── reports/
│   ├── experiments.md
│   └── figures/
├── scripts/
│   ├── download_english_dataset.py
│   ├── download_sample_dataset.py
│   ├── download_sft_dataset.py
│   ├── generate_report_figures.py
│   ├── generate_text.py
│   ├── prepare_packed_dataset.py
│   ├── test_tokenizer.py
│   ├── train_model.py
│   ├── train_sft.py
│   └── train_tokenizer.py
├── src/transformer_llm/
│   ├── components/
│   │   ├── activation.py
│   │   ├── attention.py
│   │   ├── feedforward.py
│   │   ├── normalization.py
│   │   ├── positional.py
│   │   └── transformer.py
│   ├── data/
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   ├── packed_dataset.py
│   │   └── sft_dataset.py
│   ├── generation/
│   │   └── generator.py
│   ├── model/
│   │   ├── config.py
│   │   └── language_model.py
│   ├── tokenizer/
│   │   ├── base.py
│   │   ├── bpe.py
│   │   ├── byte_bpe.py
│   │   └── loading.py
│   └── training/
│       ├── loss.py
│       ├── scheduler.py
│       └── trainer.py
├── tests/
│   ├── test_attention.py
│   ├── test_generation_module.py
│   ├── test_integration.py
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_training_modules.py
├── pyproject.toml
└── README.md
```

---

## How to Run

### 1. Install

```bash
pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv sync --extra dev
```

### 2. Download the Dataset

**Option A: TinyStories (recommended, requires the `datasets` library)**

```bash
python scripts/download_english_dataset.py \
  --dataset tinystories \
  --max_examples 50000 \
  --output_path data/raw/tinystories_train_50k.jsonl
```

**Option B: Tiny Shakespeare (no extra dependencies)**

```bash
python scripts/download_sample_dataset.py \
  --dataset tinyshakespeare \
  --output_path data/raw/tinyshakespeare/tinyshakespeare.jsonl
```

### 3. Pack the Dataset

Tokenizes and packs sequences to a fixed length for efficient training:

```bash
python scripts/prepare_packed_dataset.py \
  --input_path data/raw/tinystories_train_50k.jsonl \
  --tokenizer_path references/tokenizers/english_bytebpe_8k.json \
  --output_dir data/processed/tinystories_tiny \
  --max_seq_len 512 \
  --max_examples 50000 \
  --no_add_special_tokens
```

### 4. Train a Custom Tokenizer (Optional)

Skip this step to use the pre-trained tokenizer at `references/tokenizers/english_bytebpe_8k.json`.

```bash
python scripts/train_tokenizer.py \
  --input_path data/raw/tinystories_train_50k.jsonl \
  --output_path references/tokenizers/my_tokenizer.json \
  --vocab_size 8000
```

### 5. Train

Each config file corresponds to one experiment. Pass `--num_epochs` to override the default.

```bash
# Baseline
python scripts/train_model.py --config configs/tiny.yaml --num_epochs 1

# Experiment 1: larger model
python scripts/train_model.py --config configs/small.yaml --num_epochs 1

# Experiment 2: post-norm placement
python scripts/train_model.py --config configs/tiny_postnorm.yaml --num_epochs 1

# Experiment 3: RoPE positional encoding
python scripts/train_model.py --config configs/tiny_rope.yaml --num_epochs 1

# Experiment 4: RMSNorm
python scripts/train_model.py --config configs/tiny_rmsnorm.yaml --num_epochs 1
```

Checkpoints are saved to `models/<config_name>/best_model.pt`.

### 6. Generate Text

```bash
python scripts/generate_text.py \
  --checkpoint models/tiny/best_model.pt \
  --tokenizer references/tokenizers/english_bytebpe_8k.json \
  --prompt "Once upon a time," \
  --max_new_tokens 100 \
  --strategy greedy
```

Supported strategies: `greedy`, `top_k`, `top_p`, `beam`.

### 7. Run Tests

```bash
pytest -v
```

---

## Configs

| Config | Layers | d_model | Heads | Variant |
|---|---|---|---|---|
| `tiny.yaml` | 4 | 256 | 4 | baseline |
| `small.yaml` | 6 | 384 | 6 | larger model |
| `tiny_postnorm.yaml` | 4 | 256 | 4 | post-norm placement |
| `tiny_rope.yaml` | 4 | 256 | 4 | RoPE positional encoding |
| `tiny_rmsnorm.yaml` | 4 | 256 | 4 | RMSNorm |

---


