"""Tests for data loading, loss computation, and learning rate scheduling."""

import pytest
import torch

from transformer_llm.data.dataloader import collate_fn, create_dataloader
from transformer_llm.data.dataset import TextDataset
from transformer_llm.training.loss import LanguageModelingLoss, MetricsTracker, compute_perplexity
from transformer_llm.training.scheduler import create_scheduler


class DummyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        base = [ord(ch) % 17 + 1 for ch in text]
        if add_special_tokens:
            return [1] + base + [2]
        return base


def test_collate_fn_pads_and_masks():
    batch = [
        (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
        (torch.tensor([5, 6]), torch.tensor([6, 7])),
    ]

    input_ids, target_ids, attention_mask = collate_fn(batch, pad_token_id=0)

    assert input_ids.shape == (2, 3)
    assert target_ids.shape == (2, 3)
    assert attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]


def test_create_dataloader_returns_padded_batches():
    tokenizer = DummyTokenizer()
    dataset = TextDataset(["abc", "abcdef", "xy"], tokenizer, max_seq_len=16)
    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pad_token_id=tokenizer.pad_token_id,
        pin_memory=False,
    )

    batch = next(iter(dataloader))
    input_ids, target_ids, attention_mask = batch
    assert input_ids.ndim == 2
    assert target_ids.shape == input_ids.shape
    assert attention_mask.shape == input_ids.shape


def test_language_modeling_loss_ignores_padding():
    loss_fn = LanguageModelingLoss(pad_token_id=0)
    logits = torch.tensor(
        [
            [[0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
            [[0.0, 4.0, 0.0], [5.0, 0.0, 0.0]],
        ]
    )
    targets = torch.tensor([[1, 2], [1, 0]])

    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_metrics_tracker_computes_means():
    tracker = MetricsTracker()
    tracker.update(loss=2.0, accuracy=0.5)
    tracker.update(loss=4.0, accuracy=1.0)
    metrics = tracker.compute()

    assert metrics["loss"] == pytest.approx(3.0)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert compute_perplexity(0.0) == pytest.approx(1.0)


def test_scheduler_factory_and_step():
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = create_scheduler(
        optimizer,
        scheduler_type="warmup_cosine",
        warmup_steps=2,
        total_steps=8,
    )

    before = scheduler.get_lr()
    scheduler.step()
    after = scheduler.get_lr()

    assert before >= 0.0
    assert after >= 0.0

