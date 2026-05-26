"""
Loss functions and metrics for language modeling.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelingLoss(nn.Module):
    """
    Cross-entropy loss for language modeling with padding token masking.

    Computes next-token cross-entropy while ignoring padding positions.
    """

    def __init__(self, pad_token_id: int = 0, label_smoothing: float = 0.0):
        """
        Initialize language modeling loss.

        :param pad_token_id: ID of the padding token (ignored in loss computation).
        :type pad_token_id: int
        :param label_smoothing: Label smoothing factor (0.0 = no smoothing).
        :type label_smoothing: float
        """
        super().__init__()
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss over non-padding positions.

        :param logits: Model predictions of shape (batch_size, seq_len, vocab_size).
        :type logits: torch.Tensor
        :param targets: Target token IDs of shape (batch_size, seq_len).
        :type targets: torch.Tensor
        :returns: Scalar loss value.
        :rtype: torch.Tensor
        """
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        loss = F.cross_entropy(logits, targets, ignore_index=self.pad_token_id, label_smoothing=self.label_smoothing)

        return loss


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity measures how well the model predicts the next token.
    Formula: perplexity = exp(cross_entropy_loss).

    :param loss: Cross-entropy loss value.
    :type loss: float
    :returns: Perplexity value (lower is better).
    :rtype: float
    """
    return math.exp(loss)


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int = 0,
) -> float:
    """
    Compute token-level accuracy, ignoring padding positions.

    :param logits: Model predictions of shape (batch_size, seq_len, vocab_size).
    :type logits: torch.Tensor
    :param targets: Target token IDs of shape (batch_size, seq_len).
    :type targets: torch.Tensor
    :param pad_token_id: Padding token ID to exclude from accuracy calculation.
    :type pad_token_id: int
    :returns: Accuracy as a float between 0 and 1.
    :rtype: float
    """
    predictions = torch.argmax(logits, dim=-1)

    mask = targets != pad_token_id

    correct = ((predictions == targets) & mask).sum()
    total = mask.sum()
    accuracy = correct.float() / total.float()

    return accuracy.item()


def compute_top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
    pad_token_id: int = 0,
) -> float:
    """
    Compute top-k accuracy, ignoring padding positions.

    Measures the fraction of positions where the correct token appears in the
    top-k predictions.

    :param logits: Model predictions of shape (batch_size, seq_len, vocab_size).
    :type logits: torch.Tensor
    :param targets: Target token IDs of shape (batch_size, seq_len).
    :type targets: torch.Tensor
    :param k: Number of top predictions to consider.
    :type k: int
    :param pad_token_id: Padding token ID to exclude.
    :type pad_token_id: int
    :returns: Top-k accuracy as a float between 0 and 1.
    :rtype: float
    """
    _, top_k_preds = torch.topk(logits, k, dim=-1)

    in_top_k = (targets.unsqueeze(-1) == top_k_preds).any(dim=-1)

    mask = (targets != pad_token_id)
    correct = (in_top_k & mask).sum()
    total = mask.sum()
    accuracy = correct.float() / total.float()

    return accuracy.item()


class MetricsTracker:
    """
    Track and compute running averages of metrics during training.
    """

    def __init__(self) -> None:
        """
        Initialize metrics tracker with empty state.
        """
        self.reset()

    def reset(self) -> None:
        """
        Reset all tracked metrics.
        """
        self.metrics: dict = {}
        self.counts: dict = {}

    def update(self, **kwargs: float) -> None:
        """
        Update metrics with new values.

        :param kwargs: Metric name and value pairs to accumulate.
        """
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0

            self.metrics[name] += value
            self.counts[name] += 1

    def compute(self) -> dict:
        """
        Compute the average of all tracked metrics.

        :returns: Dictionary of metric names and their running averages.
        :rtype: dict
        """
        averages = {}
        for name in self.metrics:
            averages[name] = self.metrics[name] / self.counts[name]

        return averages

    def get(self, name: str) -> float:
        """
        Get the running average of a specific metric.

        :param name: Metric name.
        :type name: str
        :returns: Average value of the metric (0.0 if not tracked).
        :rtype: float
        """
        if name not in self.metrics:
            return 0.0
        return self.metrics[name] / self.counts[name]

    def __repr__(self) -> str:
        """
        Return a string representation of all metric averages.

        :returns: Comma-separated metric averages.
        :rtype: str
        """
        averages = self.compute()
        return ", ".join([f"{k}: {v:.4f}" for k, v in averages.items()])


def compute_all_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor,
    pad_token_id: int = 0,
) -> dict:
    """
    Compute loss, perplexity, accuracy, and top-5 accuracy for a batch.

    :param logits: Model predictions of shape (batch_size, seq_len, vocab_size).
    :type logits: torch.Tensor
    :param targets: Target token IDs of shape (batch_size, seq_len).
    :type targets: torch.Tensor
    :param loss: Precomputed scalar loss tensor.
    :type loss: torch.Tensor
    :param pad_token_id: Padding token ID to exclude from accuracy metrics.
    :type pad_token_id: int
    :returns: Dictionary with keys "loss", "perplexity", "accuracy", "top5_accuracy".
    :rtype: dict
    """
    metrics = {
        "loss": loss.item(),
        "perplexity": compute_perplexity(loss.item()),
        "accuracy": compute_accuracy(logits, targets, pad_token_id),
        "top5_accuracy": compute_top_k_accuracy(logits, targets, k=5, pad_token_id=pad_token_id),
    }

    return metrics


def test_loss_and_metrics() -> None:
    """
    Test loss functions and metrics with dummy data.
    """
    batch_size, seq_len, vocab_size = 2, 10, 100
    pad_token_id = 0

    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    targets[:, -3:] = pad_token_id

    print("Testing Loss and Metrics:")
    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}\n")

    loss_fn = LanguageModelingLoss(pad_token_id=pad_token_id)
    loss = loss_fn(logits, targets)
    print(f"Loss: {loss.item():.4f}")

    perplexity = compute_perplexity(loss.item())
    print(f"Perplexity: {perplexity:.4f}")

    accuracy = compute_accuracy(logits, targets, pad_token_id)
    print(f"Accuracy: {accuracy:.4f}")

    top5_acc = compute_top_k_accuracy(logits, targets, k=5, pad_token_id=pad_token_id)
    print(f"Top-5 Accuracy: {top5_acc:.4f}")

    print("\nTesting MetricsTracker:")
    tracker = MetricsTracker()
    for i in range(5):
        tracker.update(loss=loss.item(), accuracy=accuracy)
    print(f"Tracked metrics: {tracker}")
    print(f"Average loss: {tracker.get('loss'):.4f}")
