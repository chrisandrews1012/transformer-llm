"""
Training Module

This module contains training utilities including trainers, loss functions, and schedulers.
"""

from .trainer import Trainer, create_trainer
from .loss import (
    LanguageModelingLoss,
    MetricsTracker,
    compute_perplexity,
    compute_accuracy,
    compute_top_k_accuracy,
    compute_all_metrics,
)
from .scheduler import (
    WarmupScheduler,
    CosineAnnealingScheduler,
    WarmupCosineScheduler,
    LinearScheduler,
    create_scheduler,
    visualize_scheduler,
    compare_schedulers,
)

__all__ = [
    # Trainer
    "Trainer",
    "create_trainer",
    # Loss and metrics
    "LanguageModelingLoss",
    "MetricsTracker",
    "compute_perplexity",
    "compute_accuracy",
    "compute_top_k_accuracy",
    "compute_all_metrics",
    # Schedulers
    "WarmupScheduler",
    "CosineAnnealingScheduler",
    "WarmupCosineScheduler",
    "LinearScheduler",
    "create_scheduler",
    "visualize_scheduler",
    "compare_schedulers",
]
