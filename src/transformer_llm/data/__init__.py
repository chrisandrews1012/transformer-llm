"""
Data Module

This module contains dataset and dataloader implementations for language modeling.
"""

from .dataset import LanguageModelingDataset, TextDataset, StreamingDataset, prepare_data
from .sft_dataset import format_sft_prompt
from .dataloader import (
    collate_fn,
    create_dataloader,
    BucketBatchSampler,
    create_bucketed_dataloader,
    compute_dataset_stats,
    print_dataset_stats,
)

__all__ = [
    # Datasets
    "LanguageModelingDataset",
    "TextDataset",
    "StreamingDataset",
    "prepare_data",
    "format_sft_prompt",
    # Dataloaders
    "collate_fn",
    "create_dataloader",
    "BucketBatchSampler",
    "create_bucketed_dataloader",
    "compute_dataset_stats",
    "print_dataset_stats",
]
