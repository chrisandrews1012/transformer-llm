"""
DataLoader utilities for language modeling.

Core: ``collate_fn``, ``create_dataloader``.
Optional: ``BucketBatchSampler``, ``create_bucketed_dataloader``.
"""

import random
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate variable-length sequences into a padded batch.

    :param batch: List of (input_ids, target_ids) tensor pairs.
    :type batch: List[Tuple[torch.Tensor, torch.Tensor]]
    :param pad_token_id: ID of the padding token.
    :type pad_token_id: int
    :returns: Tuple of (input_ids, target_ids, attention_mask). All tensors have
        shape (batch_size, max_seq_len); attention_mask has 1 for real tokens and
        0 for padding.
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    input_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = (input_ids_padded != pad_token_id).long()

    return input_ids_padded, target_ids_padded, attention_mask


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_token_id: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with proper collation for language modeling.

    :param dataset: PyTorch dataset.
    :type dataset: Dataset
    :param batch_size: Batch size.
    :type batch_size: int
    :param shuffle: Whether to shuffle data.
    :type shuffle: bool
    :param num_workers: Number of worker processes for data loading.
    :type num_workers: int
    :param pad_token_id: Padding token ID.
    :type pad_token_id: int
    :param pin_memory: Whether to pin memory for faster GPU transfer.
    :type pin_memory: bool
    :returns: Configured DataLoader instance.
    :rtype: DataLoader
    """
    collate = lambda batch: collate_fn(batch, pad_token_id=pad_token_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
    )


class BucketBatchSampler:
    """
    Batch sampler that groups sequences of similar lengths (optional extension).

    Reduces padding by sorting sequences by length and batching similar-length
    sequences together.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize bucket batch sampler.

        :param dataset: Dataset to sample from.
        :type dataset: Dataset
        :param batch_size: Batch size.
        :type batch_size: int
        :param shuffle: Whether to shuffle batches.
        :type shuffle: bool
        :param drop_last: Whether to drop the last incomplete batch.
        :type drop_last: bool
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.lengths = []
        for i in range(len(dataset)):
            length = len(dataset[i][0])
            self.lengths.append((i, length))

        self.lengths.sort(key=lambda x: x[1])

    def __iter__(self):
        """
        Iterate over batches of indices.
        """
        indices = [idx for idx, _ in self.lengths]

        if self.shuffle:
            buckets = [
                indices[i : i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            ]
            for bucket in buckets:
                random.shuffle(bucket)
            random.shuffle(buckets)
            indices = [idx for bucket in buckets for idx in bucket]

        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        """
        Return the number of batches.

        :returns: Number of batches.
        :rtype: int
        """
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_bucketed_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_token_id: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with bucket batching for efficient padding (optional extension).

    :param dataset: PyTorch dataset.
    :type dataset: Dataset
    :param batch_size: Batch size.
    :type batch_size: int
    :param shuffle: Whether to shuffle data.
    :type shuffle: bool
    :param num_workers: Number of worker processes.
    :type num_workers: int
    :param pad_token_id: Padding token ID.
    :type pad_token_id: int
    :param pin_memory: Whether to pin memory.
    :type pin_memory: bool
    :returns: DataLoader with bucket batching.
    :rtype: DataLoader
    """
    batch_sampler = BucketBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    collate = lambda batch: collate_fn(batch, pad_token_id=pad_token_id)

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
    )


def compute_dataset_stats(dataset: Dataset) -> dict:
    """
    Compute length statistics about the dataset.

    :param dataset: Dataset to analyze.
    :type dataset: Dataset
    :returns: Dictionary with num_examples, min/max/mean/median length, and
        p50/p75/p90/p95/p99 percentiles.
    :rtype: dict
    """
    lengths = []
    for i in range(len(dataset)):
        input_ids, _ = dataset[i]
        lengths.append(len(input_ids))

    stats = {
        "num_examples": len(dataset),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2],
    }

    sorted_lengths = sorted(lengths)
    for p in [50, 75, 90, 95, 99]:
        idx = int(len(sorted_lengths) * p / 100)
        stats[f"p{p}_length"] = sorted_lengths[idx]

    return stats


def print_dataset_stats(dataset: Dataset) -> None:
    """
    Print dataset length statistics in a readable format.

    :param dataset: Dataset to analyze.
    :type dataset: Dataset
    """
    stats = compute_dataset_stats(dataset)

    print("Dataset Statistics:")
    print("=" * 50)
    print(f"Number of examples: {stats['num_examples']:,}")
    print(f"Sequence length:")
    print(f"  Min: {stats['min_length']}")
    print(f"  Max: {stats['max_length']}")
    print(f"  Mean: {stats['mean_length']:.1f}")
    print(f"  Median: {stats['median_length']}")
    print(f"Percentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th: {stats[f'p{p}_length']}")
    print("=" * 50)


def test_dataloader() -> None:
    """
    Test dataloader implementation with dummy data.
    """
    from ..tokenizer.base import CharacterTokenizer
    from .dataset import TextDataset

    texts = [
        "Short text.",
        "This is a medium length text example.",
        "Here is another example that is quite a bit longer than the others.",
        "Tiny.",
    ]

    tokenizer = CharacterTokenizer()
    tokenizer.train(texts)
    dataset = TextDataset(texts, tokenizer, max_seq_len=100)

    print("Testing DataLoader:")
    print(f"Dataset size: {len(dataset)}\n")

    print_dataset_stats(dataset)

    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    print(f"\nDataLoader with {len(dataloader)} batches\n")

    for batch_idx, (input_ids, target_ids, attention_mask) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Target shape: {target_ids.shape}")
        print(f"  Mask shape: {attention_mask.shape}")
        print(f"  Input IDs:\n{input_ids}")
        print(f"  Attention mask:\n{attention_mask}")
        if batch_idx == 0:
            break
