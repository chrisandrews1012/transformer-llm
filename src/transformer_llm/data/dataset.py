"""
Dataset classes for language modeling.
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    """
    Dataset for autoregressive language modeling.

    Creates (input, target) pairs where input = tokens[:-1] and
    target = tokens[1:], implementing teacher forcing during training.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512,
        split: str = "train",
        split_ratio: Tuple[float, float, float] = (0.9, 0.05, 0.05),
        split_seed: Optional[int] = 42,
        add_special_tokens: bool = True,
    ):
        """
        Initialize language modeling dataset.

        :param data_path: Path to the data file (JSONL format with "text" field).
        :type data_path: str
        :param tokenizer: Tokenizer used to encode text.
        :param max_seq_len: Maximum sequence length; longer examples are discarded.
        :type max_seq_len: int
        :param split: Data split to use ("train", "val", or "test").
        :type split: str
        :param split_ratio: Tuple of (train, val, test) fractions.
        :type split_ratio: Tuple[float, float, float]
        :param split_seed: Random seed for deterministic shuffling before splitting.
        :type split_seed: Optional[int]
        :param add_special_tokens: Whether to add BOS/EOS tokens during encoding.
        :type add_special_tokens: bool
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.split_seed = split_seed
        self.add_special_tokens = add_special_tokens

        print(f"Loading data from {data_path}...")
        self.examples = self._load_and_tokenize()

        if self.split_seed is not None:
            rng = random.Random(self.split_seed)
            rng.shuffle(self.examples)

        total = len(self.examples)
        train_size = int(total * split_ratio[0])
        val_size = int(total * split_ratio[1])
        val_end = train_size + val_size

        if split == "train":
            self.examples = self.examples[:train_size]
        elif split == "val":
            self.examples = self.examples[train_size:val_end]
        elif split == "test":
            self.examples = self.examples[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"Loaded {len(self.examples)} examples for {split} split")

    def _load_and_tokenize(self) -> List[List[int]]:
        """
        Load data from file and tokenize each example.

        :returns: List of tokenized examples (each example is a list of token IDs).
        :rtype: List[List[int]]
        """
        examples = []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "").strip()
                if not text:
                    continue

                token_ids = self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)

                if len(token_ids) < 2 or len(token_ids) > self.max_seq_len:
                    continue

                examples.append(token_ids)

        return examples

    def __len__(self) -> int:
        """
        Return the number of examples.

        :returns: Number of examples in the current split.
        :rtype: int
        """
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (input_ids, target_ids) pair.

        :param idx: Index of the example.
        :type idx: int
        :returns: Tuple of (input_ids, target_ids), each of shape (seq_len - 1,).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        token_ids = self.examples[idx]
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        return input_ids, target_ids


class TextDataset(Dataset):
    """
    Simple in-memory text dataset.

    Tokenizes all texts upfront and stores them in RAM. Suitable for small
    to medium datasets.
    """

    def __init__(self, texts: List[str], tokenizer, max_seq_len: int = 512):
        """
        Initialize text dataset.

        :param texts: List of text strings.
        :type texts: List[str]
        :param tokenizer: Tokenizer used to encode text.
        :param max_seq_len: Maximum sequence length; longer/shorter examples are
            discarded.
        :type max_seq_len: int
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        print(f"Tokenizing {len(texts)} texts...")
        self.examples = []
        for text in texts:
            token_ids = self.tokenizer.encode(text)

            if len(token_ids) < 2 or len(token_ids) > max_seq_len:
                continue

            self.examples.append(token_ids)

        print(f"Created {len(self.examples)} examples")

    def __len__(self) -> int:
        """
        Return the number of examples.

        :returns: Number of examples.
        :rtype: int
        """
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get input and target sequences for one example.

        :param idx: Index of the example.
        :type idx: int
        :returns: Tuple of (input_ids, target_ids), each of shape (seq_len - 1,).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        token_ids = self.examples[idx]
        return (
            torch.tensor(token_ids[:-1], dtype=torch.long),
            torch.tensor(token_ids[1:], dtype=torch.long),
        )


class StreamingDataset(Dataset):
    """
    Streaming dataset that reads data on-the-fly without loading all into memory.

    Useful for very large datasets; reads each example from disk on demand.
    Note: random access is slow (O(n) per call); prefer memory-mapped formats
    for production use.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512,
        max_examples: Optional[int] = None,
    ):
        """
        Initialize streaming dataset.

        :param data_path: Path to data file (JSONL format with "text" field).
        :type data_path: str
        :param tokenizer: Tokenizer used to encode text.
        :param max_seq_len: Maximum sequence length.
        :type max_seq_len: int
        :param max_examples: Maximum number of examples to use (None for all).
        :type max_examples: Optional[int]
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(data_path, "r", encoding="utf-8") as f:
            self.num_examples = sum(1 for _ in f)

        if max_examples is not None:
            self.num_examples = min(self.num_examples, max_examples)

        print(f"Streaming dataset with {self.num_examples} examples")

    def __len__(self) -> int:
        """
        Return the number of examples.

        :returns: Number of examples.
        :rtype: int
        """
        return self.num_examples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get example by reading from file.

        :param idx: Index of the example (requires scanning to line idx).
        :type idx: int
        :returns: Tuple of (input_ids, target_ids).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        with open(self.data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == idx:
                    data = json.loads(line)
                    text = data.get("text", "")
                    token_ids = self.tokenizer.encode(text)
                    break

        input_ids = token_ids[:-1]  # type: ignore
        target_ids = token_ids[1:]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def prepare_data(
    data_path: str,
    output_path: str,
    tokenizer,
    max_examples: Optional[int] = None,
) -> None:
    """
    Pre-tokenize data and save to disk to speed up training.

    :param data_path: Path to raw data file (JSONL format).
    :type data_path: str
    :param output_path: Path to save tokenized data (JSON format).
    :type output_path: str
    :param tokenizer: Tokenizer used to encode text.
    :param max_examples: Maximum number of examples to process (None for all).
    :type max_examples: Optional[int]
    """
    print(f"Preparing data from {data_path}...")

    tokenized_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break

            data = json.loads(line)
            text = data["text"]
            token_ids = tokenizer.encode(text)
            tokenized_data.append(token_ids)

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1} examples")

    with open(output_path, "w") as f:
        json.dump(tokenized_data, f)

    print(f"Saved {len(tokenized_data)} tokenized examples to {output_path}")


def test_dataset() -> None:
    """
    Test dataset implementation with dummy data.
    """
    from ..tokenizer.base import CharacterTokenizer

    texts = [
        "Hello world, this is a test.",
        "Another example sentence here.",
        "Machine learning is fascinating!",
    ]

    tokenizer = CharacterTokenizer()
    tokenizer.train(texts)

    dataset = TextDataset(texts, tokenizer, max_seq_len=50)

    print(f"Dataset size: {len(dataset)}")

    input_ids, target_ids = dataset[0]
    print(f"\nFirst example:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_ids.shape}")
    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Target IDs: {target_ids.tolist()}")

    input_text = tokenizer.decode(input_ids.tolist())
    target_text = tokenizer.decode(target_ids.tolist())
    print(f"Input text: {input_text}")
    print(f"Target text: {target_text}")
