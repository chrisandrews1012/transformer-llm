"""
Byte-level BPE tokenizer backed by the Hugging Face tokenizers library.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from tokenizers import Tokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

from .base import BaseTokenizer


class ByteLevelBPETokenizer(BaseTokenizer):
    """
    Wrapper around a Hugging Face byte-level BPE tokenizer.

    Uses a ByteLevel pre-tokenizer and decoder so punctuation, whitespace,
    casing, and line breaks are preserved more faithfully than a hand-written
    BPE implementation.
    """

    TOKENIZER_TYPE = "byte_bpe"

    def __init__(self) -> None:
        """
        Initialize byte-level BPE tokenizer with empty state.
        """
        super().__init__()
        self.tokenizer: Tokenizer | None = None
        self.token_to_id: dict = {}
        self.id_to_token: dict = {}

    def _find_token_id(self, *candidates: str, default: int | None = None) -> int:
        """
        Return the first token ID found in the vocabulary from a list of candidates.

        Allows reuse of tokenizers whose special token names differ from the
        project defaults (e.g. MiniMind uses im_start/im_end instead of BOS/EOS).

        :param candidates: Token strings to look up in order.
        :param default: Fallback ID if no candidate is found.
        :type default: int | None
        :returns: Token ID of the first matching candidate.
        :rtype: int
        :raises KeyError: If no candidate is found and default is None.
        """
        for token in candidates:
            if token in self.token_to_id:
                return self.token_to_id[token]
        if default is not None:
            return default
        raise KeyError(f"Could not find any of these special tokens: {candidates}")

    def _sync_from_tokenizer(self) -> None:
        """
        Sync internal vocabulary and special token maps from the HuggingFace tokenizer.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been initialized")

        vocab = self.tokenizer.get_vocab()
        self.token_to_id = dict(vocab)
        self.id_to_token = {idx: token for token, idx in vocab.items()}
        self.vocab = self.token_to_id
        self.inverse_vocab = self.id_to_token
        self.special_tokens = {
            "pad_token_id": self._find_token_id("<PAD>", "<|endoftext|>", default=0),
            "bos_token_id": self._find_token_id("<BOS>", "<|im_start|>", "<s>", default=1),
            "eos_token_id": self._find_token_id("<EOS>", "<|im_end|>", "</s>", default=2),
            "unk_token_id": self._find_token_id("<UNK>", "<|endoftext|>", "<unk>", default=0),
        }

    def train(
        self,
        texts: List[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
        **kwargs,
    ) -> None:
        """
        Train a byte-level BPE tokenizer on a list of texts.

        :param texts: Training texts.
        :type texts: List[str]
        :param vocab_size: Target vocabulary size.
        :type vocab_size: int
        :param min_frequency: Minimum pair frequency to apply a merge.
        :type min_frequency: int
        """
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)
        self.tokenizer = tokenizer
        self._sync_from_tokenizer()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to a list of token IDs.

        :param text: Input text string.
        :type text: str
        :param add_special_tokens: Whether to prepend BOS and append EOS.
        :type add_special_tokens: bool
        :returns: List of token IDs.
        :rtype: List[int]
        :raises ValueError: If the tokenizer has not been trained or loaded.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been trained or loaded")

        token_ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back to text.

        :param token_ids: List of token IDs.
        :type token_ids: List[int]
        :param skip_special_tokens: Whether to omit special tokens from the output.
        :type skip_special_tokens: bool
        :returns: Decoded text string.
        :rtype: str
        :raises ValueError: If the tokenizer has not been trained or loaded.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been trained or loaded")

        ids = token_ids
        if skip_special_tokens:
            special_ids = set(self.special_tokens.values())
            ids = [token_id for token_id in token_ids if token_id not in special_ids]

        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, path: str) -> None:
        """
        Save the tokenizer to a JSON file.

        :param path: File path to write.
        :type path: str
        :raises ValueError: If the tokenizer has not been trained or loaded.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been trained or loaded")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path_obj))

    def load(self, path: str) -> None:
        """
        Load a tokenizer from a JSON file.

        :param path: File path to read.
        :type path: str
        """
        self.tokenizer = Tokenizer.from_file(path)
        self._sync_from_tokenizer()
