"""
Byte Pair Encoding (BPE) tokenizer implementation.
"""

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .base import BaseTokenizer
from tqdm import tqdm


class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding (BPE) Tokenizer.

    Iteratively merges the most frequent adjacent token pair until the target
    vocabulary size is reached. Creates a subword vocabulary that balances
    between character-level and word-level tokenization.

    Used in: GPT-2, GPT-3, RoBERTa, and many other models.
    """

    def __init__(self) -> None:
        """
        Initialize BPE tokenizer with empty merge tables and vocabulary.
        """
        super().__init__()
        self.merges: Dict[Tuple[str, str], str] = {}
        self.merge_priority: Dict[Tuple[str, str], int] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def train(
        self,
        texts: List[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
        **kwargs,
    ) -> None:
        """
        Train BPE tokenizer on a corpus.

        :param texts: List of text strings to train on.
        :type texts: List[str]
        :param vocab_size: Target vocabulary size.
        :type vocab_size: int
        :param min_frequency: Minimum pair frequency required to apply a merge.
        :type min_frequency: int
        """
        print(f"Training BPE tokenizer with target vocab_size={vocab_size}")

        vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        self.special_tokens = {
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "unk_token_id": 3,
        }

        print("Pre-tokenizing texts...")
        word_freqs: Counter = Counter()

        for text in tqdm(texts, desc="Pre-tokenizing", unit="texts"):
            words = re.findall(r'\w+', text.lower())
            for word in words:
                word_freqs[word] += 1

        print(f"Found {len(word_freqs)} unique words")

        print("Building initial character vocabulary...")
        chars = set()
        for word in word_freqs.keys():
            for char in word:
                chars.add(char)
        chars.add("</w>")

        vocab.extend(sorted(chars))

        word_splits: Dict[str, List[str]] = {}
        for word, freq in word_freqs.items():
            word_splits[word] = list(word) + ["</w>"]

        print(f"Learning {vocab_size - len(vocab)} merges...")
        num_merges = vocab_size - len(vocab)

        for merge_idx in tqdm(range(num_merges), desc="Learning merges", unit="merge"):
            pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
            for word, freq in word_freqs.items():
                split = word_splits[word]
                for j in range(len(split) - 1):
                    pair = (split[j], split[j+1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)

            if pair_freqs[best_pair] < min_frequency:
                break

            merged_token = best_pair[0] + best_pair[1]

            vocab.append(merged_token)
            self.merges[best_pair] = merged_token
            self.merge_priority[best_pair] = merge_idx

            for word in word_splits:
                split = word_splits[word]
                new_split = []
                j = 0
                while j < len(split):
                    if j < len(split) - 1 and (split[j], split[j+1]) == best_pair:
                        new_split.append(merged_token)
                        j += 2
                    else:
                        new_split.append(split[j])
                        j += 1
                word_splits[word] = new_split

        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}
        self.vocab = self.token_to_id
        self.inverse_vocab = self.id_to_token

        print(f"Training complete. Final vocab size: {len(vocab)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using the learned BPE merge rules.

        :param word: Word to tokenize (without special tokens).
        :type word: str
        :returns: List of subword tokens with </w> appended to the last character.
        :rtype: List[str]
        """
        tokens = list(word) + ["</w>"]

        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]

            mergeable_pairs = [pair for pair in pairs if pair in self.merges]

            if not mergeable_pairs:
                break

            best_pair = min(mergeable_pairs, key=lambda p: self.merge_priority[p])

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(self.merges[best_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs using BPE.

        :param text: Input text.
        :type text: str
        :param add_special_tokens: Whether to prepend BOS and append EOS tokens.
        :type add_special_tokens: bool
        :returns: List of token IDs.
        :rtype: List[int]
        """
        words = re.findall(r'\w+', text.lower())

        tokens = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)

        token_ids = []
        for token in tokens:
            token_id = self.token_to_id.get(token, self.special_tokens["unk_token_id"])
            token_ids.append(token_id)

        if add_special_tokens:
            token_ids = [self.special_tokens["bos_token_id"]] + token_ids + [self.special_tokens["eos_token_id"]]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        :param token_ids: List of token IDs.
        :type token_ids: List[int]
        :param skip_special_tokens: Whether to omit special tokens from the output.
        :type skip_special_tokens: bool
        :returns: Decoded text string.
        :rtype: str
        """
        tokens = []
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue

            token = self.id_to_token.get(token_id, "<UNK>")
            tokens.append(token)

        text = "".join(tokens).replace("</w>", " ")

        return text.strip()

    def save(self, path: str) -> None:
        """
        Save tokenizer state to a JSON file.

        :param path: File path to write.
        :type path: str
        """
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()},
            "merges": {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            "merge_priority": {f"{k[0]}|||{k[1]}": v for k, v in self.merge_priority.items()},
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str) -> None:
        """
        Load tokenizer state from a JSON file.

        :param path: File path to read.
        :type path: str
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        self.merges = {
            tuple(k.split("|||")): v for k, v in data["merges"].items()
        }
        self.merge_priority = {
            tuple(k.split("|||")): v for k, v in data.get("merge_priority", {}).items()
        }
        self.special_tokens = data["special_tokens"]
        self.vocab = self.token_to_id
        self.inverse_vocab = self.id_to_token
        print(f"Tokenizer loaded from {path}")


def create_hf_bpe_tokenizer(
    texts: List[str],
    vocab_size: int = 8000,
    save_path: str = None,
):
    """
    Create a BPE tokenizer using the HuggingFace tokenizers library.

    :param texts: List of texts to train on.
    :type texts: List[str]
    :param vocab_size: Target vocabulary size.
    :type vocab_size: int
    :param save_path: Optional path to save the tokenizer JSON.
    :type save_path: str
    :returns: HuggingFace Tokenizer object (not a BPETokenizer wrapper).
    :raises ImportError: If the HuggingFace tokenizers library is not installed.
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer
    except ImportError:
        raise ImportError(
            "HuggingFace tokenizers library not installed. "
            "Install with: pip install tokenizers"
        )

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
    )

    def batch_iterator(batch_size=1000):
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    if save_path:
        tokenizer.save(save_path)

    return tokenizer
