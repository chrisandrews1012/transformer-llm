"""
Base tokenizer interface and character-level tokenizer implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Union


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.

    Defines the interface for encoding text to token IDs, decoding token IDs
    back to text, managing special tokens, and saving/loading tokenizer state.
    """

    def __init__(self) -> None:
        """
        Initialize base tokenizer with empty vocabulary and special token maps.
        """
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {}

    @abstractmethod
    def train(self, texts: List[str], vocab_size: int, **kwargs) -> None:
        """
        Train the tokenizer on a corpus of texts.

        :param texts: List of text strings to train on.
        :type texts: List[str]
        :param vocab_size: Target vocabulary size.
        :type vocab_size: int
        """
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to a list of token IDs.

        :param text: Input text string.
        :type text: str
        :param add_special_tokens: Whether to prepend BOS and append EOS tokens.
        :type add_special_tokens: bool
        :returns: List of token IDs.
        :rtype: List[int]
        """
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs to a text string.

        :param token_ids: List of token IDs.
        :type token_ids: List[int]
        :param skip_special_tokens: Whether to omit special tokens from the output.
        :type skip_special_tokens: bool
        :returns: Decoded text string.
        :rtype: str
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the tokenizer state to a file.

        :param path: Path to write the tokenizer file.
        :type path: str
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load tokenizer state from a file.

        :param path: Path to read the tokenizer file.
        :type path: str
        """
        pass

    @property
    def vocab_size(self) -> int:
        """
        Return the vocabulary size.

        :returns: Number of tokens in the vocabulary.
        :rtype: int
        """
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        """
        Return the padding token ID.

        :returns: ID of the PAD token.
        :rtype: int
        """
        return self.special_tokens.get("pad_token_id", 0)

    @property
    def bos_token_id(self) -> int:
        """
        Return the beginning-of-sequence token ID.

        :returns: ID of the BOS token.
        :rtype: int
        """
        return self.special_tokens.get("bos_token_id", 1)

    @property
    def eos_token_id(self) -> int:
        """
        Return the end-of-sequence token ID.

        :returns: ID of the EOS token.
        :rtype: int
        """
        return self.special_tokens.get("eos_token_id", 2)

    @property
    def unk_token_id(self) -> int:
        """
        Return the unknown token ID.

        :returns: ID of the UNK token.
        :rtype: int
        """
        return self.special_tokens.get("unk_token_id", 3)

    def encode_batch(
        self, texts: List[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        """
        Encode a batch of texts to token ID lists.

        :param texts: List of text strings.
        :type texts: List[str]
        :param add_special_tokens: Whether to add BOS/EOS tokens to each sequence.
        :type add_special_tokens: bool
        :returns: List of token ID lists.
        :rtype: List[List[int]]
        """
        return [self.encode(text, add_special_tokens) for text in texts]

    def decode_batch(
        self, token_ids_batch: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode a batch of token ID sequences to text strings.

        :param token_ids_batch: List of token ID lists.
        :type token_ids_batch: List[List[int]]
        :param skip_special_tokens: Whether to omit special tokens from output.
        :type skip_special_tokens: bool
        :returns: List of decoded text strings.
        :rtype: List[str]
        """
        return [self.decode(ids, skip_special_tokens) for ids in token_ids_batch]


class CharacterTokenizer(BaseTokenizer):
    """
    Simple character-level tokenizer.

    Treats each character as a distinct token. Simple to understand but produces
    long sequences and does not capture subword information.
    """

    def __init__(self) -> None:
        """
        Initialize character tokenizer with empty character maps.
        """
        super().__init__()
        self.char_to_id = {}
        self.id_to_char = {}

    def train(self, texts: List[str], vocab_size: int = None, **kwargs) -> None:
        """
        Build vocabulary from the unique characters in texts.

        :param texts: List of text strings to collect characters from.
        :type texts: List[str]
        :param vocab_size: Unused for character tokenizer (all characters are included).
        :type vocab_size: int
        """
        chars = set()
        for text in texts:
            for char in text:
                chars.add(char)

        special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

        vocab_list = special_tokens + sorted(chars)
        self.char_to_id = {char: idx for idx, char in enumerate(vocab_list)}
        self.id_to_char = {idx: char for idx, char in enumerate(vocab_list)}

        self.special_tokens = {
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "unk_token_id": 3,
        }

        self.vocab = self.char_to_id
        self.inverse_vocab = self.id_to_char

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to a list of character token IDs.

        :param text: Input text.
        :type text: str
        :param add_special_tokens: Whether to prepend BOS and append EOS.
        :type add_special_tokens: bool
        :returns: List of token IDs.
        :rtype: List[int]
        """
        token_ids = []
        for char in text:
            token_id = self.char_to_id.get(char, self.special_tokens["unk_token_id"])
            token_ids.append(token_id)

        if add_special_tokens:
            token_ids = [self.special_tokens["bos_token_id"]] + token_ids + [self.special_tokens["eos_token_id"]]

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
        """
        chars = []
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue

            char = self.id_to_char.get(token_id, "<UNK>")
            chars.append(char)

        return "".join(chars)

    def save(self, path: str) -> None:
        """
        Save tokenizer vocabulary and special tokens to a JSON file.

        :param path: File path to write.
        :type path: str
        """
        import json

        data = {
            "char_to_id": self.char_to_id,
            "id_to_char": {int(k): v for k, v in self.id_to_char.items()},
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """
        Load tokenizer state from a JSON file.

        :param path: File path to read.
        :type path: str
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        self.special_tokens = data["special_tokens"]
        self.vocab = self.char_to_id
        self.inverse_vocab = self.id_to_char
