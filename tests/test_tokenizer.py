"""Tests for BPE tokenizer: training, merge behavior, and save/load round-trip."""

import tempfile

import pytest

from transformer_llm.tokenizer.bpe import BPETokenizer


class TestBPETokenizer:

    def _make_tokenizer(self) -> BPETokenizer:
        texts = [
            "hello world hello world",
            "hello there general kenobi",
            "low lower lowest",
            "machine learning learning machine",
        ]
        tokenizer = BPETokenizer()
        tokenizer.train(texts, vocab_size=40, min_frequency=1)
        return tokenizer

    def test_bpe_training_creates_merges(self):
        tokenizer = self._make_tokenizer()

        assert tokenizer.vocab_size > 4
        assert tokenizer.merges
        assert tokenizer.merge_priority

    def test_encode_decode_round_trip(self):
        tokenizer = self._make_tokenizer()

        token_ids = tokenizer.encode("hello world", add_special_tokens=True)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)

        assert token_ids[0] == tokenizer.bos_token_id
        assert token_ids[-1] == tokenizer.eos_token_id
        assert "hello" in decoded
        assert "world" in decoded

    def test_save_and_load_preserves_merge_priority(self):
        tokenizer = self._make_tokenizer()

        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            tokenizer.save(f.name)

            loaded = BPETokenizer()
            loaded.load(f.name)

        assert loaded.token_to_id == tokenizer.token_to_id
        assert loaded.merges == tokenizer.merges
        assert loaded.merge_priority == tokenizer.merge_priority
