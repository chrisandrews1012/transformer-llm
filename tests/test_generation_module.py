"""Tests for the text generation module."""

import pytest
import torch

from transformer_llm.generation.generator import TextGenerator


class DummyTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self):
        self.id_to_piece = {1: "<BOS>", 2: "<EOS>", 3: "A", 4: "B"}

    def encode(self, text, add_special_tokens=True):
        ids = [3 if ch != "B" else 4 for ch in text] or [3]
        if add_special_tokens:
            return [self.bos_token_id] + ids
        return ids

    def decode(self, ids, skip_special_tokens=True):
        pieces = []
        for idx in ids:
            if skip_special_tokens and idx in {self.bos_token_id, self.eos_token_id}:
                continue
            pieces.append(self.id_to_piece.get(idx, "?"))
        return "".join(pieces)


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=5):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self.vocab_size), -100.0, device=input_ids.device)
        logits[:, -1, 3] = 10.0
        return logits, None


def test_text_generator_greedy_generation_runs():
    tokenizer = DummyTokenizer()
    model = DummyModel()
    generator = TextGenerator(model, tokenizer, torch.device("cpu"))

    outputs = generator.generate("A", max_new_tokens=3, do_sample=False)

    assert len(outputs) == 1
    assert isinstance(outputs[0], str)
    assert "A" in outputs[0]
