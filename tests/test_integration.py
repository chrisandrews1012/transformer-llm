"""Integration tests for training, checkpoint persistence, and generation."""

import tempfile

import pytest
import torch

from transformer_llm.model.config import get_small_config
from transformer_llm.model.language_model import TransformerLanguageModel


class TestIntegration:
    """Integration tests for training-adjacent model workflows."""

    def test_model_generation(self):
        """Test text generation."""
        config = get_small_config()
        model = TransformerLanguageModel(config)
        model.eval()

        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        generated = model.generate(input_ids, max_new_tokens=10, do_sample=False)

        # Generated sequence should be longer than input
        assert generated.shape[1] == seq_len + 10

    def test_training_step(self):
        """Test a single optimizer-backed training step."""
        config = get_small_config()
        model = TransformerLanguageModel(config)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, _ = model(input_ids)

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            target_ids.view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)

    def test_save_and_load_model(self):
        """Test saving and loading model state."""
        config = get_small_config()
        model = TransformerLanguageModel(config)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)

            loaded_model = TransformerLanguageModel(config)
            loaded_model.load_state_dict(torch.load(f.name))

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            output1, _ = model(input_ids)
            output2, _ = loaded_model(input_ids)

        assert torch.allclose(output1, output2, atol=1e-5)
