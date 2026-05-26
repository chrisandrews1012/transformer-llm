"""
Unit Tests for Language Model

This module tests the complete transformer language model.
"""

import pytest
import torch

from transformer_llm.model.config import ModelConfig, get_small_config
from transformer_llm.model.language_model import TransformerLanguageModel


class TestModelConfig:
    """Test model configuration."""

    def test_config_creation(self):
        """Test creating a valid configuration."""
        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            num_layers=4,
            num_heads=8,
            d_ff=1024,
        )

        assert config.vocab_size == 1000
        assert config.d_model == 256
        assert config.head_dim == 32  # d_model // num_heads

    def test_config_validation(self):
        """Test configuration validation."""
        # d_model must be divisible by num_heads
        with pytest.raises(ValueError):
            ModelConfig(
                vocab_size=1000,
                d_model=257,  # Not divisible by 8
                num_heads=8,
            )

    def test_gqa_config_validation(self):
        """Test GQA configuration validation."""
        # num_heads must be divisible by num_kv_heads
        with pytest.raises(ValueError):
            ModelConfig(
                vocab_size=1000,
                d_model=256,
                num_heads=8,
                attention_type="gqa",
                num_kv_heads=3,  # 8 not divisible by 3
            )
    

    def test_config_serialization(self):
        """Test saving and loading configuration."""
        import tempfile

        config = get_small_config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.save(f.name)
            loaded_config = ModelConfig.load(f.name)

        assert config.to_dict() == loaded_config.to_dict()


class TestTransformerLanguageModel:
    """Test transformer language model."""

    def test_model_creation(self):
        """Test creating a model."""
        config = get_small_config()
        model = TransformerLanguageModel(config)

        assert model is not None
        assert model.config == config

    def test_model_forward_pass(self):
        """Test forward pass through model."""
        config = get_small_config()
        model = TransformerLanguageModel(config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, hidden_states = model(input_ids, return_hidden_states=True)

        # Check output shape
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

        # Check hidden states
        assert len(hidden_states) == config.num_layers

    def test_model_with_attention_mask(self):
        """Test model with attention mask."""
        config = get_small_config()
        model = TransformerLanguageModel(config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -3:] = 0  # Mask last 3 tokens

        logits, _ = model(input_ids, attention_mask=attention_mask)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        config = get_small_config()
        model = TransformerLanguageModel(config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, _ = model(input_ids)
        loss = logits.sum()
        loss.backward()

        # Check that gradients exist for model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_model_parameter_count(self):
        """Test parameter counting."""
        config = get_small_config()
        model = TransformerLanguageModel(config)

        total_params = model.get_num_params()
        non_embedding_params = model.get_num_params(non_embedding=True)

        assert total_params > 0
        assert non_embedding_params > 0
        assert total_params > non_embedding_params

    def test_tied_embeddings(self):
        """Test that tied embeddings share weights."""
        config = get_small_config()
        config.tie_word_embeddings = True

        model = TransformerLanguageModel(config)

        # Check that embedding and lm_head share weights
        assert model.token_embedding.weight is model.lm_head.weight

    def test_different_architectures(self):
        """Test GQA, RoPE, and RMSNorm architecture variants."""
        base_config = get_small_config()

        # Test with GQA
        config_gqa = ModelConfig(**{**base_config.to_dict(), "attention_type": "gqa", "num_kv_heads": 2})
        model_gqa = TransformerLanguageModel(config_gqa)

        # Test with RoPE
        config_rope = ModelConfig(**{**base_config.to_dict(), "pos_encoding_type": "rope"})
        model_rope = TransformerLanguageModel(config_rope)

        # Test with RMSNorm
        config_rms = ModelConfig(**{**base_config.to_dict(), "norm_type": "rmsnorm"})
        model_rms = TransformerLanguageModel(config_rms)

        # All should create successfully
        assert model_gqa is not None
        assert model_rope is not None
        assert model_rms is not None

        # Test forward pass for each
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))

        for model in [model_gqa, model_rope, model_rms]:
            logits, _ = model(input_ids)
            assert logits.shape == (batch_size, seq_len, base_config.vocab_size)

    


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
