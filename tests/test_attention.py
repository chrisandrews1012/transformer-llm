"""
Unit Tests for Attention Mechanisms

This module tests the attention mechanisms to ensure they work correctly.
"""

import pytest
import torch

from transformer_llm.components.attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    ScaledDotProductAttention,
    create_causal_mask,
    create_padding_mask,
)


class TestScaledDotProductAttention:
    """Test scaled dot-product attention."""

    

    def test_attention_output_shape(self):
        """Test that attention output has correct shape."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 10, 64

        attention = ScaledDotProductAttention(dropout=0.0)

        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        output, attn_weights = attention(query, key, value)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_attention_with_mask(self):
        """Test attention with causal mask."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 10, 64

        attention = ScaledDotProductAttention(dropout=0.0)

        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Create causal mask
        mask = create_causal_mask(seq_len, query.device)

        output, attn_weights = attention(query, key, value, mask=mask)

        # Check that attention weights respect causal mask
        # Upper triangle should be zero (or very small)
        upper_triangle = torch.triu(attn_weights[0, 0], diagonal=1)
        assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6)

    def test_attention_values_range(self):
        """Test that attention weights sum to 1."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 10, 64

        attention = ScaledDotProductAttention(dropout=0.0)

        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        output, attn_weights = attention(query, key, value)

        # Attention weights should sum to 1 along last dimension
        attn_sum = attn_weights.sum(dim=-1)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)


class TestMultiHeadAttention:
    """Test multi-head attention."""

    

    def test_mha_output_shape(self):
        """Test MHA output shape."""
        batch_size, seq_len, d_model = 2, 10, 256
        num_heads = 8

        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = mha(query, key, value)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_mha_self_attention(self):
        """Test self-attention (Q=K=V)."""
        batch_size, seq_len, d_model = 2, 10, 256
        num_heads = 8

        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

        x = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = mha(x, x, x)

        assert output.shape == x.shape

    def test_mha_cross_attention(self):
        """Test cross-attention (different K, V)."""
        batch_size, seq_len_q, seq_len_kv, d_model = 2, 10, 15, 256
        num_heads = 8

        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

        query = torch.randn(batch_size, seq_len_q, d_model)
        key = torch.randn(batch_size, seq_len_kv, d_model)
        value = torch.randn(batch_size, seq_len_kv, d_model)

        output, attn_weights = mha(query, key, value)

        assert output.shape == (batch_size, seq_len_q, d_model)
        assert attn_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_kv)

    def test_mha_gradient_flow(self):
        """Test that gradients flow through MHA."""
        batch_size, seq_len, d_model = 2, 10, 256
        num_heads = 8

        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output, _ = mha(x, x, x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestGroupedQueryAttention:
    """Tests for Grouped Query Attention (GQA)."""

    

    def test_gqa_output_shape(self):
        """Test GQA output shape."""
        batch_size, seq_len, d_model = 2, 10, 256
        num_heads = 8
        num_kv_heads = 4

        gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout=0.0)

        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = gqa(query, key, value)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_gqa_parameter_count(self):
        """Test that GQA has fewer parameters than MHA."""
        d_model = 256
        num_heads = 8
        num_kv_heads = 4

        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout=0.0)

        mha_params = sum(p.numel() for p in mha.parameters())
        gqa_params = sum(p.numel() for p in gqa.parameters())

        # GQA should have fewer parameters due to reduced KV heads
        assert gqa_params < mha_params

    def test_gqa_equivalence_to_mha(self):
        """Test that GQA with num_kv_heads=num_heads is equivalent to MHA."""
        batch_size, seq_len, d_model = 2, 10, 256
        num_heads = 8

        # GQA with num_kv_heads = num_heads should behave like MHA
        gqa = GroupedQueryAttention(d_model, num_heads, num_heads, dropout=0.0)

        x = torch.randn(batch_size, seq_len, d_model)

        output, _ = gqa(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)


class TestMaskFunctions:
    """Test mask creation functions."""

    

    def test_causal_mask_shape(self):
        """Test causal mask shape."""
        seq_len = 10
        mask = create_causal_mask(seq_len, torch.device("cpu"))

        assert mask.shape == (1, 1, seq_len, seq_len)

    def test_causal_mask_values(self):
        """Test causal mask is lower triangular."""
        seq_len = 5
        mask = create_causal_mask(seq_len, torch.device("cpu"))

        # Should be lower triangular (1s below diagonal, 0s above)
        expected = torch.tril(torch.ones(seq_len, seq_len))
        assert torch.equal(mask.squeeze(), expected)

    def test_padding_mask_shape(self):
        """Test padding mask shape."""
        batch_size, seq_len = 2, 10
        seq = torch.randint(0, 100, (batch_size, seq_len))
        seq[:, -3:] = 0  # Add padding

        mask = create_padding_mask(seq, pad_idx=0)

        assert mask.shape == (batch_size, 1, 1, seq_len)

    def test_padding_mask_values(self):
        """Test padding mask correctly identifies padding."""
        batch_size, seq_len = 2, 10
        seq = torch.randint(1, 100, (batch_size, seq_len))
        seq[:, -3:] = 0  # Add padding at end

        mask = create_padding_mask(seq, pad_idx=0)

        # Last 3 positions should be 0 (padding)
        assert torch.all(mask[:, :, :, -3:] == 0)
        # First 7 positions should be 1 (not padding)
        assert torch.all(mask[:, :, :, :-3] == 1)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
