"""
Attention mechanisms: Scaled Dot-Product, Multi-Head Attention (MHA), causal/padding masks,
and Grouped Query Attention (GQA).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.
    """

    def __init__(self, dropout: float = 0.1):
        """
        Initialize scaled dot-product attention.

        :param dropout: Dropout probability applied to attention weights.
        :type dropout: float
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        :param query: Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
        :type query: torch.Tensor
        :param key: Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
        :type key: torch.Tensor
        :param value: Value tensor of shape (batch_size, num_heads, seq_len_v, head_dim).
        :type value: torch.Tensor
        :param mask: Optional mask of shape (batch_size, 1, seq_len_q, seq_len_k) or
            (batch_size, num_heads, seq_len_q, seq_len_k). Positions with mask == 0
            are masked out.
        :type mask: Optional[torch.Tensor]
        :returns: Tuple of (output, attention_weights). Output has shape
            (batch_size, num_heads, seq_len_q, head_dim); attention_weights has shape
            (batch_size, num_heads, seq_len_q, seq_len_k).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Projects inputs to Q, K, V, splits into multiple heads, applies scaled
    dot-product attention in parallel, then concatenates and projects back.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        :param d_model: Dimension of the model (embedding dimension).
        :type d_model: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param dropout: Dropout probability.
        :type dropout: float
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim) and transpose.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :type x: torch.Tensor
        :returns: Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        :rtype: torch.Tensor
        """
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads back into a single dimension.

        :param x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim).
        :type x: torch.Tensor
        :returns: Tensor of shape (batch_size, seq_len, d_model).
        :rtype: torch.Tensor
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.

        :param query: Query tensor of shape (batch_size, seq_len_q, d_model).
        :type query: torch.Tensor
        :param key: Key tensor of shape (batch_size, seq_len_k, d_model).
        :type key: torch.Tensor
        :param value: Value tensor of shape (batch_size, seq_len_v, d_model).
        :type value: torch.Tensor
        :param mask: Optional attention mask tensor.
        :type mask: Optional[torch.Tensor]
        :returns: Tuple of (output, attention_weights). Output has shape
            (batch_size, seq_len_q, d_model); attention_weights has shape
            (batch_size, num_heads, seq_len_q, seq_len_k).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        attn_output, attention_weights = self.attention(Q, K, V, mask)

        attn_output = self._combine_heads(attn_output)
        output = self.out_proj(attn_output)

        return output, attention_weights


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) mechanism (optional extension).

    Reduces KV-head count relative to query heads, shrinking the KV cache
    during inference. Each KV head is shared by (num_heads / num_kv_heads)
    query heads.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
    ):
        """
        Initialize grouped query attention.

        :param d_model: Dimension of the model (embedding dimension).
        :type d_model: int
        :param num_heads: Number of query heads.
        :type num_heads: int
        :param num_kv_heads: Number of key-value heads (must divide num_heads evenly).
        :type num_kv_heads: int
        :param dropout: Dropout probability.
        :type dropout: float
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim) and transpose.

        :param x: Input tensor of shape (batch_size, seq_len, num_heads * head_dim).
        :type x: torch.Tensor
        :param num_heads: Number of heads to split into.
        :type num_heads: int
        :returns: Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        :rtype: torch.Tensor
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat key or value heads to match the number of query heads.

        :param x: Input tensor of shape (batch_size, num_kv_heads, seq_len, head_dim).
        :type x: torch.Tensor
        :returns: Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        :rtype: torch.Tensor
        """
        batch_size, num_kv_heads, seq_len, head_dim = x.size()

        if self.num_queries_per_kv == 1:
            return x

        x = x.unsqueeze(2)
        x = x.repeat_interleave(self.num_queries_per_kv, dim=2)
        x = x.reshape(batch_size, self.num_heads, seq_len, head_dim)

        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads back into a single dimension.

        :param x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim).
        :type x: torch.Tensor
        :returns: Tensor of shape (batch_size, seq_len, d_model).
        :rtype: torch.Tensor
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of grouped query attention.

        :param query: Query tensor of shape (batch_size, seq_len_q, d_model).
        :type query: torch.Tensor
        :param key: Key tensor of shape (batch_size, seq_len_k, d_model).
        :type key: torch.Tensor
        :param value: Value tensor of shape (batch_size, seq_len_v, d_model).
        :type value: torch.Tensor
        :param mask: Optional attention mask tensor.
        :type mask: Optional[torch.Tensor]
        :returns: Tuple of (output, attention_weights). Output has shape
            (batch_size, seq_len_q, d_model).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = self._split_heads(Q, self.num_heads)
        K = self._split_heads(K, self.num_kv_heads)
        V = self._split_heads(V, self.num_kv_heads)

        K = self._repeat_kv(K)
        V = self._repeat_kv(V)

        attn_output, attention_weights = self.attention(Q, K, V, mask)

        attn_output = self._combine_heads(attn_output)
        output = self.out_proj(attn_output)

        return output, attention_weights


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (lower triangular) mask for autoregressive attention.

    Ensures position i can only attend to positions <= i.

    :param seq_len: Sequence length.
    :type seq_len: int
    :param device: Device to create the mask on.
    :type device: torch.device
    :returns: Mask tensor of shape (1, 1, seq_len, seq_len) with 1s in the
        lower triangle and 0s above.
    :rtype: torch.Tensor
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create a padding mask from a sequence of token indices.

    :param seq: Input sequence of shape (batch_size, seq_len) with token indices.
    :type seq: torch.Tensor
    :param pad_idx: Index of the padding token.
    :type pad_idx: int
    :returns: Mask tensor of shape (batch_size, 1, 1, seq_len) with 1s for real
        tokens and 0s for padding.
    :rtype: torch.Tensor
    """
    mask = (seq != pad_idx)
    return mask.view(seq.size(0), 1, 1, seq.size(1))
