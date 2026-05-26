"""
Positional encodings: sinusoidal, Rotary Position Embedding (RoPE), and learned embeddings.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention is All You Need".

    Uses fixed sine and cosine functions of different frequencies to encode
    position. Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)),
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize sinusoidal positional encoding.

        :param d_model: Dimension of the model (embedding dimension).
        :type d_model: int
        :param max_len: Maximum sequence length to pre-compute encodings for.
        :type max_len: int
        :param dropout: Dropout probability.
        :type dropout: float
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :type x: torch.Tensor
        :returns: Tensor of shape (batch_size, seq_len, d_model) with positional
            encoding added and dropout applied.
        :rtype: torch.Tensor
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) (optional extension).

    Encodes position by rotating Q and K representations in the complex plane.
    Naturally captures relative positions and can extrapolate to longer sequences.
    From "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    """

    def __init__(self, dim: int, max_len: int = 2048, base: float = 10000.0):
        """
        Initialize rotary positional embedding.

        :param dim: Dimension of each attention head.
        :type dim: int
        :param max_len: Maximum sequence length.
        :type max_len: int
        :param base: Base for the geometric progression (default 10000).
        :type base: float
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(max_len)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        """
        Pre-compute cos and sin values for all positions up to seq_len.

        :param seq_len: Sequence length to cache.
        :type seq_len: int
        """
        self.max_seq_len_cached = seq_len

        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dimensions of the input.

        :param x: Input tensor of shape (..., dim).
        :type x: torch.Tensor
        :returns: Rotated tensor of shape (..., dim).
        :rtype: torch.Tensor
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.

        :param q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
        :type q: torch.Tensor
        :param k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim).
        :type k: torch.Tensor
        :returns: Tuple of (rotated_q, rotated_k) with the same shapes as inputs.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        seq_len = q.shape[2]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned Positional Embedding (optional extension).

    Treats position embeddings as parameters learned during training.
    Simpler than sinusoidal but requires the training data to cover all positions.
    """

    def __init__(self, max_len: int, d_model: int):
        """
        Initialize learned positional embedding.

        :param max_len: Maximum sequence length.
        :type max_len: int
        :param d_model: Dimension of the model.
        :type d_model: int
        """
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings to input.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :type x: torch.Tensor
        :returns: Tensor with positional embeddings added, same shape as input.
        :rtype: torch.Tensor
        """
        batch_size, seq_len, d_model = x.size()
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)


def visualize_positional_encoding(
    encoding_type: str = "sinusoidal",
    d_model: int = 128,
    max_len: int = 100,
) -> torch.Tensor:
    """
    Visualize positional encodings and save a figure.

    :param encoding_type: Type of encoding ("sinusoidal" or "learned").
    :type encoding_type: str
    :param d_model: Model dimension.
    :type d_model: int
    :param max_len: Maximum sequence length to visualize.
    :type max_len: int
    :returns: Positional encoding matrix of shape (max_len, d_model).
    :rtype: torch.Tensor
    :raises ValueError: If visualization is not implemented for encoding_type.
    """
    import matplotlib.pyplot as plt

    if encoding_type == "sinusoidal":
        pe_module = SinusoidalPositionalEncoding(d_model, max_len)
        pe = pe_module.pe.squeeze(0).numpy()
    else:
        raise ValueError(f"Visualization not implemented for {encoding_type}")

    plt.figure(figsize=(15, 5))
    plt.imshow(pe, cmap="RdBu", aspect="auto")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.colorbar()
    plt.title(f"{encoding_type.capitalize()} Positional Encoding")
    plt.tight_layout()
    plt.savefig(f"positional_encoding_{encoding_type}.png")
    plt.close()

    return pe
