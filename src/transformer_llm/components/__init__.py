"""
Transformer Components Module

This module contains all the building blocks for transformer models.
"""

from .activation import GELU, GLU, ReLU, SiLU, get_activation
from .attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    ScaledDotProductAttention,
    create_causal_mask,
    create_padding_mask,
)
from .feedforward import (
    GLUFeedForward,
    MixtureOfExperts,
    PositionWiseFeedForward,
    create_ffn,
)
from .normalization import LayerNorm, PostNorm, PreNorm, RMSNorm
from .positional import (
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)
from .transformer import TransformerDecoderLayer, TransformerEncoderLayer

__all__ = [
    # Activation functions
    "ReLU",
    "GELU",
    "SiLU",
    "GLU",
    "get_activation",
    # Attention mechanisms
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "create_causal_mask",
    "create_padding_mask",
    # Feed-forward networks
    "PositionWiseFeedForward",
    "GLUFeedForward",
    "MixtureOfExperts",
    "create_ffn",
    # Normalization
    "LayerNorm",
    "RMSNorm",
    "PreNorm",
    "PostNorm",
    # Positional encoding
    "SinusoidalPositionalEncoding",
    "RotaryPositionalEmbedding",
    "LearnedPositionalEmbedding",
    # Transformer layers
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
]
