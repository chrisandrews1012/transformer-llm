"""
Transformer decoder (and encoder) layers: causal masking, residual connections,
and configurable norm/FFN/attention composition.
"""

from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, GroupedQueryAttention, create_causal_mask
from .feedforward import create_ffn
from .normalization import LayerNorm, RMSNorm


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer (optional extension).

    Consists of multi-head self-attention and a feed-forward network, each
    wrapped with a residual connection and normalization. Supports pre-norm
    (modern, more stable) and post-norm (original Transformer) layouts.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: str = "layernorm",
        norm_position: str = "pre",
        attention_type: str = "mha",
        num_kv_heads: Optional[int] = None,
        ffn_type: str = "standard",
    ):
        """
        Initialize transformer encoder layer.

        :param d_model: Dimension of the model.
        :type d_model: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param d_ff: Dimension of the feed-forward network.
        :type d_ff: int
        :param dropout: Dropout probability.
        :type dropout: float
        :param activation: Activation function for FFN.
        :type activation: str
        :param norm_type: Type of normalization ("layernorm" or "rmsnorm").
        :type norm_type: str
        :param norm_position: Position of normalization ("pre" or "post").
        :type norm_position: str
        :param attention_type: Type of attention ("mha" or "gqa").
        :type attention_type: str
        :param num_kv_heads: Number of key-value heads (for GQA only).
        :type num_kv_heads: Optional[int]
        :param ffn_type: Type of feed-forward network.
        :type ffn_type: str
        """
        super().__init__()
        self.d_model = d_model
        self.norm_position = norm_position

        if attention_type == "mha":
            self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        elif attention_type == "gqa":
            if num_kv_heads is None:
                num_kv_heads = num_heads // 2
            self.attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.ffn = create_ffn(ffn_type, d_model, d_ff, dropout, activation)

        if norm_type == "layernorm":
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
        elif norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of encoder layer.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :type x: torch.Tensor
        :param mask: Optional attention mask.
        :type mask: Optional[torch.Tensor]
        :returns: Output tensor of shape (batch_size, seq_len, d_model).
        :rtype: torch.Tensor
        """
        if self.norm_position == "pre":
            normed = self.norm1(x)
            attn_output, _ = self.attention(normed, normed, normed, mask)
            x = x + self.dropout(attn_output)

            normed = self.norm2(x)
            ffn_output = self.ffn(normed)
            x = x + self.dropout(ffn_output)

        else:  # post-norm
            attn_output, _ = self.attention(x, x, x, mask)
            x = x + self.dropout(attn_output)
            x = self.norm1(x)

            ffn_output = self.ffn(x)
            x = x + self.dropout(ffn_output)
            x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.

    Consists of masked causal self-attention, an optional cross-attention block,
    and a feed-forward network. Each sub-layer has a residual connection and
    normalization. Causal masking ensures position i can only attend to positions
    <= i, which is required for autoregressive generation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: str = "layernorm",
        norm_position: str = "pre",
        attention_type: str = "mha",
        num_kv_heads: Optional[int] = None,
        ffn_type: str = "standard",
        use_cross_attention: bool = False,
    ):
        """
        Initialize transformer decoder layer.

        :param d_model: Dimension of the model.
        :type d_model: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param d_ff: Dimension of the feed-forward network.
        :type d_ff: int
        :param dropout: Dropout probability.
        :type dropout: float
        :param activation: Activation function for FFN.
        :type activation: str
        :param norm_type: Type of normalization ("layernorm" or "rmsnorm").
        :type norm_type: str
        :param norm_position: Position of normalization ("pre" or "post").
        :type norm_position: str
        :param attention_type: Type of attention ("mha" or "gqa").
        :type attention_type: str
        :param num_kv_heads: Number of key-value heads (for GQA only).
        :type num_kv_heads: Optional[int]
        :param ffn_type: Type of feed-forward network.
        :type ffn_type: str
        :param use_cross_attention: Whether to include a cross-attention block
            (for encoder-decoder models).
        :type use_cross_attention: bool
        """
        super().__init__()
        self.d_model = d_model
        self.norm_position = norm_position
        self.use_cross_attention = use_cross_attention

        if attention_type == "mha":
            self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        elif attention_type == "gqa":
            if num_kv_heads is None:
                num_kv_heads = num_heads // 2
            self.self_attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        if use_cross_attention:
            if attention_type == "mha":
                self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
            elif attention_type == "gqa":
                self.cross_attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)

        self.ffn = create_ffn(ffn_type, d_model, d_ff, dropout, activation)

        if norm_type == "layernorm":
            self.norm1 = LayerNorm(d_model)
            if use_cross_attention:
                self.norm2 = LayerNorm(d_model)
            self.norm3 = LayerNorm(d_model)
        elif norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
            if use_cross_attention:
                self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of decoder layer.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :type x: torch.Tensor
        :param encoder_output: Optional encoder output for cross-attention, shape
            (batch_size, src_seq_len, d_model).
        :type encoder_output: Optional[torch.Tensor]
        :param self_attn_mask: Causal mask for self-attention.
        :type self_attn_mask: Optional[torch.Tensor]
        :param cross_attn_mask: Optional mask for cross-attention.
        :type cross_attn_mask: Optional[torch.Tensor]
        :returns: Output tensor of shape (batch_size, seq_len, d_model).
        :rtype: torch.Tensor
        """
        if self_attn_mask is None:
            self_attn_mask = create_causal_mask(x.size(1), x.device)

        if self.norm_position == "pre":
            normed = self.norm1(x)
            attn_output, _ = self.self_attention(normed, normed, normed, self_attn_mask)
            x = x + self.dropout(attn_output)

            if self.use_cross_attention and encoder_output is not None:
                normed = self.norm2(x)
                cross_attn_output, _ = self.cross_attention(
                    query=normed, key=encoder_output, value=encoder_output, mask=cross_attn_mask
                )
                x = x + self.dropout(cross_attn_output)

            normed = self.norm3(x)
            ffn_output = self.ffn(normed)
            x = x + self.dropout(ffn_output)

        else:  # post-norm
            attn_output, _ = self.self_attention(x, x, x, self_attn_mask)
            x = x + self.dropout(attn_output)
            x = self.norm1(x)

            if self.use_cross_attention and encoder_output is not None:
                cross_attn_output, _ = self.cross_attention(
                    query=x, key=encoder_output, value=encoder_output, mask=cross_attn_mask
                )
                x = x + self.dropout(cross_attn_output)
                x = self.norm2(x)

            ffn_output = self.ffn(x)
            x = x + self.dropout(ffn_output)
            x = self.norm3(x)

        return x


def visualize_causal_mask(seq_len: int = 10, save_path: str = "causal_mask.png") -> None:
    """
    Visualize the causal attention mask and save a figure.

    :param seq_len: Sequence length.
    :type seq_len: int
    :param save_path: File path for the saved figure.
    :type save_path: str
    """
    import matplotlib.pyplot as plt

    mask = create_causal_mask(seq_len, torch.device("cpu"))
    mask = mask.squeeze().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="RdYlGn", vmin=0, vmax=1)
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title("Causal Attention Mask\n(Green = Can Attend, Red = Masked)")
    plt.colorbar(label="Attention Allowed")

    for i in range(seq_len + 1):
        plt.axhline(i - 0.5, color="black", linewidth=0.5)
        plt.axvline(i - 0.5, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Causal mask visualization saved to {save_path}")


def test_transformer_layers() -> None:
    """
    Test transformer encoder and decoder layers and print output shapes.
    """
    batch_size, seq_len, d_model = 2, 10, 64
    num_heads, d_ff = 8, 256

    x = torch.randn(batch_size, seq_len, d_model)

    print("Testing Transformer Layers:")
    print(f"Input shape: {x.shape}\n")

    print("1. Encoder Layer (Pre-norm, LayerNorm, MHA):")
    encoder = TransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        norm_position="pre",
        norm_type="layernorm",
        attention_type="mha",
    )
    out_encoder = encoder(x)
    print(f"   Output shape: {out_encoder.shape}")

    print("\n2. Encoder Layer (Pre-norm, RMSNorm, GQA):")
    encoder_gqa = TransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        norm_position="pre",
        norm_type="rmsnorm",
        attention_type="gqa",
        num_kv_heads=4,
    )
    out_encoder_gqa = encoder_gqa(x)
    print(f"   Output shape: {out_encoder_gqa.shape}")

    print("\n3. Decoder Layer (Pre-norm, LayerNorm, MHA, Causal):")
    decoder = TransformerDecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        norm_position="pre",
        norm_type="layernorm",
        attention_type="mha",
    )
    out_decoder = decoder(x)
    print(f"   Output shape: {out_decoder.shape}")

    print("\n4. Decoder Layer (with Cross-Attention):")
    decoder_cross = TransformerDecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        use_cross_attention=True,
    )
    encoder_out = torch.randn(batch_size, 15, d_model)
    out_decoder_cross = decoder_cross(x, encoder_output=encoder_out)
    print(f"   Output shape: {out_decoder_cross.shape}")

    def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    print(f"\nParameter counts:")
    print(f"Encoder (MHA): {count_params(encoder):,}")
    print(f"Encoder (GQA): {count_params(encoder_gqa):,}")
    print(f"Decoder (no cross-attn): {count_params(decoder):,}")
    print(f"Decoder (with cross-attn): {count_params(decoder_cross):,}")
