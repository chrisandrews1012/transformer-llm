"""
Feed-forward networks: standard FFN, SwiGLU/GeGLU gated variants, and Mixture of Experts (MoE).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import get_activation


class PositionWiseFeedForward(nn.Module):
    """
    Standard Position-wise Feed-Forward Network.

    Structure: Linear(d_model -> d_ff) -> Activation -> Dropout -> Linear(d_ff -> d_model).
    Formula: FFN(x) = activation(x W1 + b1) W2 + b2.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """
        Initialize the position-wise feed-forward network.

        :param d_model: Dimension of the model (input and output dimension).
        :type d_model: int
        :param d_ff: Dimension of the hidden feed-forward layer.
        :type d_ff: int
        :param dropout: Dropout probability.
        :type dropout: float
        :param activation: Activation function name ("relu", "gelu", "silu").
        :type activation: str
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise feed-forward network.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :type x: torch.Tensor
        :returns: Output tensor of shape (batch_size, seq_len, d_model).
        :rtype: torch.Tensor
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GLUFeedForward(nn.Module):
    """
    Gated Linear Unit (GLU) Feed-Forward Network.

    Variants: GLU (sigmoid gate), SwiGLU (SiLU gate, used in LLaMA),
    GeGLU (GELU gate). Formula: GLU(x) = gate(x W1) * activation(x W2).
    Structure: Linear(d_model -> 2*d_ff) -> Split -> Multiply -> Linear(d_ff -> d_model).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        """
        Initialize the GLU feed-forward network.

        :param d_model: Dimension of the model.
        :type d_model: int
        :param d_ff: Dimension of the feed-forward layer (each split half).
        :type d_ff: int
        :param dropout: Dropout probability.
        :type dropout: float
        :param activation: Activation function for gating ("sigmoid", "silu", "gelu").
        :type activation: str
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, 2 * d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GLU feed-forward network.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :type x: torch.Tensor
        :returns: Output tensor of shape (batch_size, seq_len, d_model).
        :rtype: torch.Tensor
        """
        x = self.linear1(x)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x = x1 * self.activation(x2)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MixtureOfExperts(nn.Module):
    """
    Simple Mixture of Experts (MoE) Feed-Forward Network (optional extension).

    Routes each token to the top-k expert FFNs and combines their outputs
    weighted by the gating probabilities.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """
        Initialize the Mixture of Experts layer.

        :param d_model: Dimension of the model.
        :type d_model: int
        :param d_ff: Dimension of each expert's feed-forward layer.
        :type d_ff: int
        :param num_experts: Number of expert networks.
        :type num_experts: int
        :param top_k: Number of experts to route each token to.
        :type top_k: int
        :param dropout: Dropout probability.
        :type dropout: float
        :param activation: Activation function for experts.
        :type activation: str
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model

        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [PositionWiseFeedForward(d_model, d_ff, dropout, activation) for _ in range(num_experts)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply mixture of experts.

        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :type x: torch.Tensor
        :returns: Output tensor of shape (batch_size, seq_len, d_model).
        :rtype: torch.Tensor
        """
        batch_size, seq_len, d_model = x.shape

        gate_logits = self.gate(x)
        gate_scores, expert_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        gate_probs = F.softmax(gate_scores, dim=-1)

        output = torch.zeros_like(x)

        for k in range(self.top_k):
            for i in range(batch_size):
                for j in range(seq_len):
                    expert_idx = expert_indices[i, j, k]
                    expert = self.experts[expert_idx]
                    token_input = x[i:i+1, j:j+1, :]
                    expert_output = expert(token_input)
                    output[i, j, :] += gate_probs[i, j, k] * expert_output.squeeze(0).squeeze(0)

        return output

    def load_balancing_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage uniform expert usage.

        :param gate_logits: Gate logits of shape (batch_size, seq_len, num_experts).
        :type gate_logits: torch.Tensor
        :returns: Scalar load balancing loss (coefficient of variation of expert usage).
        :rtype: torch.Tensor
        """
        expert_usage = torch.softmax(gate_logits, dim=-1).mean(dim=[0, 1])
        loss = torch.std(expert_usage) / torch.mean(expert_usage)
        return loss


def create_ffn(
    ffn_type: str,
    d_model: int,
    d_ff: int,
    dropout: float = 0.1,
    activation: str = "relu",
    **kwargs,
) -> nn.Module:
    """
    Create a feed-forward network of the specified type.

    :param ffn_type: FFN variant ("standard", "glu", "swiglu", "geglu", "moe").
    :type ffn_type: str
    :param d_model: Model dimension.
    :type d_model: int
    :param d_ff: Feed-forward hidden dimension.
    :type d_ff: int
    :param dropout: Dropout probability.
    :type dropout: float
    :param activation: Activation function name.
    :type activation: str
    :returns: Feed-forward network module.
    :rtype: nn.Module
    :raises ValueError: If ffn_type is not recognized.
    """
    if ffn_type == "standard":
        return PositionWiseFeedForward(d_model, d_ff, dropout, activation)
    elif ffn_type == "glu":
        return GLUFeedForward(d_model, d_ff, dropout, activation="sigmoid")
    elif ffn_type == "swiglu":
        return GLUFeedForward(d_model, d_ff, dropout, activation="silu")
    elif ffn_type == "geglu":
        return GLUFeedForward(d_model, d_ff, dropout, activation="gelu")
    elif ffn_type == "moe":
        return MixtureOfExperts(d_model, d_ff, dropout=dropout, activation=activation, **kwargs)
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type}")


def test_ffn() -> None:
    """
    Test different FFN implementations and print output shapes.
    """
    batch_size, seq_len, d_model, d_ff = 2, 10, 64, 256

    x = torch.randn(batch_size, seq_len, d_model)

    print("Testing Feed-Forward Networks:")
    print(f"Input shape: {x.shape}\n")

    ffn_standard = PositionWiseFeedForward(d_model, d_ff)
    out_standard = ffn_standard(x)
    print(f"Standard FFN output shape: {out_standard.shape}")

    ffn_glu = GLUFeedForward(d_model, d_ff)
    out_glu = ffn_glu(x)
    print(f"GLU FFN output shape: {out_glu.shape}")

    ffn_moe = MixtureOfExperts(d_model, d_ff, num_experts=4, top_k=2)
    out_moe = ffn_moe(x)
    print(f"MoE FFN output shape: {out_moe.shape}")

    def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    print(f"\nParameter counts:")
    print(f"Standard FFN: {count_params(ffn_standard):,}")
    print(f"GLU FFN: {count_params(ffn_glu):,}")
    print(f"MoE FFN: {count_params(ffn_moe):,}")
