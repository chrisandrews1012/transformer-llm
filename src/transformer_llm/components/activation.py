"""
Activation functions: ReLU, GELU, SiLU/Swish, and GLU.
"""

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLU(nn.Module):
    """
    Rectified Linear Unit (ReLU) activation function.

    Formula: ReLU(x) = max(0, x).
    Used in the original Transformer and many CNNs.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ReLU activation.

        :param x: Input tensor of any shape.
        :type x: torch.Tensor
        :returns: Output tensor of the same shape.
        :rtype: torch.Tensor
        """
        return F.relu(x)


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    Approximation: GELU(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    Used in BERT, GPT-2, GPT-3, and many modern transformers.
    """

    def __init__(self, approximate: str = "tanh"):
        """
        Initialize GELU with the chosen approximation.

        :param approximate: Approximation method passed to F.gelu ("none" or "tanh").
        :type approximate: str
        """
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.

        :param x: Input tensor of any shape.
        :type x: torch.Tensor
        :returns: Output tensor of the same shape.
        :rtype: torch.Tensor
        """
        return F.gelu(x, approximate=self.approximate)


class SiLU(nn.Module):
    """
    Sigmoid Linear Unit (SiLU), also known as Swish.

    Formula: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
    Used in LLaMA and many modern LLMs.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SiLU/Swish activation.

        :param x: Input tensor of any shape.
        :type x: torch.Tensor
        :returns: Output tensor of the same shape.
        :rtype: torch.Tensor
        """
        return F.silu(x)


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) activation function.

    Splits the input along the last dimension and uses one half to gate
    the other: GLU(a, b) = a * sigmoid(b).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GLU activation.

        :param x: Input tensor of shape (..., 2*dim). The last dimension is
            split into two halves.
        :type x: torch.Tensor
        :returns: Output tensor of shape (..., dim).
        :rtype: torch.Tensor
        """
        a, b = torch.chunk(x, 2, dim=-1)
        return a * torch.sigmoid(b)


def get_activation(name: str) -> nn.Module:
    """
    Return an activation function module by name.

    :param name: Name of the activation function ("relu", "gelu", "silu", "swish",
        or "glu").
    :type name: str
    :returns: Activation function module.
    :rtype: nn.Module
    :raises ValueError: If the activation name is not recognized.
    """
    activations = {
        "relu": ReLU(),
        "gelu": GELU(),
        "silu": SiLU(),
        "swish": SiLU(),
        "glu": GLU(),
    }

    if name.lower() not in activations:
        raise ValueError(
            f"Unknown activation: {name}. Choose from {list(activations.keys())}"
        )

    return activations[name.lower()]


def visualize_activations(
    x_range: tuple = (-5, 5),
    num_points: int = 1000,
    save_path: str = "activation_functions.png",
) -> None:
    """
    Plot ReLU, GELU, and SiLU over a range of input values and save the figure.

    :param x_range: (min, max) range of x values to plot.
    :type x_range: tuple
    :param num_points: Number of sample points.
    :type num_points: int
    :param save_path: File path for the saved figure.
    :type save_path: str
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x = torch.linspace(x_range[0], x_range[1], num_points)

    activations = {
        "ReLU": ReLU()(x),
        "GELU": GELU()(x),
        "SiLU": SiLU()(x),
    }

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    for name, y in activations.items():
        plt.plot(x.numpy(), y.numpy(), label=name, linewidth=2)
    plt.xlabel("Input (x)")
    plt.ylabel("Output")
    plt.title("Activation Functions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.axvline(x=0, color="k", linewidth=0.5)

    plt.subplot(1, 2, 2)
    x_grad = x.clone().requires_grad_(True)
    for name in activations.keys():
        if name == "ReLU":
            y = ReLU()(x_grad)
        elif name == "GELU":
            y = GELU()(x_grad)
        elif name == "SiLU":
            y = SiLU()(x_grad)

        y.sum().backward()
        grad = x_grad.grad.clone()
        x_grad.grad.zero_()

        plt.plot(x.numpy(), grad.numpy(), label=f"{name}'", linewidth=2)

    plt.xlabel("Input (x)")
    plt.ylabel("Gradient")
    plt.title("Activation Function Derivatives")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.axvline(x=0, color="k", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Activation functions visualization saved to {save_path}")


def compare_activation_properties() -> None:
    """
    Print a comparison table of key properties for ReLU, GELU, and SiLU.
    """
    properties = {
        "ReLU": {
            "Smooth": False,
            "Bounded Below": True,
            "Bounded Above": False,
            "Monotonic": True,
            "Computational Cost": "Very Low",
            "Gradient Flow": "Good (but can die)",
            "Common Use": "CNNs, older transformers",
        },
        "GELU": {
            "Smooth": True,
            "Bounded Below": False,
            "Bounded Above": False,
            "Monotonic": False,
            "Computational Cost": "Medium",
            "Gradient Flow": "Excellent",
            "Common Use": "BERT, GPT-2/3, modern transformers",
        },
        "SiLU": {
            "Smooth": True,
            "Bounded Below": True,
            "Bounded Above": False,
            "Monotonic": False,
            "Computational Cost": "Low",
            "Gradient Flow": "Excellent",
            "Common Use": "LLaMA, modern LLMs",
        },
    }

    print("\nActivation Function Properties Comparison:")
    print("=" * 80)

    props = list(next(iter(properties.values())).keys())
    print(f"{'Property':<25} {'ReLU':<15} {'GELU':<15} {'SiLU':<15}")
    print("-" * 80)

    for prop in props:
        values = [str(properties[act][prop]) for act in ["ReLU", "GELU", "SiLU"]]
        print(f"{prop:<25} {values[0]:<15} {values[1]:<15} {values[2]:<15}")

    print("=" * 80)


def test_activations() -> None:
    """
    Test activation functions with sample inputs and print results.
    """
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    print("Testing Activation Functions:")
    print(f"Input: {x.tolist()}\n")

    relu = ReLU()
    print(f"ReLU output: {relu(x).tolist()}")

    gelu = GELU()
    print(f"GELU output: {gelu(x).tolist()}")

    silu = SiLU()
    print(f"SiLU output: {silu(x).tolist()}")

    glu = GLU()
    x_glu = torch.randn(2, 10)
    print(f"\nGLU input shape: {x_glu.shape}")
    print(f"GLU output shape: {glu(x_glu).shape}")
