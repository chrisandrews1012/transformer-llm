"""
Normalization layers: LayerNorm and RMSNorm, with pre-norm and post-norm residual paths.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    Normalizes inputs across the feature dimension per sample using learned
    scale (gamma) and shift (beta) parameters.
    Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        """
        Initialize layer normalization.

        :param normalized_shape: Number of features to normalize (typically d_model).
        :type normalized_shape: int
        :param eps: Small constant for numerical stability.
        :type eps: float
        :param elementwise_affine: Whether to learn gamma and beta parameters.
        :type elementwise_affine: bool
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        :param x: Input tensor of shape (batch_size, seq_len, d_model) or
            (batch_size, d_model).
        :type x: torch.Tensor
        :returns: Normalized tensor of the same shape as input.
        :rtype: torch.Tensor
        """
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm

    def extra_repr(self) -> str:
        """
        Return string representation for debugging.

        :returns: Human-readable parameter summary.
        :rtype: str
        """
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (optional extension).

    Simpler variant of LayerNorm that normalizes by RMS only (no mean subtraction),
    with a single learned scale parameter. Used in LLaMA and other modern LLMs.
    Formula: y = x / RMS(x) * gamma, where RMS(x) = sqrt(mean(x^2) + eps).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        """
        Initialize RMS normalization.

        :param normalized_shape: Number of features to normalize (typically d_model).
        :type normalized_shape: int
        :param eps: Small constant for numerical stability.
        :type eps: float
        """
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        :param x: Input tensor of shape (batch_size, seq_len, d_model) or
            (batch_size, d_model).
        :type x: torch.Tensor
        :returns: Normalized tensor of the same shape as input.
        :rtype: torch.Tensor
        """
        rms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x * rms
        return x_norm * self.weight

    def extra_repr(self) -> str:
        """
        Return string representation for debugging.

        :returns: Human-readable parameter summary.
        :rtype: str
        """
        return f"{self.normalized_shape}, eps={self.eps}"


class PreNorm(nn.Module):
    """
    Pre-normalization wrapper.

    Applies normalization before the sub-layer, then adds a residual connection.
    Structure: x = x + sublayer(norm(x)). Used in modern transformers (GPT, LLaMA).
    """

    def __init__(self, dim: int, fn: nn.Module, norm_type: str = "layernorm"):
        """
        Initialize pre-normalization wrapper.

        :param dim: Model dimension.
        :type dim: int
        :param fn: Sub-layer module (attention or FFN).
        :type fn: nn.Module
        :param norm_type: Type of normalization ("layernorm" or "rmsnorm").
        :type norm_type: str
        :raises ValueError: If norm_type is not recognized.
        """
        super().__init__()
        if norm_type == "layernorm":
            self.norm = LayerNorm(dim)
        elif norm_type == "rmsnorm":
            self.norm = RMSNorm(dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply pre-normalization with residual connection.

        :param x: Input tensor.
        :type x: torch.Tensor
        :returns: Output tensor after norm -> sub-layer -> residual.
        :rtype: torch.Tensor
        """
        return x + self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    """
    Post-normalization wrapper.

    Applies normalization after the sub-layer and residual connection.
    Structure: x = norm(x + sublayer(x)). Used in the original Transformer paper.
    """

    def __init__(self, dim: int, fn: nn.Module, norm_type: str = "layernorm"):
        """
        Initialize post-normalization wrapper.

        :param dim: Model dimension.
        :type dim: int
        :param fn: Sub-layer module (attention or FFN).
        :type fn: nn.Module
        :param norm_type: Type of normalization ("layernorm" or "rmsnorm").
        :type norm_type: str
        :raises ValueError: If norm_type is not recognized.
        """
        super().__init__()
        if norm_type == "layernorm":
            self.norm = LayerNorm(dim)
        elif norm_type == "rmsnorm":
            self.norm = RMSNorm(dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply post-normalization with residual connection.

        :param x: Input tensor.
        :type x: torch.Tensor
        :returns: Output tensor after sub-layer -> residual -> norm.
        :rtype: torch.Tensor
        """
        return self.norm(x + self.fn(x, **kwargs))


def compare_normalizations(
    batch_size: int = 2,
    seq_len: int = 10,
    d_model: int = 64,
) -> dict:
    """
    Compare LayerNorm and RMSNorm behavior and print statistics.

    :param batch_size: Batch size for the test input.
    :type batch_size: int
    :param seq_len: Sequence length for the test input.
    :type seq_len: int
    :param d_model: Model dimension.
    :type d_model: int
    :returns: Dictionary with mean and std statistics for input, LayerNorm output,
        and RMSNorm output.
    :rtype: dict
    """
    x = torch.randn(batch_size, seq_len, d_model)

    ln = LayerNorm(d_model)
    x_ln = ln(x)

    rms = RMSNorm(d_model)
    x_rms = rms(x)

    stats = {
        "input_mean": x.mean().item(),
        "input_std": x.std().item(),
        "layernorm_mean": x_ln.mean().item(),
        "layernorm_std": x_ln.std().item(),
        "rmsnorm_mean": x_rms.mean().item(),
        "rmsnorm_std": x_rms.std().item(),
    }

    print("Normalization Comparison:")
    print(f"Input - Mean: {stats['input_mean']:.4f}, Std: {stats['input_std']:.4f}")
    print(f"LayerNorm - Mean: {stats['layernorm_mean']:.4f}, Std: {stats['layernorm_std']:.4f}")
    print(f"RMSNorm - Mean: {stats['rmsnorm_mean']:.4f}, Std: {stats['rmsnorm_std']:.4f}")
    print("\nNote: LayerNorm centers the output (mean approx 0), while RMSNorm does not.")

    return stats
