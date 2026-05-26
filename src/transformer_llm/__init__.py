"""
Transformer Language Model

This package contains implementations of transformer-based language models
and all necessary components for training and evaluation.
"""

# Import main classes for convenience
from .model import TransformerLanguageModel, ModelConfig
from .tokenizer import BPETokenizer, CharacterTokenizer
from .training import Trainer
from .generation import TextGenerator

__version__ = "0.1.0"

__all__ = [
    "TransformerLanguageModel",
    "ModelConfig",
    "BPETokenizer",
    "CharacterTokenizer",
    "Trainer",
    "TextGenerator",
]
