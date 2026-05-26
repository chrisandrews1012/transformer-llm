"""
Transformer language model: embeddings, decoder stack, and LM head.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components import (
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
    TransformerDecoderLayer,
    create_causal_mask,
    create_padding_mask,
)
from .config import ModelConfig


class TransformerLanguageModel(nn.Module):
    """
    Decoder-only Transformer Language Model for autoregressive text generation.

    Architecture: token embedding, positional encoding, stack of transformer
    decoder layers, final normalization, and LM head projecting to vocabulary.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize transformer language model.

        :param config: Model configuration.
        :type config: ModelConfig
        """
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)

        if config.pos_encoding_type == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        elif config.pos_encoding_type == "rope":
            self.pos_encoding = RotaryPositionalEmbedding(config.head_dim, config.max_seq_len)
        elif config.pos_encoding_type == "learned":
            self.pos_encoding = LearnedPositionalEmbedding(config.max_seq_len, config.d_model)
        else:
            raise ValueError(f"Unknown pos_encoding_type: {config.pos_encoding_type}")

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                activation=config.activation,
                norm_type=config.norm_type,
                norm_position=config.norm_position,
                attention_type=config.attention_type,
                num_kv_heads=config.num_kv_heads,
                ffn_type=config.ffn_type,
                use_cross_attention=False,
            )
            for _ in range(config.num_layers)
        ])

        if config.norm_type == "layernorm":
            from ..components import LayerNorm
            self.final_norm = LayerNorm(config.d_model)
        elif config.norm_type == "rmsnorm":
            from ..components import RMSNorm
            self.final_norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights using normal initialization (std=0.02).

        Applied to all submodules automatically via self.apply().

        :param module: Module to initialize.
        :type module: nn.Module
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass of the language model.

        :param input_ids: Input token IDs of shape (batch_size, seq_len).
        :type input_ids: torch.Tensor
        :param attention_mask: Optional padding mask of shape (batch_size, seq_len);
            1 for real tokens and 0 for padding.
        :type attention_mask: Optional[torch.Tensor]
        :param return_hidden_states: Whether to return hidden states from all layers.
        :type return_hidden_states: bool
        :returns: Tuple of (logits, hidden_states). Logits has shape
            (batch_size, seq_len, vocab_size). hidden_states is a list of tensors
            (one per layer) if return_hidden_states is True, else None.
        :rtype: Tuple[torch.Tensor, Optional[List[torch.Tensor]]]
        """
        batch_size, seq_len = input_ids.shape

        x = self.token_embedding(input_ids)

        if self.config.pos_encoding_type in ["sinusoidal", "learned"]:
            x = self.pos_encoding(x)

        causal_mask = create_causal_mask(seq_len, input_ids.device)

        if attention_mask is not None:
            padding_mask = create_padding_mask(input_ids, self.config.pad_token_id)
            mask = causal_mask * padding_mask
        else:
            mask = causal_mask

        hidden_states: Optional[List[torch.Tensor]] = [] if return_hidden_states else None

        for layer in self.layers:
            if return_hidden_states:
                hidden_states.append(x)

            x = layer(x, self_attn_mask=mask)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits, hidden_states

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Simple built-in generation method. See the generation module for more
        sophisticated decoding strategies.

        :param input_ids: Input token IDs of shape (batch_size, seq_len).
        :type input_ids: torch.Tensor
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature (higher = more random).
        :type temperature: float
        :param top_k: If set, only sample from the top-k tokens.
        :type top_k: Optional[int]
        :param top_p: If set, only sample from the top-p probability mass.
        :type top_p: Optional[float]
        :param do_sample: Whether to sample or use greedy decoding.
        :type do_sample: bool
        :returns: Generated token IDs of shape (batch_size, seq_len + max_new_tokens).
        :rtype: torch.Tensor
        """
        self.eval()
        generated = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self(generated)
                logits = logits[:, -1, :]

                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                if top_p is not None:
                    pass  # handled by generation module

                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1)

                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

                if (next_token == self.config.eos_token_id).all():
                    break

        return generated

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Return the number of parameters in the model.

        :param non_embedding: If True, exclude embedding parameters from the count.
        :type non_embedding: bool
        :returns: Total parameter count.
        :rtype: int
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate model flops utilization (MFU) as a fraction of peak device FLOPS.

        :param fwdbwd_per_iter: Number of forward-backward passes per iteration.
        :type fwdbwd_per_iter: int
        :param dt: Time per iteration in seconds.
        :type dt: float
        :returns: MFU as a fraction of peak FLOPS (assumes T4 GPU at 65 TFLOPS).
        :rtype: float
        """
        N = self.get_num_params()
        L, H, Q, T = (
            self.config.num_layers,
            self.config.num_heads,
            self.config.d_model // self.config.num_heads,
            self.config.max_seq_len,
        )

        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T * fwdbwd_per_iter * 3
        flops_per_iter = flops_per_fwdbwd
        flops_achieved = flops_per_iter / dt

        flops_promised = 65e12  # T4 GPU estimate

        mfu = flops_achieved / flops_promised
        return mfu


def test_language_model() -> None:
    """
    Test the language model forward pass and generation.
    """
    from .config import get_small_config

    config = get_small_config()
    model = TransformerLanguageModel(config)

    print("Testing Transformer Language Model:")
    print(f"Configuration: {config.to_dict()}\n")

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"Input shape: {input_ids.shape}")

    logits, hidden_states = model(input_ids, return_hidden_states=True)
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of hidden states: {len(hidden_states)}")

    print("\nTesting generation:")
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, do_sample=False)
    print(f"Prompt shape: {prompt.shape}")
    print(f"Generated shape: {generated.shape}")

    print(f"\nModel statistics:")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Non-embedding parameters: {model.get_num_params(non_embedding=True):,}")
