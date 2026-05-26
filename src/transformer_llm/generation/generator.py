"""
Text generation: greedy decoding, temperature sampling, top-k/top-p filtering, and beam search.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGenerator:
    """
    Text generator with multiple sampling strategies.

    Provides greedy decoding, temperature sampling, top-k filtering,
    top-p (nucleus) filtering, and combinations thereof.
    """

    def __init__(self, model: nn.Module, tokenizer, device: torch.device):
        """
        Initialize text generator.

        :param model: Trained language model.
        :type model: nn.Module
        :param tokenizer: Tokenizer for encoding/decoding text.
        :param device: Device to run generation on.
        :type device: torch.device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text from a prompt.

        :param prompt: Input text prompt.
        :type prompt: str
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature (higher = more random).
        :type temperature: float
        :param top_k: If set, only sample from the top-k tokens.
        :type top_k: Optional[int]
        :param top_p: If set, only sample from tokens within top-p probability mass.
        :type top_p: Optional[float]
        :param do_sample: Whether to sample or use greedy decoding.
        :type do_sample: bool
        :param num_return_sequences: Number of sequences to generate.
        :type num_return_sequences: int
        :returns: List of generated text strings.
        :rtype: List[str]
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        input_ids = torch.tensor([input_ids], device=self.device)

        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)

        generated_ids = self._generate_tokens(
            input_ids, max_new_tokens, temperature, top_k, top_p, do_sample
        )

        generated_texts = []
        for ids in generated_ids:
            text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        :param input_ids: Input token IDs of shape (batch_size, seq_len).
        :type input_ids: torch.Tensor
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature.
        :type temperature: float
        :param top_k: Top-k filtering value.
        :type top_k: Optional[int]
        :param top_p: Top-p filtering threshold.
        :type top_p: Optional[float]
        :param do_sample: Whether to sample from the distribution.
        :type do_sample: bool
        :returns: Generated token IDs of shape (batch_size, seq_len + max_new_tokens).
        :rtype: torch.Tensor
        """
        generated = input_ids

        for _ in range(max_new_tokens):
            logits, _ = self.model(generated)
            logits = logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                logits = self._top_k_filtering(logits, top_k)

            if top_p is not None:
                logits = self._top_p_filtering(logits, top_p)

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if (next_token == self.tokenizer.eos_token_id).all():
                break

        return generated

    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Filter logits to keep only top-k tokens.

        :param logits: Logits of shape (batch_size, vocab_size).
        :type logits: torch.Tensor
        :param top_k: Number of top tokens to keep.
        :type top_k: int
        :returns: Filtered logits with non-top-k tokens set to -inf.
        :rtype: torch.Tensor
        """
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)

        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_values)

        return filtered_logits

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Filter logits using nucleus (top-p) sampling.

        Keeps only tokens with cumulative probability <= top_p.

        :param logits: Logits of shape (batch_size, vocab_size).
        :type logits: torch.Tensor
        :param top_p: Cumulative probability threshold.
        :type top_p: float
        :returns: Filtered logits.
        :rtype: torch.Tensor
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        filtered_logits = logits.clone()
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered_logits[indices_to_remove] = float("-inf")

        return filtered_logits

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        **kwargs,
    ) -> List[str]:
        """
        Generate text for a batch of prompts.

        :param prompts: List of input prompts.
        :type prompts: List[str]
        :param max_new_tokens: Maximum tokens to generate per prompt.
        :type max_new_tokens: int
        :returns: List of generated texts (one per prompt).
        :rtype: List[str]
        """
        results = []
        for prompt in prompts:
            generated = self.generate(prompt, max_new_tokens, **kwargs)
            results.extend(generated)

        return results


class BeamSearchGenerator:
    """
    Beam search text generator (optional extension).

    Maintains multiple hypotheses and selects the most likely sequence.
    More thorough than greedy decoding at the cost of higher compute.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device,
        beam_width: int = 5,
    ):
        """
        Initialize beam search generator.

        :param model: Trained language model.
        :type model: nn.Module
        :param tokenizer: Tokenizer for encoding/decoding text.
        :param device: Device to run generation on.
        :type device: torch.device
        :param beam_width: Number of beams to maintain.
        :type beam_width: int
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.beam_width = beam_width
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        length_penalty: float = 1.0,
    ) -> str:
        """
        Generate text using beam search.

        :param prompt: Input prompt.
        :type prompt: str
        :param max_new_tokens: Maximum tokens to generate.
        :type max_new_tokens: int
        :param length_penalty: Length penalty applied to beam scores (values > 1.0
            favor longer sequences).
        :type length_penalty: float
        :returns: Generated text string.
        :rtype: str
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=self.device)

        beams: List[Tuple[torch.Tensor, float]] = [(input_ids, 0.0)]

        for _ in range(max_new_tokens):
            candidates = []

            for seq, score in beams:
                logits, _ = self.model(seq)
                logits = logits[:, -1, :]

                log_probs = F.log_softmax(logits, dim=-1)

                top_log_probs, top_indices = torch.topk(log_probs, self.beam_width, dim=-1)

                for i in range(self.beam_width):
                    next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                    next_seq = torch.cat([seq, next_token], dim=-1)
                    next_score = score + top_log_probs[0, i].item()

                    length_normalized_score = next_score / (next_seq.size(1) ** length_penalty)

                    candidates.append((next_seq, next_score, length_normalized_score))

            candidates.sort(key=lambda x: x[2], reverse=True)
            beams = [(seq, score) for seq, score, _ in candidates[: self.beam_width]]

            if all(seq[0, -1].item() == self.tokenizer.eos_token_id for seq, _ in beams):
                break

        best_seq, _ = beams[0]
        generated_text = self.tokenizer.decode(best_seq[0].tolist(), skip_special_tokens=True)

        return generated_text


def compare_sampling_strategies(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
) -> None:
    """
    Compare different sampling strategies on the same prompt.

    :param model: Trained language model.
    :type model: nn.Module
    :param tokenizer: Tokenizer for encoding/decoding text.
    :param prompt: Input text prompt.
    :type prompt: str
    :param device: Device to run generation on.
    :type device: torch.device
    :param max_new_tokens: Maximum tokens to generate.
    :type max_new_tokens: int
    """
    generator = TextGenerator(model, tokenizer, device)

    strategies = {
        "Greedy": {"do_sample": False},
        "Temperature 0.7": {"temperature": 0.7, "do_sample": True},
        "Temperature 1.5": {"temperature": 1.5, "do_sample": True},
        "Top-k (k=50)": {"top_k": 50, "do_sample": True},
        "Top-p (p=0.9)": {"top_p": 0.9, "do_sample": True},
        "Top-k + Top-p": {"top_k": 50, "top_p": 0.9, "do_sample": True},
    }

    print(f"Prompt: {prompt}\n")
    print("=" * 80)

    for name, kwargs in strategies.items():
        generated = generator.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        print(f"\n{name}:")
        print(f"{generated[0]}")
        print("-" * 80)


def test_generator() -> None:
    """
    Test text generator with a small dummy model and tokenizer.
    """
    from ..model.config import get_small_config
    from ..model.language_model import TransformerLanguageModel
    from ..tokenizer.base import CharacterTokenizer

    texts = ["Hello world, this is a test."]
    tokenizer = CharacterTokenizer()
    tokenizer.train(texts)

    config = get_small_config()
    config.vocab_size = tokenizer.vocab_size
    model = TransformerLanguageModel(config)

    device = torch.device("cpu")
    model = model.to(device)

    generator = TextGenerator(model, tokenizer, device)

    print("Testing Text Generator:")
    prompt = "Hello"
    print(f"Prompt: {prompt}\n")

    print("1. Greedy decoding:")
    generated = generator.generate(prompt, max_new_tokens=10, do_sample=False)
    print(f"   {generated[0]}\n")

    print("2. Temperature sampling:")
    generated = generator.generate(prompt, max_new_tokens=10, temperature=0.8, do_sample=True)
    print(f"   {generated[0]}\n")

    print("3. Top-k sampling:")
    generated = generator.generate(prompt, max_new_tokens=10, top_k=5, do_sample=True)
    print(f"   {generated[0]}\n")

    print("Generator test complete!")
