# Architecture Notes

## Attention Masking

Masking must be applied to attention logits before the softmax, not after. Softmax converts raw scores into a probability distribution that sums to 1. If you zero out positions after softmax, that invariant breaks: the remaining values no longer sum to 1, and any attempt to renormalize just repeats what softmax was meant to do cleanly. Masking before softmax sidesteps this entirely: setting future positions to a large negative value causes their post-exponentiation contribution to approach zero, so softmax naturally assigns them near-zero probability while correctly scaling the rest.

## Grouped Query Attention (GQA)

Standard multi-head attention (MHA) gives every head its own Q, K, and V projections. GQA keeps per-head queries but shares keys and values across groups of heads. At the extreme (one group) this collapses to multi-query attention (MQA). The practical benefit is a smaller KV cache, which matters most for long sequences or memory-constrained inference. The tradeoff is reduced representational flexibility; in practice the quality difference is small at moderate scale.

## Positional Encoding: Sinusoidal vs RoPE

Sinusoidal encoding adds fixed sine/cosine vectors directly to token embeddings before the first layer, baking position into the representation from the start. RoPE applies a position-dependent rotation to query and key vectors inside each attention layer, encoding relative position implicitly through the dot product. Because RoPE operates on relative rather than absolute positions, it tends to generalize better to sequence lengths not seen during training.

## Evaluation Metrics and Tokenization

Token-level perplexity (`exp(loss)`) is only directly comparable across models that use the same tokenizer. A larger vocabulary produces fewer, longer tokens per sequence, changing what each prediction step covers. A model with a large-vocab tokenizer may show lower perplexity simply because each token carries more information, not because it is a better model. When comparing across tokenization schemes, character-level accuracy or bytes-per-character are more neutral metrics.

## Language Model Training: Input/Target Shifting

Training a causal language model requires shifting input and target by one position. The input is the sequence with the last token removed; the target is the same sequence with the first token removed. This forces the model to predict the next token at every position using only preceding context. Without the shift, the model would see the answer token it is supposed to predict, learn to copy rather than predict, and produce a deceptively low training loss while being unable to generate.

## Evaluating Generation Quality

A single generation sample is weak evidence. Generation is stochastic, and cherry-picking one fluent output conceals the average behavior. Quantitative metrics like validation loss and token accuracy across the full validation set give a more reliable picture because they aggregate over many examples. Generation samples are useful as a sanity check and for qualitative illustration, but conclusions about model quality should rest on the aggregate metrics.
