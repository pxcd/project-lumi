
"""Lumi v0: decoder-only Transformer initialized from scratch."""

from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class LumiConfig:
    vocab_size: int = 8192
    context_length: int = 512
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048

    def __post_init__(self):
        if self.vocab_size < 256:
            raise ValueError("Vocabulary must contain all 256 bytes")
        if self.context_length < 1 or self.n_layers < 1:
            raise ValueError("Context length and layers must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff < 1:
            raise ValueError("d_ff must be positive")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LumiConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = nn.RoPE(self.head_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, width = x.shape

        def split_heads(tensor):
            return tensor.reshape(
                batch, length, self.n_heads, self.head_dim
            ).transpose(0, 2, 1, 3)

        q = self.rope(split_heads(self.q_proj(x)))
        k = self.rope(split_heads(self.k_proj(x)))
        v = split_heads(self.v_proj(x))

        attended = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            mask="causal",
        )

        attended = attended.transpose(0, 2, 1, 3)
        attended = attended.reshape(batch, length, width)

        return self.o_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, config: LumiConfig):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.d_ff)
        self.activation = nn.GELU()
        self.down = nn.Linear(config.d_ff, config.d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(self.activation(self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: LumiConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LumiModel(nn.Module):
    def __init__(self, config: LumiConfig):
        super().__init__()

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.blocks = [
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ]

        self.final_norm = nn.LayerNorm(config.d_model)

    def __call__(self, token_ids: mx.array) -> mx.array:
        if token_ids.ndim != 2:
            raise ValueError("Expected token IDs shaped [batch, sequence]")

        if token_ids.shape[1] > self.config.context_length:
            raise ValueError("Sequence exceeds configured context length")

        x = self.embedding(token_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # Tied output projection: reuse the input embedding weights.
        logits = x @ self.embedding.weight.T

        return logits
