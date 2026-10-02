"""A compact, decoder-only Transformer language model."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 64
    embedding_size: int = 64
    number_of_heads: int = 4
    number_of_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.embedding_size % self.number_of_heads != 0:
            raise ValueError("embedding_size must be divisible by number_of_heads")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.number_of_heads = config.number_of_heads
        self.head_size = config.embedding_size // config.number_of_heads
        self.qkv = nn.Linear(config.embedding_size, 3 * config.embedding_size)
        self.projection = nn.Linear(config.embedding_size, config.embedding_size)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_size = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch_size, sequence_length, self.number_of_heads, self.head_size)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.head_size**0.5)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)
        return self.residual_dropout(self.projection(attended))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.embedding_size, 4 * config.embedding_size),
            nn.GELU(),
            nn.Linear(4 * config.embedding_size, config.embedding_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_size)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_size)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class Feneco(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_size)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.number_of_layers)))
        self.final_norm = nn.LayerNorm(config.embedding_size)
        self.language_model_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, sequence_length = token_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("input is longer than the configured context_length")
        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        logits = self.language_model_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, token_ids: torch.Tensor, number_of_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self.eval()
        for _ in range(number_of_tokens):
            context = token_ids[:, -self.config.context_length :]
            logits, _ = self(context)
            probabilities = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


def config_to_dict(config: ModelConfig) -> dict[str, int | float]:
    """Return a serialization-friendly model configuration."""
    return asdict(config)
