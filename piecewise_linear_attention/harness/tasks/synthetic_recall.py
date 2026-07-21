"""Synthetic associative-recall task.

This is the task that discriminates content-based routing from mere smoothing.
A marker token is placed at a random position in a sequence of otherwise random
content tokens; the label is the token immediately following the marker. To
answer, a model must locate the marker (content match) and copy its successor
across an arbitrary distance to the classification head — a long-range,
selective read that linear/kernel attention notoriously fails while softmax and
recall-capable approximations solve.

The model is a small bidirectional encoder built from this package's
:class:`MultiHeadAttention`, so every registered attention mechanism can be
plugged in by name and compared under a matched budget.
"""

import torch
import torch.nn as nn

from ...core.registry import build_attention

# Reserved token ids.
PAD_TOKEN = 0
MARKER_TOKEN = 1
# Content tokens occupy ids >= FIRST_CONTENT_TOKEN.
FIRST_CONTENT_TOKEN = 2


def make_recall_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple:
    """Generate one batch of the associative-recall task.

    Parameters
    ----------
    batch_size:
        Number of sequences.
    seq_len:
        Sequence length.
    vocab_size:
        Vocabulary size; content tokens are drawn from
        ``[FIRST_CONTENT_TOKEN, vocab_size)``.
    generator:
        Torch RNG. Passing a seeded generator makes the batch deterministic.

    Returns
    -------
    inputs:
        Long tensor of shape ``(batch_size, seq_len)``.
    labels:
        Long tensor of shape ``(batch_size,)`` — the token after each marker.
    """
    if vocab_size <= FIRST_CONTENT_TOKEN:
        raise ValueError(f"vocab_size must exceed {FIRST_CONTENT_TOKEN}, got {vocab_size}")
    # Content tokens in [FIRST_CONTENT_TOKEN, vocab_size).
    x = torch.randint(FIRST_CONTENT_TOKEN, vocab_size, (batch_size, seq_len), generator=generator)
    # Marker position in [1, seq_len - 1) so a successor always exists and the
    # answer is never at the very start (forces a genuine read).
    pos = torch.randint(1, seq_len - 1, (batch_size,), generator=generator)
    rows = torch.arange(batch_size)
    labels = x[rows, pos + 1].clone()
    x[rows, pos] = MARKER_TOKEN
    return x, labels


class RecallModel(nn.Module):
    """Tiny bidirectional encoder with a pluggable attention mechanism.

    A learned CLS token is prepended; its final representation is classified over
    the vocabulary. Blocks are pre-norm ``attention + MLP`` residual stacks.

    Parameters
    ----------
    attention_type:
        Registered attention name.
    vocab_size, seq_len, hidden_dim, num_heads, num_layers, dropout:
        Model/task sizing.
    attention_kwargs:
        Mechanism-specific options forwarded to ``build_attention``.
    """

    def __init__(
        self,
        attention_type: str,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        attention_kwargs: dict = None,
    ):
        super().__init__()
        attention_kwargs = attention_kwargs or {}
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # +1 position for the prepended CLS token.
        self.pos_embedding = nn.Embedding(seq_len + 1, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            _EncoderBlock(
                attention_type=attention_type,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                attention_kwargs=attention_kwargs,
            )
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Classify each sequence over the vocabulary from its CLS token."""
        batch_size, seq_len = tokens.shape
        h = self.embedding(tokens)
        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)  # (batch, seq_len + 1, hidden)
        positions = torch.arange(seq_len + 1, device=tokens.device)
        h = h + self.pos_embedding(positions)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h)[:, 0])  # logits from CLS position


class _EncoderBlock(nn.Module):
    """Pre-norm attention + MLP residual block."""

    def __init__(
        self,
        attention_type: str,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        attention_kwargs: dict,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        # Non-causal (bidirectional) encoder attention.
        self.attention = build_attention(
            attention_type,
            dim=self.head_dim,
            dropout=dropout,
            causal=False,
            **attention_kwargs,
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return (
            t.view(b, n, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(b * self.num_heads, n, self.head_dim)
        )

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        out, _ = self.attention(q, k, v)
        out = (
            out.view(b, self.num_heads, n, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(b, n, self.hidden_dim)
        )
        return self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attend(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
