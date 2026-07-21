"""Multi-head attention wrapper for transformer integration.

This module provides a drop-in replacement for standard multi-head attention
that works with StandardAttention, LinearAttention, and PiecewiseAttention mechanisms.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from ..core.registry import build_attention


class MultiHeadAttention(nn.Module):
    """Multi-head attention with pluggable attention mechanism.

    Supports multiple attention implementations for comparative analysis:
    - StandardAttention: Softmax attention baseline
    - LinearAttention: Kernel-based linear attention
    - PiecewiseAttention: First-order Taylor approximation

    Interface:
        forward(attending_states, attended_states, attn_mask=None)

    Args:
        hidden_dim: Model dimension (e.g., 256)
        num_heads: Number of attention heads (e.g., 4)
        attention_type: "standard", "linear", or "piecewise"
        dropout: Dropout probability
        pseudo_query_fn: For piecewise attention, how to select representative query
        kernel_type: For linear attention, the kernel type ("elu" or "relu")
        device: Device to place module on
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attention_type: str = "standard",
        dropout: float = 0.0,
        pseudo_query_fn: Optional[Callable] = None,
        kernel_type: str = "elu",
        causal: bool = False,
        eps: float = 1e-6,
        causal_pseudo_query: str = "mean",
        device: Optional[torch.device] = None,
    ):
        """Initialize multi-head attention.

        Args:
            hidden_dim: Model dimension (e.g., 256)
            num_heads: Number of attention heads (e.g., 4)
            attention_type: "standard", "linear", or "piecewise"
            dropout: Dropout probability
            pseudo_query_fn: For piecewise attention, how to select representative query
            kernel_type: For linear attention, kernel type ("elu" or "relu")
            causal: If True, apply causal masking (query i can only attend to keys 0...i)
            eps: For linear attention, numerical-stability constant
            causal_pseudo_query: For causal piecewise attention, anchor strategy
                ("mean" or "first")
            device: Device to place module on
        """
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.attn_dim = hidden_dim // num_heads  # e.g., 256 // 4 = 64
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type
        self.causal = causal

        # Q, K, V projections (shared across heads)
        self.q_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)

        # Single attention module shared across heads. The attention
        # implementations are parameter-free (pure functions of their config),
        # so all heads are computed in one batched call over the flattened
        # (batch * num_heads) dimension rather than a Python loop. Construction
        # is routed through the registry so every registered mechanism is usable
        # here without editing this class.
        self.attention = build_attention(
            attention_type,
            dim=self.attn_dim,
            dropout=dropout,
            causal=causal,
            scale=True,
            kernel_type=kernel_type,
            eps=eps,
            pseudo_query_fn=pseudo_query_fn,
            causal_pseudo_query=causal_pseudo_query,
        )

        # For visualization and analysis
        self.last_attention = None

        if device is not None:
            self.to(device)

    def forward(
        self,
        attending_states: torch.Tensor,  # [batch, query_len, hidden_dim]
        attended_states: torch.Tensor,  # [batch, key_len, hidden_dim]
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            attending_states: Query states - what attends [batch, query_len, hidden_dim]
            attended_states: Key/value states - what is attended to [batch, key_len, hidden_dim]

        Returns:
            output: Attention output [batch, query_len, hidden_dim]
        """
        batch_size, query_len, _ = attending_states.shape
        key_len = attended_states.shape[1]

        # Project to Q, K, V
        Q = self.q_projection(attending_states)  # [batch, query_len, hidden_dim]
        K = self.k_projection(attended_states)  # [batch, key_len, hidden_dim]
        V = self.v_projection(attended_states)  # [batch, key_len, hidden_dim]

        # Split into heads and flatten heads into the batch dimension so a
        # single attention call covers all heads:
        #   [batch, seq, hidden_dim]
        #     -> [batch, seq, num_heads, attn_dim]
        #     -> [batch, num_heads, seq, attn_dim]
        #     -> [batch * num_heads, seq, attn_dim]
        def split_heads(t: torch.Tensor, seq: int) -> torch.Tensor:
            return (
                t.view(batch_size, seq, self.num_heads, self.attn_dim)
                .permute(0, 2, 1, 3)
                .reshape(batch_size * self.num_heads, seq, self.attn_dim)
            )

        Qh = split_heads(Q, query_len)  # [batch*heads, query_len, attn_dim]
        Kh = split_heads(K, key_len)  # [batch*heads, key_len, attn_dim]
        Vh = split_heads(V, key_len)  # [batch*heads, key_len, attn_dim]

        # One batched attention call over all heads.
        # extra is attention weights [batch*heads, query_len, key_len] for
        # standard attention, or an implementation-specific tensor otherwise.
        attn_out, extra = self.attention(Qh, Kh, Vh)

        # Merge heads back: [batch*heads, query_len, attn_dim]
        #   -> [batch, num_heads, query_len, attn_dim]
        #   -> [batch, query_len, num_heads, attn_dim]
        #   -> [batch, query_len, hidden_dim]
        output = (
            attn_out.view(batch_size, self.num_heads, query_len, self.attn_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.hidden_dim)
        )

        # Final projection
        output = self.out_projection(output)

        # Store attention for visualization only when `extra` is a genuine
        # per-head attention-weight matrix [batch*heads, query_len, key_len]
        # (standard attention). Linear/piecewise return other tensors, so we
        # gate on shape rather than on `extra is not None`.
        if (
            isinstance(extra, torch.Tensor)
            and extra.dim() == 3
            and extra.shape == (batch_size * self.num_heads, query_len, key_len)
        ):
            self.last_attention = extra.view(
                batch_size, self.num_heads, query_len, key_len
            ).detach()
        else:
            self.last_attention = None

        return output
