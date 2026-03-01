"""Multi-head attention wrapper for transformer integration.

This module provides a drop-in replacement for standard multi-head attention
that works with StandardAttention, LinearAttention, and PiecewiseAttention mechanisms.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable

from ..core.attention import PiecewiseAttention, StandardAttention, LinearAttention


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

        # Create attention modules for each head
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            if attention_type == "standard":
                attn = StandardAttention(
                    dim=self.attn_dim,
                    dropout=dropout,
                    scale=True,
                    causal=causal
                )
            elif attention_type == "linear":
                attn = LinearAttention(
                    dim=self.attn_dim,
                    dropout=dropout,
                    kernel_type=kernel_type,
                    causal=causal
                )
            elif attention_type == "piecewise":
                attn = PiecewiseAttention(
                    dim=self.attn_dim,
                    dropout=dropout,
                    pseudo_query_fn=pseudo_query_fn,
                    scale=True,
                    causal=causal
                )
            else:
                raise ValueError(f"Unknown attention_type: {attention_type}. Must be 'standard', 'linear', or 'piecewise'")

            self.attention_heads.append(attn)

        # For visualization and analysis
        self.last_attention = None

        if device is not None:
            self.to(device)

    def forward(
        self,
        attending_states: torch.Tensor,  # [batch, query_len, hidden_dim]
        attended_states: torch.Tensor,   # [batch, key_len, hidden_dim]
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
        K = self.k_projection(attended_states)   # [batch, key_len, hidden_dim]
        V = self.v_projection(attended_states)   # [batch, key_len, hidden_dim]

        # Split into heads: [batch, seq, hidden_dim] → [batch, seq, num_heads, attn_dim]
        Q = Q.view(batch_size, query_len, self.num_heads, self.attn_dim)
        K = K.view(batch_size, key_len, self.num_heads, self.attn_dim)
        V = V.view(batch_size, key_len, self.num_heads, self.attn_dim)

        # Apply attention per head
        head_outputs = []
        head_attns = []

        for head_idx in range(self.num_heads):
            # Extract this head's Q, K, V: [batch, seq, attn_dim]
            q_head = Q[:, :, head_idx, :]
            k_head = K[:, :, head_idx, :]
            v_head = V[:, :, head_idx, :]

            # Apply attention
            output_head, attn_weights = self.attention_heads[head_idx](
                q_head, k_head, v_head
            )
            # output_head: [batch, query_len, attn_dim]
            # attn_weights: [batch, query_len, key_len] or None (for piecewise/linear)

            head_outputs.append(output_head)
            head_attns.append(attn_weights)

        # Concatenate heads: [batch, query_len, num_heads, attn_dim]
        output = torch.stack(head_outputs, dim=2)

        # Reshape to [batch, query_len, hidden_dim]
        output = output.reshape(batch_size, query_len, self.hidden_dim)

        # Final projection
        output = self.out_projection(output)

        # Store attention for visualization (average across heads)
        if head_attns[0] is not None:
            self.last_attention = torch.stack(head_attns, dim=1).detach()
            # Shape: [batch, num_heads, query_len, key_len]
        else:
            self.last_attention = None

        return output
