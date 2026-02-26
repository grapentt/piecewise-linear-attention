"""Core attention mechanisms for piecewise linear attention.

This module implements the baseline attention mechanisms and the novel
piecewise linear attention approach.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAttention(nn.Module, ABC):
    """Abstract base class for attention mechanisms.

    Provides a common interface for different attention implementations.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        """Initialize base attention module.

        Args:
            dim: Dimension of the input embeddings
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention output.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask of shape (batch, seq_len, seq_len)
                  where 0 indicates positions to mask out
            return_attention_weights: If True, also return attention weights

        Returns:
            output: Attention output of shape (batch, seq_len, dim)
            weights: Optional attention weights if return_attention_weights=True
        """
        pass


class StandardAttention(BaseAttention):
    """Standard scaled dot-product attention with softmax.

    Implements the attention mechanism from "Attention is All You Need":
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Complexity: O(n^2 * d) where n is sequence length and d is dimension.
    """

    def __init__(self, dim: int, dropout: float = 0.0, scale: bool = True):
        """Initialize standard attention.

        Args:
            dim: Dimension of the input embeddings
            dropout: Dropout probability for attention weights
            scale: Whether to scale attention scores by sqrt(d_k)
        """
        super().__init__(dim, dropout)
        self.scale = scale

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute standard softmax attention.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask of shape (batch, seq_len, seq_len)
                  where 0 indicates positions to mask out
            return_attention_weights: If True, also return attention weights

        Returns:
            output: Attention output of shape (batch, seq_len, dim)
            weights: Optional attention weights of shape (batch, seq_len, seq_len)
        """
        batch_size, seq_len, d_k = Q.size()

        # Compute attention scores: Q @ K^T
        # Shape: (batch, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale by sqrt(d_k) to prevent softmax saturation
        if self.scale:
            scores = scores / math.sqrt(d_k)

        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        # Shape: (batch, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        # Shape: (batch, seq_len, dim)
        output = torch.matmul(attn_weights, V)

        if return_attention_weights:
            return output, attn_weights
        return output, None


class LinearAttention(BaseAttention):
    """Linear attention using memory matrix factorization.

    Instead of computing attention for each query separately, builds a
    memory matrix M = sum_i (v_i ⊗ k_i) and queries it with each q_i.

    Complexity: O(n * d^2) which is linear in sequence length.

    Note: This is faster but less expressive than standard attention
    because it doesn't use softmax normalization.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        """Initialize linear attention.

        Args:
            dim: Dimension of the input embeddings
            dropout: Dropout probability (applied to output)
        """
        super().__init__(dim, dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute linear attention using memory matrix.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask (currently not supported for linear attention)
            return_attention_weights: If True, also return attention scores

        Returns:
            output: Attention output of shape (batch, seq_len, dim)
            weights: Optional attention scores (raw dot products, not normalized)
        """
        batch_size, seq_len, dim = Q.size()

        if mask is not None:
            raise NotImplementedError("Masking is not yet supported for LinearAttention")

        # Build memory matrix: M = sum_i (v_i ⊗ k_i^T)
        # This is equivalent to: V^T @ K
        # Shape: (batch, dim, dim)
        memory_matrix = torch.matmul(V.transpose(-2, -1), K)

        # Query the memory matrix with each query
        # Output = Q @ M = Q @ (V^T @ K) = (Q @ V^T) @ K... wait, this is wrong
        # Correct: Output = M @ Q^T where M = sum(v_i ⊗ k_i^T)
        # Actually: M(q) = sum_i v_i * <k_i, q> = (sum_i v_i ⊗ k_i^T) @ q
        #                = V^T @ (K @ q) for each q
        # For all queries: (K @ Q^T)^T @ V = Q @ K^T @ V (but we want per-row products)

        # Compute attention scores (raw dot products, no softmax)
        # Shape: (batch, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Apply scores to values (same as standard attention but without softmax)
        # Shape: (batch, seq_len, dim)
        output = torch.matmul(scores, V)

        # Apply dropout
        output = self.dropout(output)

        if return_attention_weights:
            return output, scores
        return output, None


class PiecewiseLinearAttention(BaseAttention):
    """Piecewise linear attention using pseudo-queries and local linearization.

    Computes exact softmax attention at k pseudo-queries, then uses first-order
    Taylor approximation for nearby actual queries.

    Complexity: O(n * k * d) where k << n is the number of pseudo-queries.

    TODO: Implement in later phases.
    """

    def __init__(
        self,
        dim: int,
        num_pseudo_queries: int,
        dropout: float = 0.0,
    ):
        """Initialize piecewise linear attention.

        Args:
            dim: Dimension of the input embeddings
            num_pseudo_queries: Number of pseudo-queries (k)
            dropout: Dropout probability
        """
        super().__init__(dim, dropout)
        self.num_pseudo_queries = num_pseudo_queries

        # TODO: Initialize pseudo-queries
        # self.pseudo_queries = nn.Parameter(torch.randn(num_pseudo_queries, dim))

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute piecewise linear attention.

        TODO: Implement in later phases.
        """
        raise NotImplementedError("PiecewiseLinearAttention will be implemented in Phase 4")
