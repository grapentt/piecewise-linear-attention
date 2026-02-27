"""Attention mechanisms: Standard, Linear, and Piecewise.

This module provides three attention implementations:
- StandardAttention: Baseline softmax attention
- LinearAttention: Efficient linear attention with kernel feature maps
- PiecewiseAttention: Per-sample piecewise linear attention (RECOMMENDED)
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAttention(nn.Module, ABC):
    """Abstract base class for attention mechanisms."""

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention output.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask

        Returns:
            output: Attention output of shape (batch, seq_len, dim)
            extra: Optional extra information (implementation-specific)
        """
        pass


class StandardAttention(BaseAttention):
    """Standard scaled dot-product attention with softmax.

    Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

    Complexity: O(batch * n^2 * d)
    Memory: O(batch * n^2)
    """

    def __init__(self, dim: int, dropout: float = 0.0, scale: bool = True):
        """Initialize standard attention.

        Args:
            dim: Dimension of the input embeddings
            dropout: Dropout probability for attention weights
            scale: Whether to scale attention scores by sqrt(d)
        """
        super().__init__(dim, dropout)
        self.scale = scale

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute standard softmax attention.

        Returns:
            output: Attention output of shape (batch, seq_len, dim)
            attn_weights: Attention weights of shape (batch, seq_len, seq_len)
        """
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale by sqrt(d)
        if self.scale:
            scores = scores / math.sqrt(self.dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


class LinearAttention(BaseAttention):
    """Linear attention with kernel feature maps.

    Uses φ(Q) @ (φ(K)^T @ V) with proper normalization to approximate
    softmax attention while maintaining linear complexity.

    Complexity: O(batch * n * d^2)
    Memory: O(batch * d^2)

    Reference: "Transformers are RNNs" (Katharopoulos et al., 2020)
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        kernel: str = "elu",
        eps: float = 1e-6,
    ):
        """Initialize linear attention.

        Args:
            dim: Dimension of the input embeddings
            dropout: Dropout probability
            kernel: Kernel function ('elu', 'relu', 'softplus')
            eps: Small constant for numerical stability
        """
        super().__init__(dim, dropout)
        self.kernel = kernel
        self.eps = eps

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply kernel feature map φ(x) to approximate softmax.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            φ(x): Feature-mapped tensor of same shape, guaranteed positive
        """
        if self.kernel == "elu":
            # elu(x) + 1 - standard choice from "Transformers are RNNs"
            return F.elu(x) + 1.0
        elif self.kernel == "relu":
            # ReLU(x) + eps
            return F.relu(x) + self.eps
        elif self.kernel == "softplus":
            # softplus(x) = log(1 + exp(x))
            return F.softplus(x)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}. Use 'elu', 'relu', or 'softplus'.")

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear attention with kernel feature maps.

        Returns:
            output: Attention output of shape (batch, seq_len, dim)
            normalization: Normalization terms of shape (batch, seq_len, 1)
        """
        if mask is not None:
            raise NotImplementedError("Masking not yet supported for LinearAttention")

        # Apply feature maps
        Q_prime = self.feature_map(Q)
        K_prime = self.feature_map(K)

        # Build memory matrix: M = K'^T @ V
        # Shape: (batch, dim, dim)
        memory_matrix = torch.matmul(K_prime.transpose(-2, -1), V)

        # Compute numerator: φ(Q) @ M
        # Shape: (batch, seq_len, dim)
        numerator = torch.matmul(Q_prime, memory_matrix)

        # Compute normalization: φ(Q) @ φ(K)^T @ 1
        # Shape: (batch, seq_len, 1)
        normalization = torch.matmul(
            Q_prime,
            K_prime.sum(dim=1, keepdim=True).transpose(-2, -1)
        )

        # Normalize output
        output = numerator / (normalization + self.eps)
        output = self.dropout(output)

        return output, normalization


class PiecewiseAttention(BaseAttention):
    """Piecewise linear attention using first-order Taylor approximation.

    Algorithm:
        For each batch sample:
        1. Select representative query q̃ (default: mean of all queries)
        2. Compute exact softmax attention A(q̃) and analytical Jacobian J
        3. For each query q: output ≈ A(q̃) + J @ (q - q̃)

    Complexity: O(batch * n * d^2)
    Memory: O(batch * d^2)
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        scale: bool = True,
        pseudo_query_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """Initialize piecewise attention.

        Args:
            dim: Dimension of the input embeddings
            dropout: Dropout probability
            scale: Whether to scale attention scores by sqrt(d)
            pseudo_query_fn: Optional custom function to select pseudo-query.
                           Signature: fn(Q: (batch, seq_len, dim)) -> (batch, dim)
                           Default: mean of all queries per sample
        """
        super().__init__(dim, dropout)
        self.scale = scale
        self.pseudo_query_fn = pseudo_query_fn or self._default_pseudo_query

    def _default_pseudo_query(self, Q: torch.Tensor) -> torch.Tensor:
        """Default pseudo-query selector: mean of all queries in each sample.

        This works well when queries within a sample are somewhat similar,
        which is common in many NLP tasks.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)

        Returns:
            pseudo_queries: Mean queries of shape (batch, dim)
        """
        return Q.mean(dim=1)

    def _compute_attention_and_jacobian(
        self,
        pseudo_query: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute exact attention output and Jacobian at pseudo-query.

        Uses analytical Jacobian formula for softmax attention:
        J = scale * [Σ_i α_i (v_i ⊗ k_i) - (Σ_i α_i v_i) ⊗ (Σ_i α_i k_i)]

        Args:
            pseudo_query: Representative query, shape (batch, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)
            mask: Optional mask of shape (batch, seq_len) where 0 masks out

        Returns:
            output: Attention output at pseudo-query, shape (batch, dim)
            jacobian: Jacobian matrix at pseudo-query, shape (batch, dim, dim)
        """
        batch, seq_len, dim = K.shape
        scale_factor = 1.0 / math.sqrt(dim) if self.scale else 1.0

        # Compute attention scores: (batch, seq_len)
        scores = torch.einsum('bd,bnd->bn', pseudo_query, K)
        scores = scores * scale_factor

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights: (batch, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)

        # Compute attention output: (batch, dim)
        output = torch.einsum('bn,bnd->bd', attn_weights, V)

        # Compute Jacobian analytically
        # J = scale * [Σ α_i (v_i ⊗ k_i) - v̄ ⊗ k̄]
        # where v̄ = Σ α_i v_i (= output) and k̄ = Σ α_i k_i

        # Weighted sum of keys: (batch, dim)
        weighted_k = torch.einsum('bn,bnd->bd', attn_weights, K)

        # Weighted sum of outer products: (batch, dim, dim)
        weighted_outer = torch.einsum('bn,bni,bnj->bij', attn_weights, V, K)

        # Outer product of weighted sums: (batch, dim, dim)
        outer_weighted = torch.einsum('bi,bj->bij', output, weighted_k)

        # Combine to get Jacobian
        jacobian = scale_factor * (weighted_outer - outer_weighted)

        return output, jacobian

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute piecewise linear attention.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)
            mask: Optional mask of shape (batch, seq_len) where 0 masks out

        Returns:
            output: Attention output of shape (batch, seq_len, dim)
            pseudo_queries: Representative queries used, shape (batch, dim)
        """
        batch, seq_len, dim = Q.shape

        # Step 1: Select representative query for each batch sample
        pseudo_queries = self.pseudo_query_fn(Q)  # (batch, dim)

        # Step 2: Compute exact attention and Jacobian at pseudo-queries
        pseudo_outputs, jacobians = self._compute_attention_and_jacobian(
            pseudo_queries, K, V, mask
        )  # (batch, dim), (batch, dim, dim)

        # Step 3: Apply first-order Taylor approximation to all queries
        # output[b, i] = pseudo_outputs[b] + jacobians[b] @ (Q[b, i] - pseudo_queries[b])

        # Compute deltas: Q - pseudo_queries
        # Shape: (batch, seq_len, dim)
        deltas = Q - pseudo_queries.unsqueeze(1)

        # Apply Jacobian to deltas: J @ Δq
        # Use einsum for efficient batched matrix-vector multiplication
        # Shape: (batch, seq_len, dim)
        jacobian_deltas = torch.einsum('bij,bsj->bsi', jacobians, deltas)

        # Final output: A(q̃) + J @ Δq
        # Shape: (batch, seq_len, dim)
        output = pseudo_outputs.unsqueeze(1) + jacobian_deltas

        # Apply dropout
        output = self.dropout(output)

        return output, pseudo_queries


# Convenient alias
EfficientAttention = PiecewiseAttention


__all__ = [
    "BaseAttention",
    "StandardAttention",
    "LinearAttention",
    "PiecewiseAttention",
    "EfficientAttention",
]
