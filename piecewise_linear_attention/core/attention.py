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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention output.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)

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

    def __init__(self, dim: int, dropout: float = 0.0, scale: bool = True, causal: bool = False):
        """Initialize standard attention.

        Args:
            dim: Dimension of the input embeddings
            dropout: Dropout probability for attention weights
            scale: Whether to scale attention scores by sqrt(d)
            causal: If True, apply causal masking (query i can only attend to keys 0...i)
        """
        super().__init__(dim, dropout)
        self.scale = scale
        self.causal = causal

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute standard softmax attention.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)

        Returns:
            output: Attention output of shape (batch, seq_len, dim)
            attn_weights: Attention weights of shape (batch, seq_len, seq_len)
        """
        _, seq_len, _ = Q.shape

        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale by sqrt(d)
        if self.scale:
            scores = scores / math.sqrt(self.dim)

        # Apply causal mask if enabled
        if self.causal:
            # Create causal mask: lower triangular matrix (1 for allowed, 0 for blocked)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device))
            scores = scores.masked_fill(causal_mask == 0, -1e9)

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
        kernel_type: str = "elu",
        eps: float = 1e-6,
        causal: bool = False,
    ):
        """Initialize linear attention.

        Args:
            dim: Dimension of the input embeddings
            dropout: Dropout probability
            kernel_type: Kernel function ('elu', 'relu', 'softplus')
            eps: Small constant for numerical stability
            causal: If True, use causal masking (query i can only attend to keys 0...i)
        """
        super().__init__(dim, dropout)
        self.kernel = kernel_type
        self.eps = eps
        self.causal = causal

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear attention with kernel feature maps.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)

        Returns:
            output: Attention output of shape (batch, seq_len, dim)
            normalization: Normalization terms of shape (batch, seq_len, 1)
        """
        # Apply feature maps
        Q_prime = self.feature_map(Q)  # (batch, seq_len, dim)
        K_prime = self.feature_map(K)  # (batch, seq_len, dim)

        if self.causal:
            # Causal masking using cumulative sum
            # For each query position i, we can only attend to keys 0...i

            # Compute cumulative key-value products: cumsum(K' ⊙ V)
            # Shape: (batch, seq_len, dim, dim)
            k_prime_v = torch.einsum('bnd,bnm->bndm', K_prime, V)
            # Cumulative sum along sequence dimension
            k_prime_v_cumsum = torch.cumsum(k_prime_v, dim=1)

            # Compute numerator: Q'[i] @ cumsum(K'[j] ⊙ V[j] for j <= i)
            # Shape: (batch, seq_len, dim)
            numerator = torch.einsum('bnd,bndm->bnm', Q_prime, k_prime_v_cumsum)

            # Compute normalization: Q'[i] @ cumsum(K'[j] for j <= i)
            # Shape: (batch, seq_len, 1)
            k_prime_cumsum = torch.cumsum(K_prime, dim=1)
            normalization = torch.einsum('bnd,bnd->bn', Q_prime, k_prime_cumsum).unsqueeze(-1)
        else:
            # Non-causal linear attention (original implementation)
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

    Both causal and non-causal modes compute the same Jacobian-based
    approximation: A(q) ≈ A(q̃) + J_A(q̃) @ (q - q̃)

    The difference is computational strategy:
    - Non-causal: Computes Jacobian explicitly via chain rule. One Jacobian
                  shared by all positions. O(batch * d²) memory.
    - Causal: Computes Jacobian implicitly by linearizing exp(q·k) before
              summation, enabling efficient cumsum for position-dependent
              masking. One Jacobian per position. O(batch * n * d²) memory.

    The causal implementation linearizes the exponential kernel first, which
    mathematically gives the same Jacobian but enables cumsum optimization.

    Complexity: O(batch * n * d²) for both modes
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        scale: bool = True,
        pseudo_query_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        causal: bool = False,
        causal_pseudo_query: str = "mean",
    ):
        """Initialize piecewise attention.

        Args:
            dim: Dimension of the input embeddings
            dropout: Dropout probability
            scale: Whether to scale attention scores by sqrt(d)
            pseudo_query_fn: Optional custom function to select pseudo-query.
                           Signature: fn(Q: (batch, seq_len, dim)) -> (batch, dim)
                           Default: mean of all queries per sample (non-causal mode)
            causal: If True, use causal masking (query i can only attend to keys 0...i)
            causal_pseudo_query: Strategy for selecting pseudo-query in causal mode.
                               "mean": mean of all queries (best for training with full sequence)
                               "first": first query (best for autoregressive inference)
        """
        super().__init__(dim, dropout)
        self.scale = scale
        self.causal = causal
        self.causal_pseudo_query = causal_pseudo_query
        if causal_pseudo_query not in ("mean", "first"):
            raise ValueError(f"causal_pseudo_query must be 'mean' or 'first', got {causal_pseudo_query}")
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute exact attention output and Jacobian at pseudo-query.

        Uses analytical Jacobian formula for softmax attention:
        J = scale * [Σ_i α_i (v_i ⊗ k_i) - (Σ_i α_i v_i) ⊗ (Σ_i α_i k_i)]

        Args:
            pseudo_query: Representative query, shape (batch, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)

        Returns:
            output: Attention output at pseudo-query, shape (batch, dim)
            jacobian: Jacobian matrix at pseudo-query, shape (batch, dim, dim)
        """
        _, _, dim = K.shape
        scale_factor = 1.0 / math.sqrt(dim) if self.scale else 1.0

        # Compute attention scores: (batch, seq_len)
        scores = torch.einsum('bd,bnd->bn', pseudo_query, K)
        scores = scores * scale_factor

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute piecewise linear attention.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)

        Returns:
            output: Attention output of shape (batch, seq_len, dim)
            pseudo_queries: Representative queries used, shape (batch, dim)
        """
        _, _, dim = Q.shape

        if self.causal:
            # Causal piecewise attention: Jacobian computation via cumsum
            # Complexity: O(n·d²) vs O(n²·d) naive causal attention

            scale_factor = 1.0 / math.sqrt(dim) if self.scale else 1.0

            # Step 1: Select fixed anchor query q̄
            if self.causal_pseudo_query == "mean":
                # Use mean of ALL queries (best for training with full sequence)
                q_bar = Q.mean(dim=1)  # (batch, dim)
            elif self.causal_pseudo_query == "first":
                # Use first query (best for autoregressive inference)
                q_bar = Q[:, 0, :]  # (batch, dim)

            # Step 2: Compute anchor attention scores s_j = exp(q̄·k_j / √d)
            # Shape: (batch, seq_len)
            anchor_scores = torch.einsum('bd,bnd->bn', q_bar, K) * scale_factor
            s = torch.exp(anchor_scores)  # (batch, seq_len)

            # Step 3: Precompute cumsum terms (all independent of q_i!)
            # A_j = s_j * v_j  (weighted values)
            A = s.unsqueeze(-1) * V  # (batch, seq_len, dim)

            # B_j = s_j * (k_j ⊗ v_j)  (weighted outer products for numerator)
            # Shape: (batch, seq_len, dim, dim)
            B = torch.einsum('bn,bni,bnj->bnij', s, K, V)

            # C_j = s_j  (weighted counts for denominator)
            C = s  # (batch, seq_len)

            # D_j = s_j * k_j  (weighted keys for denominator)
            D = s.unsqueeze(-1) * K  # (batch, seq_len, dim)

            # Step 4: Apply cumulative sum along sequence dimension
            # These give us Σ_{j=1}^i for each position i
            S_A = torch.cumsum(A, dim=1)  # (batch, seq_len, dim)
            S_B = torch.cumsum(B, dim=1)  # (batch, seq_len, dim, dim)
            S_C = torch.cumsum(C, dim=1)  # (batch, seq_len)
            S_D = torch.cumsum(D, dim=1)  # (batch, seq_len, dim)

            # Step 5: Compute query deltas
            delta_q = Q - q_bar.unsqueeze(1)  # (batch, seq_len, dim)

            # Step 6: Compute numerator for each position i
            # Numerator_i = S^A_i + scale_factor * S^B_i @ (q_i - q̄)
            # Shape: (batch, seq_len, dim)
            numerator = S_A + scale_factor * torch.einsum('bnij,bnj->bni', S_B, delta_q)

            # Step 7: Compute denominator for each position i
            # Denominator_i = S^C_i + scale_factor * (q_i - q̄) @ S^D_i
            # Shape: (batch, seq_len)
            denominator = S_C + scale_factor * torch.einsum('bni,bni->bn', delta_q, S_D)

            # Step 8: Normalize
            output = numerator / (denominator.unsqueeze(-1) + 1e-6)

            # Return anchor query as pseudo_queries for API compatibility
            pseudo_queries = q_bar
        else:
            # Non-causal piecewise attention (original implementation)
            # Step 1: Select representative query for each batch sample
            pseudo_queries = self.pseudo_query_fn(Q)  # (batch, dim)

            # Step 2: Compute exact attention and Jacobian at pseudo-queries
            pseudo_outputs, jacobians = self._compute_attention_and_jacobian(
                pseudo_queries, K, V
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
