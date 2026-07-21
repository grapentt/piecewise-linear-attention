"""Attention mechanisms: Standard, Linear, and Piecewise.

This module provides three attention implementations:
- StandardAttention: Baseline softmax attention
- LinearAttention: Efficient linear attention with kernel feature maps
- PiecewiseAttention: Per-sample piecewise linear attention (RECOMMENDED)
"""

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .anchors import AnchorStrategy


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
            k_prime_v = K_prime.unsqueeze(-1) * V.unsqueeze(-2)
            # Cumulative sum along sequence dimension
            k_prime_v_cumsum = torch.cumsum(k_prime_v, dim=1)

            # Compute numerator: Q'[i] @ cumsum(K'[j] ⊙ V[j] for j <= i)
            # Shape: (batch, seq_len, dim)
            numerator = (k_prime_v_cumsum @ Q_prime.unsqueeze(-1)).squeeze(-1)

            # Compute normalization: Q'[i] @ cumsum(K'[j] for j <= i)
            # Shape: (batch, seq_len, 1)
            k_prime_cumsum = torch.cumsum(K_prime, dim=1)
            normalization = (Q_prime * k_prime_cumsum).sum(dim=-1, keepdim=True)
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
                Q_prime, K_prime.sum(dim=1, keepdim=True).transpose(-2, -1)
            )

        # Normalize output
        output = numerator / (normalization + self.eps)
        output = self.dropout(output)

        return output, normalization


class PerformerAttention(BaseAttention):
    """Performer attention with FAVOR+ positive random features.

    Approximates softmax attention in linear time using positive random
    features (Choromanski et al., 2021, "Rethinking Attention with
    Performers"). The softmax kernel exp(q·k / sqrt(d)) is estimated with an
    unbiased positive-feature map

        phi(x) = exp(W x - ||x||^2 / 2) / sqrt(m)

    applied to scaled queries/keys, giving

        Attention(Q, K, V) ~= phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ phi(K)^T @ 1).

    The random projection ``W`` (shape ``(num_features, dim)``) is drawn once
    from a fixed seed and stored as a registered buffer, so it moves with
    ``.to(device)`` and stays constant across forward passes — the estimator is
    deterministic for a given seed, which matters for reproducible training and
    evaluation.

    Complexity: O(batch * n * m * d).

    Parameters
    ----------
    dim:
        Per-head embedding dimension.
    dropout:
        Dropout probability applied to the output.
    num_features:
        Number of random features ``m``. Defaults to ``max(2 * dim, 64)``.
    eps:
        Numerical-stability constant added to features and denominator.
    causal:
        If True, use a causal prefix-sum formulation (query i attends to keys
        0..i).
    seed:
        Seed for the random projection; logged with experiment results so runs
        are reproducible.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        num_features: Optional[int] = None,
        eps: float = 1e-6,
        causal: bool = False,
        seed: int = 0,
    ):
        super().__init__(dim, dropout)
        self.num_features = num_features if num_features is not None else max(2 * dim, 64)
        self.eps = eps
        self.causal = causal
        self.seed = seed

        # Draw the random projection once from a fixed seed on CPU, then register
        # it as a (non-persistent) buffer so it participates in .to()/.cuda()
        # moves but is regenerated deterministically rather than stored in
        # checkpoints. Shape: (num_features, dim).
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.randn(self.num_features, dim, generator=generator)
        self.register_buffer("projection", projection, persistent=False)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the FAVOR+ positive random-feature map phi(x).

        Parameters
        ----------
        x:
            Input tensor of shape ``(..., dim)``.

        Returns
        -------
        torch.Tensor
            Positive features of shape ``(..., num_features)``.
        """
        # Scale inputs by d^{-1/4} so that dot products of scaled q, k reproduce
        # the softmax argument q·k / sqrt(d).
        scale = self.dim**0.25
        xs = x / scale
        # Projection onto random directions: (..., num_features).
        proj = xs @ self.projection.t()
        # Stabilizer term ||xs||^2 / 2 (broadcast over the feature axis).
        norm = (xs**2).sum(dim=-1, keepdim=True) / 2.0
        features = torch.exp(proj - norm) / math.sqrt(self.num_features)
        return features + self.eps

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Performer (FAVOR+) attention.

        Parameters
        ----------
        Q, K, V:
            Query/key/value tensors of shape ``(batch, seq_len, dim)``.

        Returns
        -------
        output:
            Attention output of shape ``(batch, seq_len, dim)``.
        normalization:
            Denominator terms of shape ``(batch, seq_len, 1)``.
        """
        Q_prime = self.feature_map(Q)  # (batch, seq_len, num_features)
        K_prime = self.feature_map(K)  # (batch, seq_len, num_features)

        if self.causal:
            # Prefix-sum formulation: query i attends to keys 0..i.
            # kv[j] = phi(K_j) ⊗ V_j, cumulative-summed over the sequence.
            k_prime_v = K_prime.unsqueeze(-1) * V.unsqueeze(-2)
            k_prime_v_cumsum = torch.cumsum(k_prime_v, dim=1)
            numerator = (k_prime_v_cumsum * Q_prime.unsqueeze(-1)).sum(dim=-2)

            k_prime_cumsum = torch.cumsum(K_prime, dim=1)
            normalization = (Q_prime * k_prime_cumsum).sum(dim=-1, keepdim=True)
        else:
            # Non-causal: full-context linear attention via the (m, d) memory.
            memory_matrix = torch.matmul(K_prime.transpose(-2, -1), V)  # (b, m, d)
            numerator = torch.matmul(Q_prime, memory_matrix)  # (b, n, d)
            normalization = torch.matmul(
                Q_prime,
                K_prime.sum(dim=1, keepdim=True).transpose(-2, -1),
            )  # (b, n, 1)

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
        anchor_strategy: Optional["AnchorStrategy"] = None,
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
            anchor_strategy: Optional multi-anchor strategy. When provided and it
                           yields more than one anchor, each query is linearized
                           around its nearest anchor (non-causal mode only). When
                           ``None`` or single-anchor, the original single-anchor
                           behavior is used unchanged.
        """
        super().__init__(dim, dropout)
        self.scale = scale
        self.causal = causal
        self.causal_pseudo_query = causal_pseudo_query
        if causal_pseudo_query not in ("mean", "first"):
            raise ValueError(
                f"causal_pseudo_query must be 'mean' or 'first', got {causal_pseudo_query}"
            )
        self.pseudo_query_fn = pseudo_query_fn or self._default_pseudo_query
        self.anchor_strategy = anchor_strategy
        if (
            anchor_strategy is not None
            and getattr(anchor_strategy, "num_anchors", 1) > 1
            and causal
        ):
            raise ValueError("Multi-anchor strategies are only supported in non-causal mode")

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
        pseudo_queries: torch.Tensor,
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
        # scores = pseudo_queries @ K^T for each batch
        scores = (K @ pseudo_queries.unsqueeze(-1)).squeeze(-1) * scale_factor

        # Compute attention weights: (batch, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)

        # Compute attention output: (batch, dim)
        # output = attn_weights @ V
        output = (attn_weights.unsqueeze(1) @ V).squeeze(1)

        # Compute Jacobian analytically
        # J = scale * [Σ α_i (v_i ⊗ k_i) - v̄ ⊗ k̄]
        # where v̄ = Σ α_i v_i (= output) and k̄ = Σ α_i k_i

        # Weighted sum of keys: (batch, dim)
        weighted_k = (attn_weights.unsqueeze(1) @ K).squeeze(1)

        # Weighted sum of outer products: (batch, dim, dim)
        # Σ α_i (v_i ⊗ k_i) = V^T @ diag(α) @ K
        # Equivalent to: (V * α^T)^T @ K where α is broadcast
        weighted_v = attn_weights.unsqueeze(-1) * V  # (batch, seq_len, dim)
        weighted_outer = weighted_v.transpose(-2, -1) @ K

        # Outer product of weighted sums: (batch, dim, dim)
        outer_weighted = output.unsqueeze(-1) @ weighted_k.unsqueeze(1)

        # Combine to get Jacobian
        jacobian = scale_factor * (weighted_outer - outer_weighted)

        return output, jacobian

    def _compute_multi_anchor(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Piecewise-linear attention with per-query nearest-anchor expansion.

        Each query is linearized around its nearest anchor:
        ``out_i = A(a_{j(i)}) + J(a_{j(i)}) @ (q_i - a_{j(i)})`` where ``j(i)`` is
        the index of the anchor closest to query ``i``.

        The expansion is applied in a fused form that never materializes a
        per-query ``(batch, seq_len, dim, dim)`` Jacobian. Distributing the
        matmul over the rank-1 Jacobian gives, for a query linearized around
        anchor ``j`` (``o_j`` = anchor output, ``wk_j`` = weighted keys,
        ``M_j = Σ_n α_{jn} v_n k_nᵀ``, ``Δq = q - a_j``):

            out = o_j + scale·(M_j @ Δq) − scale·o_j·(wk_jᵀ Δq),

        which is algebraically identical to ``o_j + J_j @ Δq`` but needs only the
        compact per-anchor factors ``o_j, wk_j`` (batch, m, dim) and ``M_j``
        (batch, m, dim, dim). The expansion of every query around every anchor is
        formed as a dense ``(batch, m, seq_len, dim)`` tensor and the assigned
        slice is selected along the small anchor axis ``m``. This is
        ``O(batch · m · seq_len · dim²)`` — with the constant ``m`` (typically
        2–8) it is genuinely linear in ``seq_len``, unlike a ``O(seq_len²·dim)``
        softmax, and avoids the large per-query Jacobian gather.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim).
            K: Key tensor of shape (batch, seq_len, dim).
            V: Value tensor of shape (batch, seq_len, dim).
            anchors: Anchor tensor of shape (batch, m, dim).

        Returns:
            output: Attention output of shape (batch, seq_len, dim).
        """
        _, _, dim = Q.shape
        scale_factor = 1.0 / math.sqrt(dim) if self.scale else 1.0

        # Exact softmax attention output and the compact analytic-Jacobian
        # factors at each anchor. All are (batch, m, *) — no per-query (d, d)
        # tensor is ever formed.
        scores = torch.einsum("bmd,bnd->bmn", anchors, K) * scale_factor
        alpha = torch.softmax(scores, dim=-1)  # (batch, m, seq_len)
        out_anchor = torch.einsum("bmn,bnd->bmd", alpha, V)  # (batch, m, dim)
        weighted_k = torch.einsum("bmn,bnd->bmd", alpha, K)  # (batch, m, dim)
        weighted_outer = torch.einsum("bmn,bnd,bne->bmde", alpha, V, K)  # (b,m,d,d)

        # Hard-assign each query to its nearest anchor (non-differentiable);
        # gradients still flow through the expansion below w.r.t. Q, K, V.
        # cdist has no half-precision kernel on some backends, so the distance
        # is computed in float; the result is an integer index, so the upcast is
        # free and does not affect the fp16 expansion below.
        with torch.no_grad():
            assign = torch.cdist(Q.float(), anchors.float()).argmin(dim=-1)  # (batch, seq_len)

        # Fused first-order expansion of every query around every anchor.
        # delta_all[b, j, i] = q_i - a_j.
        delta_all = Q.unsqueeze(1) - anchors.unsqueeze(2)  # (batch, m, seq_len, dim)
        # M_j @ Δq for all anchors/queries.
        m_term = torch.einsum("bmde,bmne->bmnd", weighted_outer, delta_all)  # (b,m,n,d)
        # Rank-1 term o_j·(wk_jᵀ Δq) without forming any (d, d) product.
        wk_dot = torch.einsum("bmd,bmnd->bmn", weighted_k, delta_all)  # (b,m,n)
        rank1 = out_anchor.unsqueeze(2) * wk_dot.unsqueeze(-1)  # (b,m,n,d)
        expanded = out_anchor.unsqueeze(2) + scale_factor * (m_term - rank1)  # (b,m,n,d)

        # Select each query's assigned-anchor slice along the anchor axis m.
        # This is a cheap O(seq_len·dim) gather, not the O(seq_len·dim²) gather
        # of a full per-query Jacobian.
        idx = assign.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, dim)  # (b,1,n,d)
        return torch.gather(expanded, 1, idx).squeeze(1)  # (batch, seq_len, dim)

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
            anchor_scores = (K @ q_bar.unsqueeze(-1)).squeeze(-1) * scale_factor
            s = torch.exp(anchor_scores)  # (batch, seq_len)

            # Step 3: Precompute cumsum terms (all independent of q_i!)
            # A_j = s_j * v_j  (weighted values)
            A = s.unsqueeze(-1) * V  # (batch, seq_len, dim)

            # B_j = s_j * (k_j ⊗ v_j)  (weighted outer products for numerator)
            # Shape: (batch, seq_len, dim, dim)
            s_expanded = s.unsqueeze(-1).unsqueeze(-1)  # (batch, seq_len, 1, 1)
            B = s_expanded * K.unsqueeze(-1) * V.unsqueeze(-2)  # (batch, seq_len, dim, dim)

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
            # Use batched matrix-vector multiplication: (b,n,d,d) @ (b,n,d,1) -> (b,n,d,1)
            numerator = S_A + scale_factor * (S_B @ delta_q.unsqueeze(-1)).squeeze(-1)

            # Step 7: Compute denominator for each position i
            # Denominator_i = S^C_i + scale_factor * (q_i - q̄) @ S^D_i
            # Shape: (batch, seq_len)
            # Element-wise dot product: sum over dimension
            denominator = S_C + scale_factor * (delta_q * S_D).sum(dim=-1)

            # Step 8: Normalize
            output = numerator / (denominator.unsqueeze(-1) + 1e-6)

            # Return anchor query as pseudo_queries for API compatibility
            pseudo_queries = q_bar
        else:
            # Non-causal piecewise attention.
            use_multi = (
                self.anchor_strategy is not None
                and getattr(self.anchor_strategy, "num_anchors", 1) > 1
            )
            if use_multi:
                # Multi-anchor mode: each query is linearized around its nearest
                # anchor. The anchors double as the reported pseudo_queries.
                anchors = self.anchor_strategy.select(Q)  # (batch, m, dim)
                output = self._compute_multi_anchor(Q, K, V, anchors)
                pseudo_queries = anchors
            else:
                # Single-anchor mode (original implementation).
                # Step 1: Select representative query for each batch sample.
                if self.anchor_strategy is not None:
                    # A single-anchor strategy yields (batch, 1, dim); squeeze it
                    # back to the legacy (batch, dim) representative query.
                    pseudo_queries = self.anchor_strategy.select(Q).squeeze(1)
                else:
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
                # Batched matrix-vector multiplication
                # Shape: (batch, seq_len, dim)
                jacobian_deltas = (jacobians @ deltas.transpose(-2, -1)).transpose(-2, -1)

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
    "PerformerAttention",
    "PiecewiseAttention",
    "EfficientAttention",
]
