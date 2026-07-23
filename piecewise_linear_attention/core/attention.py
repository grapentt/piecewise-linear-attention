"""Attention mechanisms: Standard, Linear, and Piecewise.

This module provides three attention implementations:
- StandardAttention: Baseline softmax attention
- LinearAttention: Efficient linear attention with kernel feature maps
- PiecewiseAttention: Per-sample piecewise linear attention (RECOMMENDED)
"""

import math
import warnings
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
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention output.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)
            key_padding_mask: Optional bool tensor ``(batch, key_len)`` where
                ``True`` marks a valid key and ``False`` a padded one. When given,
                padded key positions must contribute nothing to any query's output.
                Each mechanism applies it in the axis its own aggregation requires
                (softmax-over-keys methods add a large negative bias ``-1e9`` to
                padded scores before the softmax; feature-map methods mask ``φ(K)``
                in numerator *and* normalizer). Non-causal use only; ``None``
                attends over all keys. Assumes at least one valid key per row (a
                fully-padded row would get a uniform, not undefined, weighting).

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
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute standard softmax attention.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)
            key_padding_mask: Optional bool ``(batch, key_len)``, True=valid. Padded
                key columns get score ``-1e9`` before the softmax so they receive
                ~zero weight and leave both numerator and denominator. Zeroing K/V
                rows would NOT work here: a zero score still yields ``exp(0)=1`` in
                the softmax denominator, corrupting the normalizer.

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

        # Exclude padded key positions: set their score to -1e9 so softmax gives
        # them ~zero weight and they drop out of the denominator. Broadcast the
        # (batch, key_len) mask over the query axis.
        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask.unsqueeze(1), -1e9)

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
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear attention with kernel feature maps.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)
            key_padding_mask: Optional bool ``(batch, key_len)``, True=valid.
                Applied by zeroing ``φ(K)`` rows at padded positions (non-causal
                only). This is essential and subtle: the normalizer sums ``φ(K)``
                over all keys, and ``φ(0) ≠ 0`` (e.g. ``elu(0)+1 = 1``), so zeroing
                *raw* K would leave a spurious ``φ(0)`` in the denominator. Masking
                ``φ(K)`` itself removes padded keys from both numerator and
                normalizer.

        Returns:
            output: Attention output of shape (batch, seq_len, dim)
            normalization: Normalization terms of shape (batch, seq_len, 1)
        """
        # Apply feature maps
        Q_prime = self.feature_map(Q)  # (batch, seq_len, dim)
        K_prime = self.feature_map(K)  # (batch, seq_len, dim)

        # Exclude padded keys by zeroing their feature rows. Must be applied to
        # φ(K), not raw K: φ(0) is nonzero, so masking raw K would still pollute
        # the K_prime.sum normalizer below. Non-causal path only.
        if key_padding_mask is not None and not self.causal:
            K_prime = K_prime * key_padding_mask.unsqueeze(-1).to(K_prime.dtype)

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
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Performer (FAVOR+) attention.

        Parameters
        ----------
        Q, K, V:
            Query/key/value tensors of shape ``(batch, seq_len, dim)``.
        key_padding_mask:
            Optional bool ``(batch, key_len)``, True=valid. Applied by zeroing
            ``φ(K)`` rows at padded positions (non-causal only). As with linear
            attention this must mask ``φ(K)`` rather than raw K: the FAVOR+ map has
            ``φ(0) = 1/sqrt(m) + eps > 0``, so a zeroed raw-K row would still add a
            spurious positive term to the ``K_prime.sum`` normalizer. Multiplying
            ``φ(K)`` by the mask (post-feature-map) also strips the ``+eps``.

        Returns
        -------
        output:
            Attention output of shape ``(batch, seq_len, dim)``.
        normalization:
            Denominator terms of shape ``(batch, seq_len, 1)``.
        """
        Q_prime = self.feature_map(Q)  # (batch, seq_len, num_features)
        K_prime = self.feature_map(K)  # (batch, seq_len, num_features)

        # Exclude padded keys by zeroing their feature rows (φ(K), post-map, so the
        # +eps is stripped too). φ(0) is strictly positive here, so masking raw K
        # would leak into the normalizer; masking φ(K) removes padded keys from both
        # numerator and denominator. Non-causal path only.
        if key_padding_mask is not None and not self.causal:
            K_prime = K_prime * key_padding_mask.unsqueeze(-1).to(K_prime.dtype)

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

    # Minimum cluster size for estimating a query-residual subspace in the
    # reduced-rank build; smaller clusters fall back to the exact dense M_j.
    _MIN_CLUSTER = 8

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        scale: bool = True,
        pseudo_query_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        causal: bool = False,
        causal_pseudo_query: str = "mean",
        anchor_strategy: Optional["AnchorStrategy"] = None,
        diagonal_only: bool = False,
        build_topp: Optional[float] = None,
        build_reduced_d: Optional[float] = None,
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
            diagonal_only: If True, drop the mean-field (softmax-covariance)
                           term ``v̄ ⊗ k̄`` from the Jacobian, retaining only the
                           uncentered second moment ``Σ_i α_i (v_i ⊗ k_i)``. This
                           is an ablation switch, not a diagonal approximation: the
                           retained term is still a full dense ``(dim, dim)`` matrix.
                           Default False keeps the exact analytic Jacobian.
            build_topp: Optional top-p (nucleus) truncation for the multi-anchor
                           Jacobian build. When ``None`` (default) each per-anchor
                           second-moment operator ``M_j`` and weighted-key vector
                           ``wk_j`` are built from all ``seq_len`` keys (exact).
                           When set to ``p`` in ``(0, 1]`` (e.g. ``0.999``) they are
                           built from only the smallest set of top-weighted keys whose
                           cumulative softmax mass reaches ``p``, cutting the build's
                           sequence axis from ``seq_len`` to ``t ≪ seq_len`` in the
                           peaked regime. The zeroth-order term ``A(a_j)`` stays exact
                           (built from all keys), so the anchor value is unchanged and
                           only the linear correction is approximated. On a diffuse
                           (near-uniform) attention distribution ``t`` grows back to
                           ``seq_len`` and the build silently reverts to exact — it
                           trades speed, never accuracy. Ignored in single-anchor and
                           causal modes.

                           Whether it is a net *speedup* is entirely contingent on
                           how peaked the per-anchor softmax is: the win requires
                           the nucleus to be a small fraction of the sequence, since
                           the sort and gather it adds are only repaid once the
                           truncated ``t`` is well below ``seq_len``. Empirically the
                           crossover is around a nucleus of ~5–10% of the keys; above
                           that the sort/gather overhead makes it slower than the
                           dense build. It is therefore off by default and intended
                           for distributions with concentrated attention.

                           Measured on real trained-model activations, this shared
                           per-batch cap does *not* pay off: because the cap is sized
                           to the worst (most diffuse) anchor and k-means anchors are
                           query-side centroids, the worst-anchor nucleus stays near
                           the full sequence (~0.9 of the keys at ``p=0.99``), so it
                           self-guards to the dense build. The median anchor's nucleus
                           is much smaller, so a *per-anchor* (ragged) cap could still
                           help — that is a separate, unimplemented build path.
            build_reduced_d: Optional reduced-rank truncation of the multi-anchor
                           Jacobian build along the **query-residual** axis. When
                           ``None`` (default) each ``M_j`` is built as the full
                           ``(dim, dim)`` operator (exact). When set to a variance
                           threshold ``p`` in ``(0, 1]`` (e.g. ``0.95``), it exploits
                           that ``M_j`` is only ever consumed as ``M_j·Δq`` where
                           ``Δq = q − a_j``: within a k-means cluster the residuals
                           ``Δq`` span a low-rank subspace, so ``M_j`` is built only on
                           the ``d′_j`` directions that reach cumulative variance ``p``
                           (per-anchor SVD of the residuals), giving ``M_j^red`` of shape
                           ``(dim, d′_j)`` at build cost ``n·dim·d′_j`` instead of
                           ``n·dim²``. The apply becomes
                           ``M_j^red @ (R_jᵀ Δq)``. Only ``M_j`` is approximated: the
                           zeroth-order anchor value ``o_j`` and the rank-1 ``wk_j``
                           term stay exact.

                           Self-guarding — it trades speed, never accuracy: any anchor
                           whose ``d′_j`` is not meaningfully below ``dim`` (``> 0.5·dim``),
                           any cluster too small to estimate a subspace, and any SVD
                           failure all fall back to the exact dense ``M_j`` (identity
                           basis). As ``p → 1`` the retained subspace grows to the full
                           residual span and the output approaches the dense build (up to
                           float reassociation).

                           Mutually exclusive with ``build_topp`` (both truncate the
                           Jacobian build, along different axes). Ignored in single-anchor
                           and causal modes. Whether the reduced FLOPs convert to a
                           wall-clock speedup depends on the per-anchor SVD cost, which is
                           paid every forward when anchors are re-fit — profile before
                           relying on it for speed.
        """
        super().__init__(dim, dropout)
        self.scale = scale
        self.causal = causal
        self.diagonal_only = diagonal_only
        if build_topp is not None and not (0.0 < build_topp <= 1.0):
            raise ValueError(f"build_topp must be in (0, 1], got {build_topp}")
        self.build_topp = build_topp
        if build_reduced_d is not None and not (0.0 < build_reduced_d <= 1.0):
            raise ValueError(f"build_reduced_d must be in (0, 1], got {build_reduced_d}")
        if build_reduced_d is not None and build_topp is not None:
            raise ValueError(
                "build_reduced_d and build_topp are mutually exclusive; both truncate "
                "the Jacobian build (along the query-residual and key axes respectively)."
            )
        self.build_reduced_d = build_reduced_d
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
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute exact attention output and Jacobian at pseudo-query.

        Uses analytical Jacobian formula for softmax attention:
        J = scale * [Σ_i α_i (v_i ⊗ k_i) - (Σ_i α_i v_i) ⊗ (Σ_i α_i k_i)]

        Args:
            pseudo_query: Representative query, shape (batch, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)
            key_padding_mask: Optional bool ``(batch, seq_len)``, True=valid. Padded
                key scores are set to ``-inf`` before the softmax so they receive
                ~zero weight; the Jacobian factors below reuse ``attn_weights`` and
                inherit the exclusion automatically.

        Returns:
            output: Attention output at pseudo-query, shape (batch, dim)
            jacobian: Jacobian matrix at pseudo-query, shape (batch, dim, dim)
        """
        _, _, dim = K.shape
        scale_factor = 1.0 / math.sqrt(dim) if self.scale else 1.0

        # Compute attention scores: (batch, seq_len)
        # scores = pseudo_queries @ K^T for each batch
        scores = (K @ pseudo_queries.unsqueeze(-1)).squeeze(-1) * scale_factor

        # Exclude padded keys before the softmax over the key axis.
        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask, -1e9)

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

        # Combine to get Jacobian. When diagonal_only is set, the mean-field
        # term v̄ ⊗ k̄ is dropped (ablation); the retained weighted_outer is
        # still a full dense (batch, dim, dim) matrix, not a diagonal.
        if self.diagonal_only:
            jacobian = scale_factor * weighted_outer
        else:
            jacobian = scale_factor * (weighted_outer - outer_weighted)

        return output, jacobian

    def _build_jacobian_factors_topp(
        self,
        alpha: torch.Tensor,
        V: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the weighted-key and second-moment Jacobian factors from a
        per-anchor top-p (nucleus) subset of keys.

        The build of ``M_j = Σ_n α_{jn} v_n k_nᵀ`` streams all ``seq_len`` keys
        under each anchor's softmax weights ``α_j`` — the dominant cost at high
        anchor count, and memory-bandwidth-bound. On real activations ``α_j`` is
        peaked, so almost all of the mass sits on a handful of keys. This keeps,
        per anchor, only the smallest set of top-weighted keys whose cumulative
        mass reaches ``self.build_topp``, and builds the factors from that subset.
        The retained weights are used *unrenormalized*, so the result is the exact
        partial sum over the nucleus — the truncated tail (mass ``< 1 − p``) is
        simply dropped, matching the dense build as ``p → 1``.

        A single batched cap ``t = max_j |nucleus_j|`` is used across anchors so the
        contraction stays one matmul; anchors with a smaller nucleus have their
        surplus slots zero-weighted and contribute nothing. When ``α`` is diffuse
        the nucleus grows to all ``seq_len`` keys and this reproduces the dense
        build (up to float reassociation) -- in that limit the gathered
        ``(batch, m, t, dim)`` factors grow to ``(batch, m, seq_len, dim)``, so peak
        *memory*, not just FLOPs, reverts to the dense cost.

        Args:
            alpha: Per-anchor softmax weights, shape (batch, m, seq_len).
            V: Value tensor, shape (batch, seq_len, dim).
            K: Key tensor, shape (batch, seq_len, dim).

        Returns:
            weighted_k: Σ over the nucleus of α·k, shape (batch, m, dim).
            second_moment: Σ over the nucleus of α·(v ⊗ k), shape (batch, m, dim, dim).
        """
        b, m, _ = alpha.shape

        # Sort weights descending per anchor and find each anchor's nucleus size:
        # the fewest top keys whose cumulative mass reaches build_topp. Include the
        # key that crosses the threshold (keep where the PRECEDING cumsum < p).
        sorted_alpha, sort_idx = torch.sort(alpha, dim=-1, descending=True)  # (b, m, n)
        cumulative = sorted_alpha.cumsum(dim=-1)
        keep = torch.ones_like(sorted_alpha, dtype=torch.bool)
        keep[..., 1:] = cumulative[..., :-1] < self.build_topp
        # Batched cap: the largest nucleus across all anchors (sync on a scalar).
        t = int(keep.sum(dim=-1).max().item())

        # Restrict to the first t sorted slots and zero the weights past each
        # anchor's own nucleus so shorter nuclei contribute exactly nothing.
        top_idx = sort_idx[..., :t]  # (b, m, t)
        alpha_kept = torch.where(
            keep[..., :t], sorted_alpha[..., :t], torch.zeros_like(sorted_alpha[..., :t])
        )  # (b, m, t)

        # Gather the corresponding keys and values per anchor: (b, m, t, dim).
        # Advanced-index into V, K directly on the sequence axis so the full
        # (b, m, n, dim) broadcast of V/K is never formed — gathering through an
        # expanded view would re-materialize exactly the m× intermediate the
        # dense build was rewritten to avoid.
        batch_idx = torch.arange(b, device=alpha.device).view(b, 1, 1).expand(b, m, t)
        V_top = V[batch_idx, top_idx]  # (b, m, t, dim)
        K_top = K[batch_idx, top_idx]  # (b, m, t, dim)

        weighted_k = torch.einsum("bmt,bmtd->bmd", alpha_kept, K_top)  # (b, m, dim)
        second_moment = torch.einsum(
            "bmt,bmtd,bmte->bmde", alpha_kept, V_top, K_top
        )  # (b, m, dim, dim)
        return weighted_k, second_moment

    def _build_jacobian_factors_reduced(
        self,
        alpha: torch.Tensor,
        V: torch.Tensor,
        K: torch.Tensor,
        Q: torch.Tensor,
        anchors: torch.Tensor,
        assign: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a reduced-rank second-moment operator on the query-residual axis.

        ``M_j = Σ_n α_{jn} v_n k_nᵀ`` is only ever consumed as ``M_j·Δq`` with
        ``Δq = q_i − a_j`` for queries assigned to anchor ``j``. Within a k-means
        cluster the residuals ``Δq`` span a low-rank subspace, so ``M_j`` need only
        be built on that subspace. For each anchor we take the residuals of its
        assigned queries, find an orthonormal basis ``R_j`` (right singular vectors)
        for the smallest subspace reaching cumulative variance ``self.build_reduced_d``,
        and build the reduced operator directly:

            M_j^red = Σ_n α_{jn} v_n (R_jᵀ k_n)ᵀ,   shape (dim, d′_j)

        at build cost ``n·dim·d′_j`` rather than ``n·dim²``. The apply reconstructs
        ``M_j·Δq ≈ M_j^red @ (R_jᵀ Δq)`` — exact on the retained subspace, dropping
        only the residual tail beyond ``d′_j`` (Eckart–Young bounds its energy by the
        discarded ``1 − p`` variance share).

        Self-guarding — trades speed, never accuracy. Three cases fall back to the
        exact dense ``M_j`` via an **identity basis** ``R_j = I`` with ``d′_j = dim``
        (so the padded apply reproduces the dense operator on that anchor): a cluster
        with fewer than ``_MIN_CLUSTER`` members (too few to estimate a subspace),
        an SVD that fails to converge, and an anchor whose required rank is not
        meaningfully below ``dim`` (``d′_j > 0.5·dim``).

        The per-anchor bases have ragged widths ``d′_j``; they are padded to a single
        batched width ``d′_max = max_j d′_j`` so the downstream build and apply stay one
        matmul each (mirroring the capacity-bucket apply). Anchors narrower than
        ``d′_max`` have their surplus columns zeroed and contribute nothing.

        The SVD (subspace estimation) is non-differentiable and runs under
        ``torch.no_grad`` like the nearest-anchor assignment; gradients still flow
        through the build and apply w.r.t. ``Q, K, V`` because ``R`` and ``M_red`` enter
        the returned tensors as constants (the basis, not its derivative, is used).

        Args:
            alpha: Per-anchor softmax weights, shape (batch, m, seq_len).
            V: Value tensor, shape (batch, seq_len, dim).
            K: Key tensor, shape (batch, seq_len, dim).
            Q: Query tensor, shape (batch, seq_len, dim).
            anchors: Anchor tensor, shape (batch, m, dim).
            assign: Nearest-anchor assignment, shape (batch, seq_len).

        Returns:
            R: Per-anchor query-residual bases, shape (batch, m, dim, d′_max),
                zero-padded past each anchor's own d′_j (identity-padded for fallbacks).
            M_red: Reduced second-moment operators, shape (batch, m, dim, d′_max),
                zero-padded to d′_max.
            dprime: Per-anchor retained rank d′_j, shape (batch, m), integer.
        """
        b, m, n = alpha.shape
        dim = V.shape[-1]
        p = self.build_reduced_d
        device = V.device

        eye = torch.eye(dim, device=device, dtype=torch.float32)

        # First pass (fp32, no grad): per-anchor residual SVD → basis R_j and rank d′_j.
        # Collected as python lists of (dim, d′_j) tensors, then padded to d′_max.
        R_list = []  # length b*m, each (dim, d′_j)
        dprime = torch.empty(b, m, dtype=torch.long, device=device)
        with torch.no_grad():
            Qf = Q.float()
            Af = anchors.float()
            for bi in range(b):
                for j in range(m):
                    mask = assign[bi] == j
                    # Exclude padded queries from the residual set: a padded
                    # query's value must not shape the SVD basis (which is applied
                    # to valid queries sharing this anchor), or it would leak into
                    # their reduced-rank output.
                    if key_padding_mask is not None:
                        mask = mask & key_padding_mask[bi]
                    cnt = int(mask.sum().item())
                    fallback = cnt < self._MIN_CLUSTER
                    if not fallback:
                        dq = Qf[bi][mask] - Af[bi, j]  # (cnt, dim)
                        try:
                            svals = torch.linalg.svdvals(dq)  # (min(cnt, dim),)
                        except Exception:
                            fallback = True
                    if not fallback:
                        energy = svals.double() ** 2
                        total = energy.sum().clamp_min(1e-12)
                        cum = (torch.cumsum(energy, dim=0) / total).cpu()
                        dpr = int(
                            torch.searchsorted(
                                cum, torch.tensor(p, dtype=cum.dtype)
                            ).item()
                        ) + 1
                        dpr = min(dpr, int(svals.numel()))
                        # Self-guard: not meaningfully low-rank → exact dense fallback.
                        if dpr > 0.5 * dim:
                            fallback = True
                    if fallback:
                        dprime[bi, j] = dim
                        R_list.append(eye)  # identity basis → exact dense M_j
                        continue
                    # Right singular vectors of the residuals as basis columns.
                    _, _, Vh = torch.linalg.svd(dq, full_matrices=False)
                    dprime[bi, j] = dpr
                    R_list.append(Vh[:dpr].mH.contiguous())  # (dim, dpr)

        dprime_max = int(dprime.max().item())

        # Pad each basis to (dim, dprime_max) and stack to (b, m, dim, dprime_max).
        R = torch.zeros(b, m, dim, dprime_max, device=device, dtype=V.dtype)
        for idx, Rj in enumerate(R_list):
            bi, j = divmod(idx, m)
            R[bi, j, :, : Rj.shape[1]] = Rj.to(V.dtype)

        # Reduced build: project K into each anchor's subspace, then contract with
        # α and V. M_red[b,j] = Σ_n α_{jn} v_n (R_jᵀ k_n)ᵀ. Projecting first keeps the
        # contraction's inner dim at dprime_max ≪ dim.
        #   Kr[b,j,n,r] = Σ_e K[b,n,e] R[b,j,e,r]           (b, m, n, dprime_max)
        #   M_red[b,j,d,r] = Σ_n α[b,j,n] V[b,n,d] Kr[b,j,n,r]
        Kr = torch.einsum("bne,bmer->bmnr", K, R)  # (b, m, n, dprime_max)
        M_red = torch.einsum("bmn,bnd,bmnr->bmdr", alpha, V, Kr)  # (b, m, dim, dprime_max)
        return R, M_red, dprime

    def _compute_multi_anchor(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        anchors: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Piecewise-linear attention with per-query nearest-anchor expansion.

        Each query is linearized around its nearest anchor:
        ``out_i = A(a_{j(i)}) + J(a_{j(i)}) @ (q_i - a_{j(i)})`` where ``j(i)`` is
        the index of the anchor closest to query ``i``.

        The expansion distributes the matmul over the rank-1 Jacobian: for a
        query linearized around anchor ``j`` (``o_j`` = anchor output,
        ``wk_j`` = weighted keys, ``M_j = Σ_n α_{jn} v_n k_nᵀ``, ``Δq = q - a_j``):

            out = o_j + scale·(M_j @ Δq) − scale·o_j·(wk_jᵀ Δq),

        which is algebraically identical to ``o_j + J_j @ Δq``.

        The second-moment operator ``M`` is kept **compact** at ``(batch, m, dim,
        dim)`` and is *never* gathered to a per-query ``(batch, seq_len, dim,
        dim)`` tensor (that gather is a memory cliff — ≈2 GB at ``batch=64,
        seq_len=2048, dim=64`` — whose bandwidth, not FLOPs, dominates the apply).
        Instead queries are **grouped by their assigned anchor** and each group is
        applied with a single batched matmul against the compact ``M``: each
        query's ``Δq`` is scattered into a fixed-capacity bucket ``(batch, m, C,
        dim)`` (``C`` = the largest group size), one matmul computes ``M_j @ Δq``
        for every query at once, and the results are gathered back. The apply is
        then ``O(batch · seq_len · dim²)`` and never materializes a per-query
        ``(dim, dim)`` object. When the assignment is badly skewed (``C`` far
        larger than the balanced ``seq_len / m``, e.g. a degenerate single-cluster
        input) the bucket would itself grow toward ``(batch, m, seq_len, dim)``, so
        a guard falls back to a direct per-query matmul against the gathered
        compact ``M`` — correct and still free of the ``(d, d)`` gather, just
        without the batched-group speedup.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim).
            K: Key tensor of shape (batch, seq_len, dim).
            V: Value tensor of shape (batch, seq_len, dim).
            anchors: Anchor tensor of shape (batch, m, dim).

        Returns:
            output: Attention output of shape (batch, seq_len, dim).
        """
        b, n, dim = Q.shape
        m = anchors.shape[1]
        scale_factor = 1.0 / math.sqrt(dim) if self.scale else 1.0

        # Exact softmax attention output and the compact analytic-Jacobian
        # factors at each anchor. All are (batch, m, *) except the second-moment
        # operator M, which is only ever (batch, m, dim, dim) — no per-query
        # (dim, dim) tensor is ever formed.
        scores = torch.einsum("bmd,bnd->bmn", anchors, K) * scale_factor
        # Exclude padded keys before the softmax over the key axis (n). The mask
        # (b, n) broadcasts over the anchor axis m; masking scores here propagates
        # to out_anchor and every Jacobian factor (weighted_k / second-moment /
        # top-p / reduced-rank builds all consume alpha), so no separate build
        # masking is needed.
        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask.unsqueeze(1), -1e9)
        alpha = torch.softmax(scores, dim=-1)  # (batch, m, seq_len)
        # Zeroth-order term A(a_j): always built from ALL keys, so the anchor
        # value is exact even when the Jacobian factors are top-p truncated.
        out_anchor = torch.einsum("bmn,bnd->bmd", alpha, V)  # (batch, m, dim)

        # Hard-assign each query to its nearest anchor (non-differentiable);
        # gradients still flow through the expansion below w.r.t. Q, K, V.
        # cdist has no half-precision kernel on some backends, so the distance
        # is computed in float; the result is an integer index, so the upcast is
        # free and does not affect the fp16 expansion below. Computed here (before
        # the build) because the reduced-rank build needs the per-anchor query
        # membership to estimate each cluster's residual subspace.
        with torch.no_grad():
            assign = torch.cdist(Q.float(), anchors.float()).argmin(dim=-1)  # (batch, seq_len)

        # ``reduced`` flags the low-rank build path: the second-moment operator is
        # factored as ``M_red @ Rᵀ`` (shapes (b, m, dim, d′) each) rather than a full
        # (b, m, dim, dim) matrix, so the apply projects Δq through R before the matmul.
        reduced = self.build_reduced_d is not None
        R = None
        if reduced:
            # Reduced-rank build on the query-residual axis: M_j built only on the
            # d′_j directions each cluster's Δq excites. weighted_k stays exact.
            weighted_k = torch.einsum("bmn,bnd->bmd", alpha, K)  # (batch, m, dim)
            R, second_moment, _dprime = self._build_jacobian_factors_reduced(
                alpha, V, K, Q, anchors, assign, key_padding_mask=key_padding_mask
            )  # R: (b, m, dim, d′_max), second_moment (M_red): (b, m, dim, d′_max)
        elif self.build_topp is None:
            # Exact dense build of the Jacobian factors.
            weighted_k = torch.einsum("bmn,bnd->bmd", alpha, K)  # (batch, m, dim)
            # M_j = Σ_n α_{jn} v_n k_nᵀ, formed with α folded directly into the
            # contraction. The earlier "α ⊙ V" pre-scaling materialized a
            # (batch, m, seq_len, dim) tensor — an m× fp32 blow-up of V (≈2 GB at
            # batch=64, m=64, seq_len=2048, dim=64) written to and re-read from
            # memory purely to carry the weights into the matmul. Since the build
            # is memory-bandwidth-bound (not FLOP-bound), that intermediate, not
            # the arithmetic, dominates. The three-operand einsum contracts α, V,
            # K without ever forming it — bit-identical output, grads ~1e-6.
            second_moment = torch.einsum("bmn,bnd,bne->bmde", alpha, V, K)  # (b, m, d, d)
        else:
            # Top-p (nucleus) truncated build: use only the smallest set of
            # top-weighted keys per anchor whose cumulative softmax mass reaches
            # build_topp. Shrinks the build's sequence axis from n to t ≪ n when
            # α is peaked, and reverts to the full n (exact) when α is diffuse.
            weighted_k, second_moment = self._build_jacobian_factors_topp(alpha, V, K)

        # Gather the m-free per-query factors along the anchor axis (cheap
        # O(seq_len · dim) gathers, never a (dim, dim) per-query object).
        idx_d = assign.unsqueeze(-1).expand(-1, -1, dim)  # (batch, seq_len, dim)
        a_sel = torch.gather(anchors, 1, idx_d)  # (batch, seq_len, dim)
        o_sel = torch.gather(out_anchor, 1, idx_d)  # (batch, seq_len, dim)
        wk_sel = torch.gather(weighted_k, 1, idx_d)  # (batch, seq_len, dim)
        delta = Q - a_sel  # (batch, seq_len, dim)

        # Group-by-anchor apply of M_j @ Δq. Route each query's Δq into a
        # fixed-capacity bucket keyed by ``flat = assign · C + within-group-slot``
        # so all queries sharing an anchor are applied with ONE batched matmul
        # against the compact M.
        #
        # The within-group slot and the capacity ``C`` are derived from a stable
        # argsort of the assignment rather than a ``(batch, seq_len, m)`` one-hot
        # cumsum. The one-hot's memory grows with ``m`` (≈34 MB at batch=64,
        # seq_len=2048, m=64) — exactly the regime being optimized — whereas the
        # argsort touches only ``(batch, seq_len)`` tensors. Because a stable sort
        # preserves original sequence order within each anchor group, a query's
        # running index within its group is its position in the sorted order minus
        # the start of that group. Bit-identical slots and capacity to the one-hot
        # cumsum, ~2× faster and ~10× less memory at m=64.
        with torch.no_grad():
            order = torch.argsort(assign, stable=True, dim=1)  # (batch, seq_len)
            sorted_assign = torch.gather(assign, 1, order)
            ar = torch.arange(n, device=Q.device).expand(b, n)
            is_group_start = torch.ones(b, n, dtype=torch.bool, device=Q.device)
            is_group_start[:, 1:] = sorted_assign[:, 1:] != sorted_assign[:, :-1]
            # Start index (in sorted order) of the group each position belongs to.
            group_start = torch.cummax(
                torch.where(is_group_start, ar, torch.zeros_like(ar)), dim=1
            ).values
            within_group_sorted = ar - group_start  # running index within group
            capacity = int(within_group_sorted.max().item()) + 1  # C = largest group
            pos = torch.zeros(b, n, dtype=torch.long, device=Q.device)
            pos.scatter_(1, order, within_group_sorted.long())
            flat = (assign * capacity + pos).long()  # (batch, seq_len) into m·C

        # Skew guard: the grouped bucket has shape (batch, m·C, dim); its memory
        # and padding work grow with the largest group size C. Only when the
        # bucket would exceed the size of the direct-gather fallback's per-query
        # (batch, seq_len, dim, dim) operator — i.e. m·C > seq_len·dim, reached
        # only under a pathologically imbalanced assignment where C approaches
        # seq_len (e.g. a degenerate single-cluster input) — is it worth falling
        # back to a direct per-query matmul against the gathered compact M. That
        # fallback still avoids forming any per-query (dim, dim) object from
        # scratch; it only forgoes the batched-group speedup. For ordinary
        # (even 10×-imbalanced) k-means/stride assignments the bucket path wins,
        # so the guard stays inactive.
        if m * capacity > n * dim:
            if reduced:
                # Direct per-query apply against the compact reduced factors:
                # M_j·Δq ≈ M_red_j @ (R_jᵀ Δq). Gather both (dim, d′_max) factors
                # per query — still smaller than a (dim, dim) gather since d′_max ≤ dim.
                dpm = second_moment.shape[-1]
                idx_dr = assign.view(b, n, 1, 1).expand(-1, -1, dim, dpm)
                R_sel = torch.gather(R, 1, idx_dr)  # (b, n, dim, d′_max)
                mred_sel = torch.gather(second_moment, 1, idx_dr)  # (b, n, dim, d′_max)
                rq = torch.einsum("bner,bne->bnr", R_sel, delta)  # (b, n, d′_max)
                m_term = torch.einsum("bndr,bnr->bnd", mred_sel, rq)  # (b, n, dim)
            else:
                m_sel = torch.gather(
                    second_moment, 1, assign.view(b, n, 1, 1).expand(-1, -1, dim, dim)
                )  # (batch, seq_len, dim, dim)
                m_term = torch.einsum("bnde,bne->bnd", m_sel, delta)  # (batch, seq_len, dim)
        else:
            flat_d = flat.unsqueeze(-1).expand(-1, -1, dim)  # (batch, seq_len, dim)
            bucket = torch.zeros(b, m * capacity, dim, device=Q.device, dtype=Q.dtype)
            bucket.scatter_(1, flat_d, delta)  # place each Δq into its group slot
            if reduced:
                # Project each bucketed Δq into its anchor's subspace, then apply the
                # reduced operator: M_j·Δq ≈ M_red_j @ (R_jᵀ Δq). Projection happens
                # INSIDE the bucket (never per query), keeping the (dim, dim)-free
                # memory discipline. R and M_red share the batched d′_max width.
                bkt = bucket.view(b, m, capacity, dim)
                rq = torch.matmul(bkt, R)  # (b, m, C, d′_max): Rᵀ Δq per query
                res = torch.matmul(rq, second_moment.transpose(-1, -2))  # (b, m, C, dim)
                res = res.reshape(b, m * capacity, dim)
            else:
                # dq @ Mᵀ computes (M @ dq) row-wise, so transpose the last two dims.
                res = torch.matmul(bucket.view(b, m, capacity, dim), second_moment.transpose(-1, -2))
                res = res.reshape(b, m * capacity, dim)
            m_term = res.gather(1, flat_d)  # (batch, seq_len, dim): M_{assign_i} @ Δq_i

        # Rank-1 term o_j·(wk_jᵀ Δq) without forming any (dim, dim) product.
        wk_dot = torch.einsum("bnd,bnd->bn", wk_sel, delta)  # (batch, seq_len)
        return o_sel + scale_factor * (m_term - o_sel * wk_dot.unsqueeze(-1))

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute piecewise linear attention.

        Args:
            Q: Query tensor of shape (batch, seq_len, dim)
            K: Key tensor of shape (batch, seq_len, dim)
            V: Value tensor of shape (batch, seq_len, dim)
            key_padding_mask: Optional bool ``(batch, key_len)``, True=valid. The
                anchor weights are a softmax over the key axis (both single- and
                multi-anchor), so padded key scores are set to ``-inf`` before that
                softmax and drop out of the anchor output *and* every Jacobian
                factor (which reuse the same weights). Zeroing K/V rows would not
                work — a zeroed key still gets ``exp(0)=1`` in the softmax
                denominator. Non-causal only (the causal path ignores the mask).

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
                # anchor. The anchors double as the reported pseudo_queries. The
                # mask is passed to anchor selection so padded queries don't pull
                # the anchors (which would perturb valid queries' outputs).
                anchors = self.anchor_strategy.select(
                    Q, key_padding_mask=key_padding_mask
                )  # (batch, m, dim)
                output = self._compute_multi_anchor(
                    Q, K, V, anchors, key_padding_mask=key_padding_mask
                )
                pseudo_queries = anchors
            else:
                # Single-anchor mode (original implementation).
                # Step 1: Select representative query for each batch sample. The
                # pseudo-query must exclude padded queries — otherwise a padded
                # token shifts the shared anchor and leaks into every valid
                # query's Taylor expansion.
                if self.anchor_strategy is not None:
                    # A single-anchor strategy yields (batch, 1, dim); squeeze it
                    # back to the legacy (batch, dim) representative query.
                    pseudo_queries = self.anchor_strategy.select(
                        Q, key_padding_mask=key_padding_mask
                    ).squeeze(1)
                elif key_padding_mask is not None:
                    # Masked mean over valid queries for the default selector.
                    w = key_padding_mask.unsqueeze(-1).to(Q.dtype)
                    pseudo_queries = (Q * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
                else:
                    pseudo_queries = self.pseudo_query_fn(Q)  # (batch, dim)

                # Step 2: Compute exact attention and Jacobian at pseudo-queries
                pseudo_outputs, jacobians = self._compute_attention_and_jacobian(
                    pseudo_queries, K, V, key_padding_mask=key_padding_mask
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


class NystromformerAttention(BaseAttention):
    """Nyströmformer attention via Nyström approximation of softmax self-attention.

    Approximates the full softmax attention matrix in linear time using the
    Nyström method (Xiong et al., 2021, "Nyströmformer: A Nyström-Based
    Algorithm for Approximating Self-Attention"). Rather than materializing the
    dense ``(n, n)`` softmax matrix, ``m`` *landmark* points are chosen as the
    segment means of the queries and keys, and the attention output is
    reconstructed from three small softmax kernels

        kernel1 = softmax(Q       @ K_land^T)   # (b, n, m)
        kernel2 = softmax(Q_land  @ K_land^T)   # (b, m, m)
        kernel3 = softmax(Q_land  @ K^T)        # (b, m, n)

    as ``output = kernel1 @ pinv(kernel2) @ (kernel3 @ V)``. All three softmax
    scores are scaled by ``1 / sqrt(dim)``, matching standard scaled
    dot-product attention. The middle factor ``pinv(kernel2)`` is the
    Moore–Penrose pseudo-inverse of the ``(m, m)`` landmark-to-landmark kernel,
    computed with the cubically-convergent iterative scheme used by the
    reference implementation — no explicit matrix inverse or SVD is formed, so
    the whole operator is differentiable and runs on-device.

    Complexity: O(batch * n * m * d) with a small constant ``m``, versus the
    O(batch * n^2 * d) of full softmax attention.

    Landmark selection uses contiguous **segment means**: the ``n`` tokens are
    split into ``m`` equal contiguous segments and averaged within each. This
    mixes information across the whole sequence, including future tokens, so
    Nyströmformer has **no exact causal formulation** — a causal landmark would
    need per-position segment means, which the segment-mean construction does
    not provide. Constructing this module with ``causal=True`` therefore raises
    ``NotImplementedError``.

    When ``seq_len <= num_landmarks`` the landmark approximation cannot help and
    the pseudo-inverse becomes ill-posed, so the module falls back to *exact*
    softmax attention for that call and records ``num_landmarks = seq_len`` in
    the diagnostics. The segment-mean construction here requires ``n`` to be
    divisible by ``m``; when it is not, the effective landmark count is reduced
    to the largest divisor of ``n`` that does not exceed ``num_landmarks`` (this
    keeps the reshape exact rather than padding the sequence). A prime ``n``
    collapses to a single global landmark; a warning is emitted when the
    effective count drops far below the requested value.

    This implementation is deliberately **parameter-free**: unlike the full
    paper, it omits the learned depthwise-convolution skip connection so that it
    matches the repository's parameter-free-attention baselines. Only the
    Nyström approximation core is modelled.

    The second element returned by :meth:`forward` is a per-batch conditioning
    diagnostic: the Frobenius residual ``||A @ Z @ A - A||_F`` of the iterative
    pseudo-inverse ``Z`` of ``A = kernel2``. This residual measures how well the
    fixed-iteration scheme inverted the landmark kernel: it is near zero when
    the kernel is well-conditioned, but the iteration can fail to converge when
    the landmark kernel becomes near-singular (few, highly-similar landmarks),
    in which case the residual — and the resulting attention — degrade. Recording
    this fragility is a deliberate differentiator of the method. The residual is
    cached on ``self.last_pinv_residual`` together with the effective landmark
    count ``self.last_num_landmarks`` for later inspection.

    Parameters
    ----------
    dim:
        Per-head embedding dimension.
    dropout:
        Dropout probability applied to the output.
    num_landmarks:
        Requested number of Nyström landmarks ``m``. Reduced to a divisor of
        the sequence length when ``n`` is not divisible by ``m``.
    pinv_iterations:
        Number of iterations of the cubically-convergent Moore–Penrose
        pseudo-inverse scheme applied to the landmark kernel.
    eps:
        Numerical-stability constant used when initializing the iteration.
    causal:
        Must be ``False``. Passing ``True`` raises ``NotImplementedError``
        because segment-mean landmarks mix future tokens.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        num_landmarks: int = 64,
        pinv_iterations: int = 6,
        eps: float = 1e-8,
        causal: bool = False,
    ):
        super().__init__(dim, dropout)
        if causal:
            raise NotImplementedError(
                "NystromformerAttention has no exact causal formulation: landmarks "
                "are segment means over contiguous token blocks, so every landmark "
                "mixes future tokens. Use StandardAttention/PiecewiseAttention "
                "(causal=True) for autoregressive settings."
            )
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.eps = eps
        self.causal = causal

        # Diagnostics populated on each forward pass.
        self.last_pinv_residual: Optional[torch.Tensor] = None
        self.last_num_landmarks: Optional[int] = None

    def _iterative_pinv(self, A: torch.Tensor) -> torch.Tensor:
        """Moore–Penrose pseudo-inverse via a cubically-convergent iteration.

        Starting from ``Z_0 = A^T / (max_row_abs_sum(A) * max_col_abs_sum(A))``
        the update

            Z <- 0.25 * Z @ (13 I - A @ Z @ (15 I - A @ Z @ (7 I - A @ Z)))

        is applied ``pinv_iterations`` times, converging to ``A^+`` at a cubic
        rate without forming an explicit inverse or SVD.

        Parameters
        ----------
        A:
            Batched square matrices of shape ``(batch, m, m)``.

        Returns
        -------
        torch.Tensor
            Approximate pseudo-inverse of shape ``(batch, m, m)``.
        """
        m = A.shape[-1]
        identity = torch.eye(m, device=A.device, dtype=A.dtype).expand_as(A)

        # Row-sum / column-sum norm bound guarantees the iteration converges.
        max_abs_row_sum = torch.max(torch.sum(torch.abs(A), dim=-1), dim=-1).values
        max_abs_col_sum = torch.max(torch.sum(torch.abs(A), dim=-2), dim=-1).values
        denom = (max_abs_row_sum * max_abs_col_sum).clamp_min(self.eps)
        Z = A.transpose(-2, -1) / denom.unsqueeze(-1).unsqueeze(-1)

        for _ in range(self.pinv_iterations):
            AZ = torch.matmul(A, Z)
            Z = torch.matmul(
                0.25 * Z,
                13 * identity
                - torch.matmul(
                    AZ,
                    15 * identity - torch.matmul(AZ, 7 * identity - AZ),
                ),
            )
        return Z

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute Nyströmformer attention.

        Parameters
        ----------
        Q, K, V:
            Query/key/value tensors of shape ``(batch, seq_len, dim)``.
        key_padding_mask:
            Optional bool ``(batch, key_len)``, True=valid. Nyström has two leaks
            for padded keys, both handled here: (1) ``kernel3`` softmaxes over the
            key axis, so padded columns are set to ``-inf`` before that softmax;
            (2) landmarks are contiguous segment *means* over the sequence, so a
            padded position would dilute its segment's landmark toward zero and
            corrupt every query's output globally — landmarks are therefore built
            with a per-segment *masked* mean (dividing by the valid count, with a
            guard for all-padding segments). The exact-softmax fallback path is
            masked identically to ``StandardAttention``. Non-causal only.

        Returns
        -------
        output:
            Attention output of shape ``(batch, seq_len, dim)``.
        pinv_residual:
            Per-batch pseudo-inverse Frobenius residual
            ``||A @ Z @ A - A||_F`` of shape ``(batch,)`` measuring the
            conditioning of the landmark kernel; ``None`` on the exact-softmax
            fallback path.
        """
        batch, seq_len, dim = Q.shape
        scale = 1.0 / math.sqrt(dim)
        sqrt_scale = math.sqrt(scale)

        # Pre-scale Q and K by sqrt(scale) each, so that any Q_s @ K_s^T below
        # already carries the full 1/sqrt(dim) factor of scaled dot-product
        # scores.
        Q_s = Q * sqrt_scale
        K_s = K * sqrt_scale

        m = self.num_landmarks

        # Fallback: with as many (or more) landmarks than tokens the Nyström
        # approximation cannot help and kernel2 is ill-posed, so compute exact
        # softmax attention instead.
        if seq_len <= m:
            scores = torch.matmul(Q_s, K_s.transpose(-2, -1))  # (b, n, n)
            if key_padding_mask is not None:
                scores = scores.masked_fill(~key_padding_mask.unsqueeze(1), -1e9)
            attn = F.softmax(scores, dim=-1)
            output = self.dropout(torch.matmul(attn, V))
            self.last_num_landmarks = seq_len
            self.last_pinv_residual = None
            return output, None

        # The reshape-based segment mean requires n % m == 0. If it does not
        # divide, fall back to the largest divisor of n not exceeding m so the
        # reshape stays exact (no sequence padding). A prime n collapses to 1.
        if seq_len % m != 0:
            m = max(d for d in range(1, m + 1) if seq_len % d == 0)
            if m < self.num_landmarks // 2:
                warnings.warn(
                    f"Nyströmformer reduced landmarks from {self.num_landmarks} to "
                    f"{m} because seq_len={seq_len} is not divisible by "
                    f"{self.num_landmarks}; the approximation degrades. Choose a "
                    f"seq_len divisible by the requested landmark count.",
                    stacklevel=2,
                )

        segment = seq_len // m

        # Landmarks as contiguous segment means: reshape (b, m, segment, d) and
        # average within each segment. With a key-padding mask, use a masked mean
        # so padded positions do not dilute a segment's landmark (a plain mean over
        # zeroed padded rows would bias every landmark, corrupting all queries).
        if key_padding_mask is not None:
            seg_mask = key_padding_mask.reshape(batch, m, segment, 1).to(Q_s.dtype)
            seg_count = seg_mask.sum(dim=2).clamp(min=1.0)  # (b, m, 1), >=1 guards empty segs
            Q_landmarks = (Q_s.reshape(batch, m, segment, dim) * seg_mask).sum(dim=2) / seg_count
            K_landmarks = (K_s.reshape(batch, m, segment, dim) * seg_mask).sum(dim=2) / seg_count
        else:
            Q_landmarks = Q_s.reshape(batch, m, segment, dim).mean(dim=2)  # (b, m, d)
            K_landmarks = K_s.reshape(batch, m, segment, dim).mean(dim=2)  # (b, m, d)

        # Three softmax kernels (scores already carry the 1/sqrt(dim) scale).
        kernel1 = F.softmax(torch.matmul(Q_s, K_landmarks.transpose(-2, -1)), dim=-1)  # (b, n, m)
        kernel2 = F.softmax(
            torch.matmul(Q_landmarks, K_landmarks.transpose(-2, -1)), dim=-1
        )  # (b, m, m)
        # kernel3 softmaxes over the key axis n, so padded key columns must be
        # excluded before the softmax or they steal normalized mass from valid keys.
        kernel3_scores = torch.matmul(Q_landmarks, K_s.transpose(-2, -1))  # (b, m, n)
        if key_padding_mask is not None:
            kernel3_scores = kernel3_scores.masked_fill(
                ~key_padding_mask.unsqueeze(1), -1e9
            )
        kernel3 = F.softmax(kernel3_scores, dim=-1)  # (b, m, n)

        # Iterative pseudo-inverse of the landmark-to-landmark kernel.
        kernel2_pinv = self._iterative_pinv(kernel2)  # (b, m, m)

        # Conditioning diagnostic: Frobenius residual ||A Z A - A||_F per batch.
        # Near zero for well-conditioned kernels; diverges as m shrinks.
        residual = torch.matmul(torch.matmul(kernel2, kernel2_pinv), kernel2) - kernel2
        pinv_residual = torch.linalg.norm(residual.reshape(batch, -1), dim=-1)  # (b,)

        # output = kernel1 @ pinv(kernel2) @ (kernel3 @ V)
        kernel3_v = torch.matmul(kernel3, V)  # (b, m, d)
        output = torch.matmul(kernel1, torch.matmul(kernel2_pinv, kernel3_v))  # (b, n, d)
        output = self.dropout(output)

        # Record diagnostics only once the successful path has fully computed.
        self.last_num_landmarks = m
        self.last_pinv_residual = pinv_residual.detach()

        return output, pinv_residual


class LinformerAttention(BaseAttention):
    """Linformer attention with low-rank projection of the sequence axis.

    Approximates full softmax attention in linear time by projecting the key
    and value sequences from length ``n`` down to a fixed rank ``k`` along the
    sequence axis (Wang et al., 2020, "Linformer: Self-Attention with Linear
    Complexity"). Two learned projection matrices ``E`` and ``F`` compress the
    keys and values,

        K_proj = E_n^T @ K   (shape (batch, k, dim)),
        V_proj = F_n^T @ V   (shape (batch, k, dim)),

    after which attention is computed against the compressed memory,

        Attention(Q, K, V) = softmax(Q @ K_proj^T / sqrt(d)) @ V_proj.

    Because the score matrix is ``(n, k)`` rather than ``(n, n)``, the cost is
    linear in the sequence length: O(batch * n * k * d).

    Same-length-only baseline
    --------------------------
    The projections ``E`` and ``F`` are defined over a fixed maximum sequence
    length ``max_seq_len`` and act directly on absolute sequence positions. A
    sequence of length ``n`` uses the first ``n`` rows, ``E[:n]`` and ``F[:n]``.
    This makes Linformer a SAME-LENGTH-ONLY baseline: it CANNOT generalize to
    sequences longer than ``max_seq_len`` (there are simply no projection rows
    for those positions), and because the learned projection is tied to
    absolute positions, its compression is specialized to the training-time
    layout rather than being length-agnostic.

    Parameter cost
    --------------
    Unlike softmax attention or Performer (whose random features are drawn from
    a fixed seed and stored in a non-persistent buffer), Linformer is NOT
    parameter-free: it adds ``2 * max_seq_len * k`` learned parameters (the two
    projection matrices ``E`` and ``F``).

    Parameters
    ----------
    dim:
        Per-head embedding dimension.
    dropout:
        Dropout probability applied to the output.
    max_seq_len:
        Fixed maximum sequence length the projections are defined over. Inputs
        with ``seq_len > max_seq_len`` are rejected.
    k:
        Low-rank projection dimension along the sequence axis. The paper uses
        values in ``{128, 256}``; larger ``k`` trades compute for a tighter
        approximation of full attention.
    causal:
        Must be False. The sequence-axis projection mixes information across the
        whole sequence, so there is no valid causal (autoregressive) form;
        passing True raises ``NotImplementedError``.

    Notes
    -----
    With random initialization the low-rank approximation is not expected to
    match full softmax attention numerically, even at large ``k`` — the
    projections must be trained. The meaningful init-time sanity check is
    therefore that the output has the correct shape and is finite, not that it
    reproduces softmax within a tolerance.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        k: int = 256,
        causal: bool = False,
    ):
        super().__init__(dim, dropout)
        if causal:
            raise NotImplementedError(
                "LinformerAttention has no causal form: the sequence-axis "
                "projection compresses the entire key/value sequence, so a "
                "query cannot be restricted to attend only to earlier positions."
            )
        self.max_seq_len = max_seq_len
        self.k = k
        self.causal = causal

        # Learned projections over the SEQUENCE axis: (max_seq_len, k). These
        # compress K and V from length n (using the first n rows) down to k.
        # They add 2 * max_seq_len * k learned parameters — Linformer is not
        # parameter-free.
        self.E = nn.Parameter(torch.empty(max_seq_len, k))
        self.F = nn.Parameter(torch.empty(max_seq_len, k))
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.F)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Linformer low-rank attention.

        Parameters
        ----------
        Q, K, V:
            Query/key/value tensors of shape ``(batch, seq_len, dim)``.
        key_padding_mask:
            Optional bool ``(batch, key_len)``, True=valid. Applied by zeroing
            padded rows of K and V *before* the E/F sequence-axis projection: the
            einsum multiplies each position by a projection weight, so a zeroed row
            contributes exactly nothing to any of the ``k`` compressed slots. This
            is the one method where row-zeroing is exact (there is no per-key
            softmax and no feature map). Masking after projection would be
            meaningless, as the ``k`` axis no longer corresponds to positions.

        Returns
        -------
        output:
            Attention output of shape ``(batch, seq_len, dim)``.
        attn:
            Low-rank attention weights of shape ``(batch, seq_len, k)``.

        Raises
        ------
        ValueError
            If ``seq_len`` exceeds ``max_seq_len``; the fixed projection is
            defined only for positions ``0..max_seq_len - 1``.
        """
        _, seq_len, _ = Q.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"LinformerAttention received seq_len={seq_len} but is limited "
                f"to max_seq_len={self.max_seq_len}: the learned sequence-axis "
                f"projection has no rows for positions beyond max_seq_len, so "
                f"this baseline is same-length-only."
            )

        # Exclude padded keys/values by zeroing their rows before the sequence-axis
        # projection. Exact here because the E/F einsum is linear in each position:
        # a zeroed row drops out of every compressed slot (no per-key softmax, no
        # feature map that would leave φ(0) ≠ 0).
        if key_padding_mask is not None:
            kv_mask = key_padding_mask.unsqueeze(-1).to(K.dtype)
            K = K * kv_mask
            V = V * kv_mask

        # Use the first n rows of each projection: (n, k).
        E_n = self.E[:seq_len]
        F_n = self.F[:seq_len]

        # Project K and V along the sequence (n) axis: (b, n, d) -> (b, k, d).
        K_proj = torch.einsum("nk,bnd->bkd", E_n, K)
        V_proj = torch.einsum("nk,bnd->bkd", F_n, V)

        # Scaled dot-product against the compressed memory: (b, n, k).
        scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / math.sqrt(self.dim)
        attn = F.softmax(scores, dim=-1)

        # Weighted sum over the k compressed value slots: (b, n, d).
        output = torch.matmul(attn, V_proj)
        output = self.dropout(output)

        return output, attn


class LunaAttention(BaseAttention):
    """Luna linear unified nested attention.

    Implements the two-step "pack and unpack" nested attention of Luna
    (Ma et al., 2021, "Luna: Linear Unified Nested Attention"). A fixed
    number ``p`` of learned *pack* queries first summarize the input
    sequence into a compact length-``p`` representation, and the real
    queries then *unpack* that summary. Both steps are ordinary scaled
    dot-product softmax attentions, but because the summary has fixed
    length ``p`` neither step ever forms an ``n x n`` score matrix:

        Step 1 (pack):   Yp = softmax(P @ K^T / sqrt(d)) @ V
        Step 2 (unpack): out = softmax(Q @ Yp^T / sqrt(d)) @ Yp

    Here ``P`` (shape ``(p, dim)``) is a learned sequence of pack queries
    that is broadcast over the batch. Step 1 produces the packed context
    ``Yp`` of shape ``(batch, p, dim)`` by having each of the ``p`` pack
    queries attend over the whole sequence (softmax over the ``n`` keys).
    Step 2 lets each of the ``n`` real queries attend over the ``p``
    summary vectors (softmax over ``p``), using ``Yp`` as both keys and
    values. This nesting is what makes Luna a landmark-style method: the
    ``p`` pack queries act as ``p`` learned landmarks, directly comparable
    to the Nyströmformer landmarks or the Linformer low-rank projections.

    Unlike the parameter-free softmax baselines, Luna adds ``p * dim``
    learned parameters (the pack-query sequence ``P``). The pack queries
    are stored as an ``nn.Parameter`` and initialized from a normal
    distribution with standard deviation ``dim ** -0.5``.

    Complexity: O(batch * n * p * d), i.e. linear in the sequence length
    ``n`` for a fixed number of pack queries ``p`` (never O(n^2)).
    Memory: O(batch * n * p).

    Parameters
    ----------
    dim:
        Per-head embedding dimension ``d``.
    dropout:
        Dropout probability applied to the output.
    num_pack:
        Number of learned pack queries ``p`` (the length of the packed
        summary and, equivalently, the number of landmarks). Defaults to
        256.
    causal:
        If True, raise ``NotImplementedError``. Luna has no simple exact
        causal form for the pack step: each pack query attends over the
        entire sequence, so the packed summary ``Yp`` mixes information
        from all positions and cannot be masked per query position without
        changing the algorithm. A causal variant is intentionally not
        provided here.

    Notes
    -----
    The module is *not* parameter-free: it introduces ``p * dim`` learned
    parameters via the pack-query sequence ``P``.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        num_pack: int = 256,
        causal: bool = False,
    ):
        super().__init__(dim, dropout)
        if causal:
            raise NotImplementedError(
                "LunaAttention does not support causal masking: each pack query "
                "attends over the whole sequence, so the packed summary mixes all "
                "positions and cannot be causally masked per query position without "
                "changing the algorithm. Use a non-causal configuration."
            )
        self.num_pack = num_pack
        self.causal = causal

        # Learned pack-query sequence P of shape (p, dim). This is Luna's
        # extra input sequence and the sole source of the module's
        # p * dim added parameters. Broadcast over the batch at forward time.
        pack_queries = torch.empty(num_pack, dim)
        nn.init.normal_(pack_queries, std=dim**-0.5)
        self.pack_queries = nn.Parameter(pack_queries)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Luna nested (pack -> summary -> unpack) attention.

        Parameters
        ----------
        Q, K, V:
            Query/key/value tensors of shape ``(batch, seq_len, dim)``.
        key_padding_mask:
            Optional bool ``(batch, key_len)``, True=valid. Applied only to the
            Step-1 *pack* softmax (``scores1``, over the n keys), where padded
            columns are set to ``-inf`` so they leave the packed summary ``Yp``.
            Step 2 softmaxes over the ``p`` summary vectors, which are already
            clean once ``Yp`` is, so it needs no masking. Zeroing K/V rows would
            not work: the pack softmax over keys still gives a zeroed column
            ``exp(0)=1`` weight, polluting ``Yp`` and hence every query output.

        Returns
        -------
        output:
            Attention output of shape ``(batch, seq_len, dim)``.
        packed_context:
            The length-``p`` packed summary ``Yp`` of shape
            ``(batch, num_pack, dim)``, returned as a diagnostic.
        """
        batch, _, _ = Q.shape
        scale = 1.0 / math.sqrt(self.dim)

        # Broadcast the learned pack queries over the batch: (batch, p, dim).
        P = self.pack_queries.unsqueeze(0).expand(batch, -1, -1)

        # Step 1 (pack): the p pack queries attend over the whole sequence.
        # scores1: (batch, p, n), softmax over the n keys.
        scores1 = torch.matmul(P, K.transpose(-2, -1)) * scale
        # Exclude padded key columns before the key-axis softmax so they leave
        # the packed summary Yp entirely.
        if key_padding_mask is not None:
            scores1 = scores1.masked_fill(~key_padding_mask.unsqueeze(1), -1e9)
        attn1 = F.softmax(scores1, dim=-1)
        packed_context = torch.matmul(attn1, V)  # (batch, p, dim) == Yp

        # Step 2 (unpack): the real queries attend over the p-vector summary,
        # using Yp as both keys and values. scores2: (batch, n, p), softmax
        # over the p summary vectors.
        scores2 = torch.matmul(Q, packed_context.transpose(-2, -1)) * scale
        attn2 = F.softmax(scores2, dim=-1)
        output = torch.matmul(attn2, packed_context)  # (batch, n, dim)

        output = self.dropout(output)

        return output, packed_context


# Convenient alias
EfficientAttention = PiecewiseAttention


__all__ = [
    "BaseAttention",
    "StandardAttention",
    "LinearAttention",
    "PerformerAttention",
    "NystromformerAttention",
    "LinformerAttention",
    "LunaAttention",
    "PiecewiseAttention",
    "EfficientAttention",
]
