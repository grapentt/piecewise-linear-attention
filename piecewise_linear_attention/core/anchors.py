"""Anchor-selection strategies for piecewise-linear attention.

Piecewise-linear (anchor-Taylor) attention expands the softmax attention map to
first order around one or more *anchor* queries. An :class:`AnchorStrategy`
decides where those anchors sit for a given batch of queries.

The single-anchor case (``num_anchors == 1``) recovers the original method,
where one representative query per sample (e.g. the mean) is used. Multiple
anchors partition the query space so each query is linearized around a nearby
anchor, which is what lets the approximation track content-based routing over
long sequences.

Selection that involves discrete operations (nearest-anchor assignment, Lloyd
updates) is non-differentiable and therefore runs under ``torch.no_grad()``.
This does not stop gradients from flowing through the attention output: the
downstream ``A(a) + J(a) @ (q - a)`` expression is differentiable in ``Q, K, V``
regardless of how the anchors ``a`` were chosen.
"""

from abc import ABC, abstractmethod
from typing import Callable

import torch


class AnchorStrategy(ABC):
    """Base class for anchor-selection strategies.

    A strategy maps a batch of queries to a batch of anchors.

    Attributes
    ----------
    num_anchors:
        Number of anchors produced per sample. ``1`` selects the original
        single-anchor behavior.
    """

    num_anchors: int = 1

    @abstractmethod
    def select(self, Q: torch.Tensor) -> torch.Tensor:
        """Select anchors for the given queries.

        Parameters
        ----------
        Q:
            Query tensor of shape ``(batch, seq_len, dim)``.

        Returns
        -------
        torch.Tensor
            Anchors of shape ``(batch, num_anchors, dim)``.
        """
        raise NotImplementedError


class MeanAnchor(AnchorStrategy):
    """Single anchor at the mean of each sample's queries.

    This reproduces the default behavior of the original piecewise attention.
    """

    num_anchors = 1

    def select(self, Q: torch.Tensor) -> torch.Tensor:
        return Q.mean(dim=1, keepdim=True)  # (batch, 1, dim)


class FirstAnchor(AnchorStrategy):
    """Single anchor at the first query of each sample.

    Useful for autoregressive settings where the first position is a stable
    reference.
    """

    num_anchors = 1

    def select(self, Q: torch.Tensor) -> torch.Tensor:
        return Q[:, :1, :]  # (batch, 1, dim)


class StrideAnchor(AnchorStrategy):
    """Deterministic multi-anchor: ``k`` queries evenly spaced by position.

    Anchors are the queries at ``k`` indices linearly spaced across the
    sequence. This is parameter-free and deterministic, making it a cheap and
    reproducible multi-anchor baseline.

    Parameters
    ----------
    k:
        Number of anchors.
    """

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.num_anchors = k

    def select(self, Q: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = Q.shape
        k = min(self.num_anchors, seq_len)
        # Evenly spaced positions across the sequence (endpoints included).
        idx = torch.linspace(0, seq_len - 1, k, device=Q.device).round().long()
        return Q[:, idx, :]  # (batch, k, dim)


class KMeansAnchor(AnchorStrategy):
    """Multi-anchor via lightweight k-means (Lloyd iterations) on the queries.

    Anchors are cluster centroids of each sample's queries. Clustering is a
    discrete, non-differentiable operation and runs under ``torch.no_grad()``;
    the returned centroids are detached from the autograd graph.

    Parameters
    ----------
    k:
        Number of clusters/anchors.
    iters:
        Number of Lloyd iterations.
    """

    def __init__(self, k: int, iters: int = 3):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if iters < 0:
            raise ValueError(f"iters must be >= 0, got {iters}")
        self.num_anchors = k
        self.iters = iters

    def select(self, Q: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = Q.shape
        k = min(self.num_anchors, seq_len)
        with torch.no_grad():
            # Lloyd's algorithm is run in float32: cdist has no half-precision
            # kernel on some backends and scatter/division accumulate error in
            # fp16. The selected anchors are cast back to the input dtype.
            Qf = Q.float()
            # Deterministic init: evenly spaced queries as initial centroids.
            idx = torch.linspace(0, seq_len - 1, k, device=Q.device).round().long()
            centroids = Qf[:, idx, :].clone()  # (batch, k, dim)

            for _ in range(self.iters):
                # Assign each query to its nearest centroid.
                dists = torch.cdist(Qf, centroids)  # (batch, seq_len, k)
                assign = dists.argmin(dim=-1)  # (batch, seq_len)

                # Recompute centroids as the mean of assigned queries.
                new_centroids = torch.zeros_like(centroids)
                counts = torch.zeros(batch, k, device=Q.device)
                new_centroids.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, dim), Qf)
                counts.scatter_add_(1, assign, torch.ones(batch, seq_len, device=Q.device))
                # Empty clusters keep their previous centroid (avoid div-by-zero).
                nonempty = counts > 0
                counts = counts.clamp(min=1).unsqueeze(-1)
                averaged = new_centroids / counts
                centroids = torch.where(nonempty.unsqueeze(-1), averaged, centroids)

        return centroids.to(Q.dtype).detach()  # (batch, k, dim)


class CallableAnchor(AnchorStrategy):
    """Adapter wrapping a legacy ``pseudo_query_fn`` as a single-anchor strategy.

    The legacy signature is ``fn(Q: (batch, seq_len, dim)) -> (batch, dim)``.
    This adapter reshapes the result to ``(batch, 1, dim)`` so it fits the
    strategy interface without changing user code.

    Parameters
    ----------
    fn:
        Callable mapping queries to one representative query per sample.
    """

    num_anchors = 1

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        self.fn = fn

    def select(self, Q: torch.Tensor) -> torch.Tensor:
        anchor = self.fn(Q)  # (batch, dim)
        if anchor.dim() != 2:
            raise ValueError(
                "pseudo_query_fn must return shape (batch, dim); " f"got {tuple(anchor.shape)}"
            )
        return anchor.unsqueeze(1)  # (batch, 1, dim)


__all__ = [
    "AnchorStrategy",
    "MeanAnchor",
    "FirstAnchor",
    "StrideAnchor",
    "KMeansAnchor",
    "CallableAnchor",
]
