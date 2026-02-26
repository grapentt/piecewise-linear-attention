"""Pseudo-query initialization and management.

This module provides utilities for initializing and managing pseudo-queries
used in piecewise linear attention.
"""

from typing import Optional

import torch
import torch.nn as nn


def initialize_pseudo_queries_kmeans(
    queries: torch.Tensor,
    num_pseudo_queries: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Initialize pseudo-queries using K-means clustering on actual queries.

    This function runs K-means clustering on a collection of query vectors
    to find k representative cluster centers. These centers serve as
    pseudo-queries for piecewise linear attention approximation.

    Args:
        queries: Query tensor of shape (num_samples, dim) or (batch, seq_len, dim).
                 Will be flattened to (num_samples, dim) if 3D.
        num_pseudo_queries: Number of clusters (k) to create
        max_iter: Maximum number of K-means iterations
        tol: Convergence tolerance (stop if centroid movement < tol)
        seed: Random seed for reproducibility

    Returns:
        pseudo_queries: Tensor of shape (num_pseudo_queries, dim) containing
                       the cluster centers

    Example:
        >>> Q = torch.randn(32, 128, 64)  # batch, seq_len, dim
        >>> pseudo_q = initialize_pseudo_queries_kmeans(Q, num_pseudo_queries=16)
        >>> pseudo_q.shape
        torch.Size([16, 64])
    """
    # Flatten queries if 3D (batch, seq_len, dim) -> (batch*seq_len, dim)
    if queries.ndim == 3:
        batch, seq_len, dim = queries.shape
        queries_flat = queries.reshape(-1, dim)
    elif queries.ndim == 2:
        queries_flat = queries
        dim = queries.shape[1]
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {queries.ndim}D")

    num_samples, dim = queries_flat.shape

    if num_pseudo_queries > num_samples:
        raise ValueError(
            f"num_pseudo_queries ({num_pseudo_queries}) cannot exceed "
            f"number of samples ({num_samples})"
        )

    device = queries.device
    dtype = queries.dtype

    # Set random seed if provided
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    # Initialize centroids using k-means++ algorithm
    centroids = _kmeans_plusplus_init(
        queries_flat, num_pseudo_queries, generator=generator
    )

    # Run K-means iterations
    for iteration in range(max_iter):
        # Assign each query to nearest centroid
        # Shape: (num_samples, num_pseudo_queries)
        distances = torch.cdist(queries_flat, centroids)
        # Shape: (num_samples,)
        assignments = distances.argmin(dim=1)

        # Compute new centroids as mean of assigned points
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_pseudo_queries):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = queries_flat[mask].mean(dim=0)
            else:
                # If cluster is empty, reinitialize with random point
                idx = torch.randint(0, num_samples, (1,), generator=generator, device=device)
                new_centroids[k] = queries_flat[idx[0]]

        # Check convergence
        centroid_shift = torch.norm(new_centroids - centroids, dim=1).max()
        centroids = new_centroids

        if centroid_shift < tol:
            break

    return centroids


def _kmeans_plusplus_init(
    data: torch.Tensor,
    k: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Initialize centroids using k-means++ algorithm.

    K-means++ initialization chooses centroids that are far apart from each other,
    leading to faster convergence and better clustering quality.

    Args:
        data: Data points of shape (n, d)
        k: Number of centroids to initialize
        generator: Random number generator for reproducibility

    Returns:
        centroids: Initial centroids of shape (k, d)
    """
    n, d = data.shape
    device = data.device

    # Choose first centroid uniformly at random
    idx = torch.randint(0, n, (1,), generator=generator, device=device)
    centroids = [data[idx[0]]]

    # Choose remaining k-1 centroids
    for _ in range(k - 1):
        # Compute distance from each point to nearest existing centroid
        current_centroids = torch.stack(centroids)
        distances = torch.cdist(data, current_centroids)
        min_distances = distances.min(dim=1)[0]

        # Square the distances for probability weighting
        probabilities = min_distances**2
        probabilities /= probabilities.sum()

        # Sample next centroid with probability proportional to squared distance
        idx = torch.multinomial(probabilities, 1, generator=generator)
        centroids.append(data[idx[0]])

    return torch.stack(centroids)


def initialize_pseudo_queries_random(
    dim: int,
    num_pseudo_queries: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Initialize pseudo-queries randomly from standard normal distribution.

    This is a simple baseline initialization method. K-means initialization
    is generally preferred as it adapts to the actual query distribution.

    Args:
        dim: Dimension of queries
        num_pseudo_queries: Number of pseudo-queries to create
        device: Device to create tensor on
        dtype: Data type of tensor
        seed: Random seed for reproducibility

    Returns:
        pseudo_queries: Random pseudo-queries of shape (num_pseudo_queries, dim)
    """
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        pseudo_queries = torch.randn(
            num_pseudo_queries, dim, device=device, dtype=dtype, generator=generator
        )
    else:
        pseudo_queries = torch.randn(
            num_pseudo_queries, dim, device=device, dtype=dtype
        )

    return pseudo_queries


def initialize_pseudo_queries_uniform(
    queries: torch.Tensor,
    num_pseudo_queries: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Initialize pseudo-queries uniformly sampled from actual queries.

    Simply selects k random queries from the input to serve as pseudo-queries.

    Args:
        queries: Query tensor of shape (num_samples, dim) or (batch, seq_len, dim)
        num_pseudo_queries: Number of pseudo-queries to sample
        seed: Random seed for reproducibility

    Returns:
        pseudo_queries: Sampled pseudo-queries of shape (num_pseudo_queries, dim)
    """
    # Flatten queries if 3D
    if queries.ndim == 3:
        queries_flat = queries.reshape(-1, queries.shape[-1])
    else:
        queries_flat = queries

    num_samples = queries_flat.shape[0]
    device = queries.device

    if num_pseudo_queries > num_samples:
        raise ValueError(
            f"num_pseudo_queries ({num_pseudo_queries}) cannot exceed "
            f"number of samples ({num_samples})"
        )

    # Sample indices
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        indices = torch.randperm(num_samples, generator=generator, device=device)[
            :num_pseudo_queries
        ]
    else:
        indices = torch.randperm(num_samples, device=device)[:num_pseudo_queries]

    return queries_flat[indices].clone()
