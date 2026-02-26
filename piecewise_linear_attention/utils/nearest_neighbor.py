"""Nearest neighbor search utilities for finding closest pseudo-queries."""

import torch


def find_nearest_pseudo_queries(
    queries: torch.Tensor,
    pseudo_queries: torch.Tensor,
    metric: str = "euclidean",
) -> torch.Tensor:
    """Find nearest pseudo-query for each query.

    Args:
        queries: Query tensor of shape (batch, seq_len, dim) or (seq_len, dim)
        pseudo_queries: Pseudo-query tensor of shape (num_pseudo, dim)
        metric: Distance metric ('euclidean' or 'cosine')

    Returns:
        indices: Indices of nearest pseudo-queries, shape matches input queries
                 (batch, seq_len) or (seq_len,)

    Example:
        >>> queries = torch.randn(4, 32, 64)  # batch=4, seq_len=32, dim=64
        >>> pseudo_queries = torch.randn(16, 64)  # 16 pseudo-queries
        >>> indices = find_nearest_pseudo_queries(queries, pseudo_queries)
        >>> indices.shape
        torch.Size([4, 32])
    """
    if metric == "euclidean":
        return _find_nearest_euclidean(queries, pseudo_queries)
    elif metric == "cosine":
        return _find_nearest_cosine(queries, pseudo_queries)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'euclidean' or 'cosine'.")


def _find_nearest_euclidean(
    queries: torch.Tensor,
    pseudo_queries: torch.Tensor,
) -> torch.Tensor:
    """Find nearest pseudo-query using Euclidean distance."""
    # Handle both 2D and 3D inputs
    original_shape = queries.shape[:-1]  # (batch, seq_len) or (seq_len,)
    dim = queries.shape[-1]

    # Reshape to (num_queries, dim) for batch processing
    if queries.ndim == 3:
        batch, seq_len, dim = queries.shape
        queries_flat = queries.reshape(batch * seq_len, dim)
    elif queries.ndim == 2:
        queries_flat = queries
    else:
        raise ValueError(f"queries must be 2D or 3D, got {queries.ndim}D")

    # Compute pairwise distances: (num_queries, num_pseudo)
    distances = torch.cdist(queries_flat, pseudo_queries)

    # Find nearest pseudo-query for each query
    nearest_indices = distances.argmin(dim=1)

    # Reshape back to original shape
    nearest_indices = nearest_indices.reshape(original_shape)

    return nearest_indices


def _find_nearest_cosine(
    queries: torch.Tensor,
    pseudo_queries: torch.Tensor,
) -> torch.Tensor:
    """Find nearest pseudo-query using cosine similarity."""
    original_shape = queries.shape[:-1]
    dim = queries.shape[-1]

    # Reshape to (num_queries, dim)
    if queries.ndim == 3:
        batch, seq_len, dim = queries.shape
        queries_flat = queries.reshape(batch * seq_len, dim)
    elif queries.ndim == 2:
        queries_flat = queries
    else:
        raise ValueError(f"queries must be 2D or 3D, got {queries.ndim}D")

    # Normalize vectors
    queries_norm = torch.nn.functional.normalize(queries_flat, p=2, dim=1)
    pseudo_norm = torch.nn.functional.normalize(pseudo_queries, p=2, dim=1)

    # Compute cosine similarity: (num_queries, num_pseudo)
    similarities = torch.matmul(queries_norm, pseudo_norm.T)

    # Find most similar (highest similarity)
    nearest_indices = similarities.argmax(dim=1)

    # Reshape back
    nearest_indices = nearest_indices.reshape(original_shape)

    return nearest_indices


def apply_piecewise_linear_approximation(
    queries: torch.Tensor,
    nearest_indices: torch.Tensor,
    pseudo_queries: torch.Tensor,
    pseudo_outputs: torch.Tensor,
    jacobians: torch.Tensor,
) -> torch.Tensor:
    """Apply piecewise linear approximation to queries.

    For each query q, computes:
        A(q) ≈ A(q̃_j) + J_j · (q - q̃_j)

    where q̃_j is the nearest pseudo-query to q, A(q̃_j) is the precomputed
    attention output at q̃_j, and J_j is the precomputed Jacobian at q̃_j.

    Args:
        queries: Query tensor of shape (batch, seq_len, dim)
        nearest_indices: Indices of nearest pseudo-queries, shape (batch, seq_len)
        pseudo_queries: Pseudo-query tensor of shape (num_pseudo, dim)
        pseudo_outputs: Precomputed attention outputs at pseudo-queries,
                       shape (num_pseudo, dim)
        jacobians: Precomputed Jacobians at pseudo-queries,
                  shape (num_pseudo, dim, dim)

    Returns:
        outputs: Approximated attention outputs, shape (batch, seq_len, dim)
    """
    batch, seq_len, dim = queries.shape

    # Get the pseudo-queries, outputs, and Jacobians for nearest neighbors
    # Shape: (batch, seq_len, dim)
    nearest_pseudo_queries = pseudo_queries[nearest_indices]
    nearest_pseudo_outputs = pseudo_outputs[nearest_indices]
    # Shape: (batch, seq_len, dim, dim)
    nearest_jacobians = jacobians[nearest_indices]

    # Compute delta: q - q̃
    # Shape: (batch, seq_len, dim)
    delta = queries - nearest_pseudo_queries

    # Apply linear approximation: A(q̃) + J · (q - q̃)
    # Need to compute: nearest_jacobians @ delta for each position
    # Use einsum for efficient batched matrix-vector multiplication
    # 'bsij,bsj->bsi' means: for each (b, s), compute J[i,j] * delta[j]
    jacobian_delta = torch.einsum('bsij,bsj->bsi', nearest_jacobians, delta)

    # Final output: A(q̃) + J · Δq
    outputs = nearest_pseudo_outputs + jacobian_delta

    return outputs
