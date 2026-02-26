"""Jacobian computation for attention functions.

This module implements analytical and automatic differentiation-based
Jacobian computation for the softmax attention function.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def compute_attention_jacobian_analytical(
    query: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: bool = True,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Jacobian of attention function analytically.

    For the attention function A(q) = sum_i v_i * softmax_i(<k_i, q> / sqrt(d)),
    computes the Jacobian J = dA/dq.

    Mathematical formula:
        J = (1/sqrt(d)) * sum_i [alpha_i * (v_i ⊗ k_i)] -
            (1/sqrt(d)) * (sum_i alpha_i * v_i) ⊗ (sum_i alpha_i * k_i)

    where alpha_i are the attention weights (after softmax) and ⊗ is outer product.

    Args:
        query: Single query vector of shape (dim,)
        K: Key matrix of shape (seq_len, dim)
        V: Value matrix of shape (seq_len, dim)
        scale: Whether to scale by sqrt(d)
        mask: Optional attention mask of shape (seq_len,) where 0 masks out positions

    Returns:
        jacobian: Jacobian matrix of shape (dim, dim) where J[i,j] = dA_i/dq_j

    Note:
        This function operates on a single query vector. For batched computation,
        call this function in a loop or use vmap.
    """
    seq_len, dim = K.shape
    assert V.shape == (seq_len, dim), "K and V must have same shape"
    assert query.shape == (dim,), f"query must be 1D with shape ({dim},), got {query.shape}"

    # Compute scaled attention scores: s_i = <k_i, q> / sqrt(d)
    # Shape: (seq_len,)
    scores = torch.matmul(K, query)  # <k_i, q>

    scale_factor = 1.0
    if scale:
        scale_factor = 1.0 / math.sqrt(dim)
        scores = scores * scale_factor

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Compute attention weights: alpha_i = softmax_i(s)
    # Shape: (seq_len,)
    alpha = F.softmax(scores, dim=0)

    # Compute the Jacobian using the analytical formula:
    # J = scale_factor * sum_i [alpha_i * (v_i ⊗ k_i)] -
    #     scale_factor * (sum_i alpha_i * v_i) ⊗ (sum_i alpha_i * k_i)

    # First term: sum_i [alpha_i * (v_i ⊗ k_i)]
    # This is: V^T @ diag(alpha) @ K = V^T @ (alpha * K)
    # Shape: (dim, dim)
    weighted_K = alpha.unsqueeze(1) * K  # (seq_len, dim)
    first_term = torch.matmul(V.T, weighted_K)  # (dim, dim)

    # Second term: (sum_i alpha_i * v_i) ⊗ (sum_i alpha_i * k_i)
    # weighted_v: sum_i alpha_i * v_i, shape: (dim,)
    weighted_v = torch.matmul(alpha, V)  # (dim,)
    # weighted_k: sum_i alpha_i * k_i, shape: (dim,)
    weighted_k = torch.matmul(alpha, K)  # (dim,)
    # Outer product
    second_term = torch.outer(weighted_v, weighted_k)  # (dim, dim)

    # Combine terms
    jacobian = scale_factor * (first_term - second_term)

    return jacobian


def compute_attention_jacobian_autograd(
    query: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: bool = True,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Jacobian of attention function using autograd.

    This is a reference implementation for verification purposes.
    Use compute_attention_jacobian_analytical for production.

    Args:
        query: Single query vector of shape (dim,)
        K: Key matrix of shape (seq_len, dim)
        V: Value matrix of shape (seq_len, dim)
        scale: Whether to scale by sqrt(d)
        mask: Optional attention mask of shape (seq_len,)

    Returns:
        jacobian: Jacobian matrix of shape (dim, dim)
    """
    seq_len, dim = K.shape

    # Enable gradient computation
    query_copy = query.clone().detach().requires_grad_(True)

    # Compute attention output
    scores = torch.matmul(K, query_copy)

    if scale:
        scores = scores / math.sqrt(dim)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    alpha = F.softmax(scores, dim=0)
    output = torch.matmul(alpha, V)

    # Compute Jacobian using autograd
    # jacobian[i, j] = d(output[i]) / d(query[j])
    jacobian = torch.zeros(dim, dim, device=query.device, dtype=query.dtype)

    for i in range(dim):
        # Compute gradient of output[i] with respect to query
        if query_copy.grad is not None:
            query_copy.grad.zero_()

        output[i].backward(retain_graph=True)
        jacobian[i, :] = query_copy.grad.clone()

    return jacobian


def compute_attention_with_jacobian(
    query: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: bool = True,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute both attention output and its Jacobian efficiently.

    This function computes both the attention output A(q) and its Jacobian dA/dq
    in a single pass, reusing intermediate computations.

    Args:
        query: Single query vector of shape (dim,)
        K: Key matrix of shape (seq_len, dim)
        V: Value matrix of shape (seq_len, dim)
        scale: Whether to scale by sqrt(d)
        mask: Optional attention mask of shape (seq_len,)

    Returns:
        output: Attention output of shape (dim,)
        jacobian: Jacobian matrix of shape (dim, dim)
    """
    seq_len, dim = K.shape

    # Compute attention scores and weights
    scores = torch.matmul(K, query)

    scale_factor = 1.0
    if scale:
        scale_factor = 1.0 / math.sqrt(dim)
        scores = scores * scale_factor

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    alpha = F.softmax(scores, dim=0)

    # Compute attention output
    output = torch.matmul(alpha, V)  # (dim,)

    # Compute Jacobian (reusing alpha)
    weighted_K = alpha.unsqueeze(1) * K
    first_term = torch.matmul(V.T, weighted_K)

    weighted_v = output  # Already computed as torch.matmul(alpha, V)
    weighted_k = torch.matmul(alpha, K)
    second_term = torch.outer(weighted_v, weighted_k)

    jacobian = scale_factor * (first_term - second_term)

    return output, jacobian


def batch_compute_jacobians(
    queries: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: bool = True,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Jacobians for multiple queries efficiently.

    Args:
        queries: Query vectors of shape (num_queries, dim)
        K: Key matrix of shape (seq_len, dim)
        V: Value matrix of shape (seq_len, dim)
        scale: Whether to scale by sqrt(d)
        mask: Optional attention mask of shape (seq_len,)

    Returns:
        jacobians: Jacobian matrices of shape (num_queries, dim, dim)
    """
    num_queries, dim = queries.shape
    jacobians = torch.zeros(num_queries, dim, dim, device=queries.device, dtype=queries.dtype)

    for i in range(num_queries):
        jacobians[i] = compute_attention_jacobian_analytical(
            queries[i], K, V, scale=scale, mask=mask
        )

    return jacobians
