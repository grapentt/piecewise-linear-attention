"""Piecewise Linear Attention: Efficient attention mechanism using piecewise approximation.

Main exports:
- PiecewiseAttention: RECOMMENDED - 8-9× speedup, 52% error
- StandardAttention: Baseline softmax attention
- LinearAttention: Linear attention with kernel feature maps

Quick start:
    >>> from piecewise_linear_attention import PiecewiseAttention
    >>>
    >>> attention = PiecewiseAttention(dim=64)
    >>> output, _ = attention(Q, K, V)
"""

__version__ = "0.2.0"

# Main attention implementations
from .core.attention import (
    StandardAttention,
    LinearAttention,
    PiecewiseAttention,
    EfficientAttention,
)

__all__ = [
    # Main attention mechanisms
    "StandardAttention",
    "LinearAttention",
    "PiecewiseAttention",
    # Convenient alias
    "EfficientAttention",
]
