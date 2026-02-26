"""Piecewise Linear Attention: Efficient attention mechanism using piecewise linear approximation."""

__version__ = "0.1.0"

from .core.attention import StandardAttention, LinearAttention, PiecewiseLinearAttention

__all__ = [
    "StandardAttention",
    "LinearAttention",
    "PiecewiseLinearAttention",
]
