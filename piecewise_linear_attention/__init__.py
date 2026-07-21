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

# Anchor-selection strategies for piecewise (multi-)anchor attention
from .core.anchors import (
    AnchorStrategy,
    CallableAnchor,
    FirstAnchor,
    KMeansAnchor,
    MeanAnchor,
    StrideAnchor,
)

# Main attention implementations
from .core.attention import (
    EfficientAttention,
    LinearAttention,
    PerformerAttention,
    PiecewiseAttention,
    StandardAttention,
)

# Attention registry
from .core.registry import (
    available_attention_types,
    build_attention,
    register_attention,
)

__all__ = [
    # Main attention mechanisms
    "StandardAttention",
    "LinearAttention",
    "PerformerAttention",
    "PiecewiseAttention",
    # Convenient alias
    "EfficientAttention",
    # Anchor strategies
    "AnchorStrategy",
    "MeanAnchor",
    "FirstAnchor",
    "StrideAnchor",
    "KMeansAnchor",
    "CallableAnchor",
    # Registry
    "build_attention",
    "register_attention",
    "available_attention_types",
]
