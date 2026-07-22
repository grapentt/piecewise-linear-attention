"""Registry for attention mechanisms.

A single dispatch point maps a short string name (e.g. ``"standard"``) to a
constructed attention module. This replaces per-call-site ``if/elif`` chains and
gives one place to register new mechanisms so they become usable everywhere
(multi-head wrapper, benchmark harness, configs) at once.

Example
-------
>>> from piecewise_linear_attention.core.registry import build_attention
>>> attn = build_attention("standard", dim=64, dropout=0.0, causal=False)

Register a custom mechanism::

    from piecewise_linear_attention.core.registry import register_attention

    @register_attention("my_attention")
    def _build(*, dim, dropout, causal, **kwargs):
        return MyAttention(dim=dim, dropout=dropout, causal=causal)
"""

from typing import Callable, Dict, List

from .attention import (
    BaseAttention,
    LinearAttention,
    LinformerAttention,
    LunaAttention,
    NystromformerAttention,
    PerformerAttention,
    PiecewiseAttention,
    StandardAttention,
)

# A builder takes the common attention kwargs plus mechanism-specific extras and
# returns a constructed module. Keeping the signature keyword-only makes call
# sites explicit and forward-compatible.
AttentionBuilder = Callable[..., BaseAttention]

ATTENTION_REGISTRY: Dict[str, AttentionBuilder] = {}


def register_attention(name: str) -> Callable[[AttentionBuilder], AttentionBuilder]:
    """Register an attention builder under ``name``.

    Parameters
    ----------
    name:
        Unique lookup key for :func:`build_attention`.

    Returns
    -------
    Callable
        A decorator that registers and returns the builder unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered.
    """

    def decorator(builder: AttentionBuilder) -> AttentionBuilder:
        if name in ATTENTION_REGISTRY:
            raise ValueError(f"Attention type '{name}' is already registered")
        ATTENTION_REGISTRY[name] = builder
        return builder

    return decorator


def available_attention_types() -> List[str]:
    """Return the sorted list of registered attention type names."""
    return sorted(ATTENTION_REGISTRY)


def build_attention(
    name: str,
    *,
    dim: int,
    dropout: float = 0.0,
    causal: bool = False,
    **kwargs,
) -> BaseAttention:
    """Construct a registered attention module.

    Parameters
    ----------
    name:
        Registered attention type (see :func:`available_attention_types`).
    dim:
        Per-head embedding dimension passed to the mechanism.
    dropout:
        Dropout probability for attention outputs/weights.
    causal:
        Whether to apply causal masking.
    **kwargs:
        Mechanism-specific options (e.g. ``kernel_type`` for linear attention,
        ``pseudo_query_fn`` / ``causal_pseudo_query`` for piecewise). Ignored by
        mechanisms that do not use them.

    Returns
    -------
    BaseAttention
        The constructed module.

    Raises
    ------
    ValueError
        If ``name`` is not registered; the message lists available types.
    """
    if name not in ATTENTION_REGISTRY:
        available = ", ".join(available_attention_types())
        raise ValueError(f"Unknown attention type: '{name}'. Available types: {available}")
    return ATTENTION_REGISTRY[name](dim=dim, dropout=dropout, causal=causal, **kwargs)


@register_attention("standard")
def _build_standard(*, dim, dropout, causal, scale=True, **_ignored) -> BaseAttention:
    """Softmax scaled dot-product attention."""
    return StandardAttention(dim=dim, dropout=dropout, scale=scale, causal=causal)


@register_attention("linear")
def _build_linear(
    *, dim, dropout, causal, kernel_type="elu", eps=1e-6, **_ignored
) -> BaseAttention:
    """Kernel-feature-map linear attention (Katharopoulos et al., 2020)."""
    return LinearAttention(
        dim=dim, dropout=dropout, kernel_type=kernel_type, eps=eps, causal=causal
    )


@register_attention("performer")
def _build_performer(
    *, dim, dropout, causal, num_features=None, performer_seed=0, eps=1e-6, **_ignored
) -> BaseAttention:
    """Performer FAVOR+ positive-random-feature attention."""
    return PerformerAttention(
        dim=dim,
        dropout=dropout,
        num_features=num_features,
        eps=eps,
        causal=causal,
        seed=performer_seed,
    )


@register_attention("nystromformer")
def _build_nystromformer(
    *, dim, dropout, causal, num_landmarks=64, pinv_iterations=6, eps=1e-8, **_ignored
) -> BaseAttention:
    """Nyströmformer landmark attention (Xiong et al., 2021).

    No exact causal form; ``causal=True`` raises in the constructor.
    """
    return NystromformerAttention(
        dim=dim,
        dropout=dropout,
        num_landmarks=num_landmarks,
        pinv_iterations=pinv_iterations,
        eps=eps,
        causal=causal,
    )


@register_attention("linformer")
def _build_linformer(*, dim, dropout, causal, max_seq_len=512, k=256, **_ignored) -> BaseAttention:
    """Linformer low-rank sequence-projection attention (Wang et al., 2020).

    Same-length-only baseline; requires ``max_seq_len`` at construction and has
    no causal form.
    """
    return LinformerAttention(dim=dim, dropout=dropout, max_seq_len=max_seq_len, k=k, causal=causal)


@register_attention("luna")
def _build_luna(*, dim, dropout, causal, num_pack=256, **_ignored) -> BaseAttention:
    """Luna nested pack/unpack attention (Ma et al., 2021)."""
    return LunaAttention(dim=dim, dropout=dropout, num_pack=num_pack, causal=causal)


@register_attention("piecewise")
def _build_piecewise(
    *,
    dim,
    dropout,
    causal,
    scale=True,
    pseudo_query_fn=None,
    causal_pseudo_query="mean",
    **_ignored,
) -> BaseAttention:
    """First-order Taylor (piecewise-linear) attention around anchor queries."""
    return PiecewiseAttention(
        dim=dim,
        dropout=dropout,
        scale=scale,
        pseudo_query_fn=pseudo_query_fn,
        causal=causal,
        causal_pseudo_query=causal_pseudo_query,
    )


@register_attention("piecewise_kmeans")
def _build_piecewise_kmeans(
    *, dim, dropout, causal, scale=True, num_anchors=4, kmeans_iters=3, **_ignored
) -> BaseAttention:
    """Multi-anchor piecewise attention with k-means-selected anchors."""
    from .anchors import KMeansAnchor

    return PiecewiseAttention(
        dim=dim,
        dropout=dropout,
        scale=scale,
        causal=causal,
        anchor_strategy=KMeansAnchor(k=num_anchors, iters=kmeans_iters),
    )


@register_attention("piecewise_stride")
def _build_piecewise_stride(
    *, dim, dropout, causal, scale=True, num_anchors=4, **_ignored
) -> BaseAttention:
    """Multi-anchor piecewise attention with evenly strided anchors."""
    from .anchors import StrideAnchor

    return PiecewiseAttention(
        dim=dim,
        dropout=dropout,
        scale=scale,
        causal=causal,
        anchor_strategy=StrideAnchor(k=num_anchors),
    )
