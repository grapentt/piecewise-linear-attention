"""Behavioral safety net for :class:`MultiHeadAttention`.

These tests pin the *observable contract* of the multi-head wrapper across every
supported ``attention_type`` so that internal restructuring (e.g. batching the
per-head computation) can be proven equivalent rather than merely plausible.

The equivalence test recomputes the reference with an independent naive per-head
loop and asserts the batched implementation matches it. That makes the guarantee
self-contained (no checked-in binary fixture) while still catching any change
that is not a pure numerical no-op: the per-head loop and the batched
implementation must be numerically interchangeable.
"""

import pytest
import torch

from piecewise_linear_attention.models.multihead import MultiHeadAttention

# All attention families the wrapper is expected to support. Kept as a single
# source of truth so new families are exercised by every test below.
ATTENTION_TYPES = ["standard", "linear", "piecewise"]


def _make_module(attention_type, *, hidden_dim=32, num_heads=4, causal=False):
    """Construct a deterministically-initialized wrapper on CPU.

    Seeding immediately before construction makes the projection weights
    reproducible regardless of test ordering.
    """
    torch.manual_seed(0)
    return MultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        attention_type=attention_type,
        dropout=0.0,
        causal=causal,
    )


def _make_inputs(*, batch=2, seq_len=6, hidden_dim=32):
    """Fixed, reproducible query/key/value states."""
    torch.manual_seed(1)
    x = torch.randn(batch, seq_len, hidden_dim)
    return x


def _naive_per_head_reference(mha, attending, attended):
    """Independent reference: apply attention head-by-head in a Python loop.

    This mirrors the semantics the wrapper must preserve — split into heads,
    attend each head separately, concatenate, project — without the batched
    reshape the production code uses. Equivalence between this and the wrapper's
    output is the property under test.
    """
    batch, q_len, _ = attending.shape
    k_len = attended.shape[1]
    num_heads, head_dim, hidden = mha.num_heads, mha.attn_dim, mha.hidden_dim

    Q = mha.q_projection(attending).view(batch, q_len, num_heads, head_dim)
    K = mha.k_projection(attended).view(batch, k_len, num_heads, head_dim)
    V = mha.v_projection(attended).view(batch, k_len, num_heads, head_dim)

    head_outputs = []
    for head in range(num_heads):
        out_head, _ = mha.attention(Q[:, :, head, :], K[:, :, head, :], V[:, :, head, :])
        head_outputs.append(out_head)
    merged = torch.stack(head_outputs, dim=2).reshape(batch, q_len, hidden)
    return mha.out_projection(merged)


class TestMultiHeadContract:
    """Shape, finiteness and gradient contract per attention type."""

    @pytest.mark.parametrize("attention_type", ATTENTION_TYPES)
    def test_output_shape(self, attention_type):
        """Output preserves (batch, query_len, hidden_dim).

        Locks: the head split/merge round-trip. Catches an off-by-reshape in a
        batched-heads refactor that would silently transpose or truncate.
        """
        mha = _make_module(attention_type)
        x = _make_inputs()
        out = mha(x, x)
        assert out.shape == x.shape

    @pytest.mark.parametrize("attention_type", ATTENTION_TYPES)
    def test_output_is_finite(self, attention_type):
        """No NaN/inf for well-scaled inputs.

        Locks: numerical stability across all families. Catches a normalization
        or masking mistake introduced during restructuring.
        """
        mha = _make_module(attention_type)
        x = _make_inputs()
        out = mha(x, x)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("attention_type", ATTENTION_TYPES)
    def test_gradient_flow(self, attention_type):
        """Gradients reach the input and all projection weights.

        Locks: the computation stays in the autograd graph. Catches an
        accidental ``detach``/``no_grad`` wrapping the wrong region during the
        refactor.
        """
        mha = _make_module(attention_type)
        x = _make_inputs().requires_grad_(True)
        out = mha(x, x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for name, p in mha.named_parameters():
            assert p.grad is not None, f"no grad for {name}"


class TestLastAttentionExposure:
    """The ``last_attention`` visualization hook."""

    def test_standard_exposes_weights(self):
        """Standard attention records weights shaped (batch, heads, q, k).

        Locks: the visualization contract for softmax attention. Catches a
        refactor that forgets to reshape batched (B*H, n, n) weights back to
        (B, H, n, n), or drops the hook entirely.
        """
        batch, seq_len, hidden_dim, num_heads = 2, 6, 32, 4
        mha = _make_module("standard", hidden_dim=hidden_dim, num_heads=num_heads)
        x = _make_inputs(batch=batch, seq_len=seq_len, hidden_dim=hidden_dim)
        mha(x, x)
        assert mha.last_attention is not None
        assert mha.last_attention.shape == (batch, num_heads, seq_len, seq_len)
        # Softmax rows are a probability distribution over keys.
        row_sums = mha.last_attention.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestBatchedHeadEquivalence:
    """The batched-heads path equals the naive per-head loop.

    This is the load-bearing safety net for the batched-heads refactor. The
    reference is recomputed by an independent per-head loop in the test, so the
    guarantee is self-contained (no binary fixture) yet still catches any change
    that is not a pure numerical no-op.
    """

    @pytest.mark.parametrize("attention_type", ATTENTION_TYPES)
    def test_matches_naive_loop(self, attention_type):
        """Batched forward matches head-by-head computation to 1e-5.

        Locks: exact numerical equivalence between the batched (B*H) computation
        and a per-head loop for every family. Catches a reshape/permute mistake
        that would mix heads or transpose the output.

        Assessment: not over-testing — the wrapper had no prior test and the
        entire refactor rests on this equivalence. Recomputing the reference
        independently (rather than trusting the same code) is what makes it a
        genuine guard.
        """
        mha = _make_module(attention_type)
        x = _make_inputs()
        with torch.no_grad():
            out = mha(x, x)
            reference = _naive_per_head_reference(mha, x, x)
        assert torch.allclose(
            out, reference, atol=1e-5
        ), f"{attention_type}: batched output diverged from per-head loop"
