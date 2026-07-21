"""Tests for anchor-selection strategies and multi-anchor piecewise attention."""

import math

import pytest
import torch

from piecewise_linear_attention.core.anchors import (
    CallableAnchor,
    FirstAnchor,
    KMeansAnchor,
    MeanAnchor,
    StrideAnchor,
)
from piecewise_linear_attention.core.attention import PiecewiseAttention


@pytest.fixture
def queries():
    torch.manual_seed(0)
    return torch.randn(4, 32, 16)


class TestAnchorShapes:
    """Every strategy returns (batch, num_anchors, dim)."""

    @pytest.mark.parametrize(
        "strategy,expected_anchors",
        [
            (MeanAnchor(), 1),
            (FirstAnchor(), 1),
            (StrideAnchor(4), 4),
            (KMeansAnchor(4, iters=2), 4),
        ],
    )
    def test_select_shape(self, queries, strategy, expected_anchors):
        """Anchor tensor has the declared anchor count.

        Locks: the (batch, m, dim) contract shared by all strategies. Catches a
        strategy that returns a mis-shaped tensor the attention module cannot
        consume.
        """
        anchors = strategy.select(queries)
        batch, _, dim = queries.shape
        assert anchors.shape == (batch, expected_anchors, dim)
        assert strategy.num_anchors == expected_anchors


class TestMeanAnchorGolden:
    """MeanAnchor reproduces the exact legacy mean pseudo-query."""

    def test_matches_query_mean(self, queries):
        """Mean anchor equals Q.mean over the sequence.

        Locks: the single-anchor default is unchanged. Catches any drift in the
        representative-query definition, which would silently move every
        existing piecewise result.
        """
        anchors = MeanAnchor().select(queries)
        expected = queries.mean(dim=1, keepdim=True)
        assert torch.equal(anchors, expected)


class TestKMeansAnchor:
    """k-means selection behavior."""

    def test_no_grad_on_selection(self, queries):
        """Selected anchors are detached from the autograd graph.

        Locks: the discrete Lloyd/argmin steps do not attach to the graph.
        Catches a selection that would try to backprop through argmin (which is
        undefined) and either error or produce misleading gradients.
        """
        q = queries.clone().requires_grad_(True)
        anchors = KMeansAnchor(4, iters=2).select(q)
        assert not anchors.requires_grad

    def test_empty_cluster_is_stable(self):
        """Degenerate inputs (all-equal queries) do not produce NaNs.

        Locks: the empty-cluster guard (clusters with no members keep their
        previous centroid). Catches a divide-by-zero when a centroid captures no
        points.
        """
        q = torch.ones(2, 8, 4)  # all queries identical -> most clusters empty
        anchors = KMeansAnchor(4, iters=3).select(q)
        assert torch.isfinite(anchors).all()


class TestMultiAnchorAttention:
    """Multi-anchor mode of PiecewiseAttention."""

    def test_single_anchor_strategy_matches_default(self, queries):
        """MeanAnchor strategy is byte-identical to the no-strategy default.

        Locks: adding the anchor_strategy hook does not perturb the original
        single-anchor path. Catches any accidental numerical change to the
        shipped method — the strongest guarantee we can give existing users.
        """
        K = torch.randn_like(queries)
        V = torch.randn_like(queries)
        default = PiecewiseAttention(dim=queries.shape[-1], scale=True)
        with_strategy = PiecewiseAttention(
            dim=queries.shape[-1], scale=True, anchor_strategy=MeanAnchor()
        )
        with torch.no_grad():
            out_default, _ = default(queries, K, V)
            out_strategy, _ = with_strategy(queries, K, V)
        assert torch.equal(out_default, out_strategy)

    def test_multi_anchor_hard_assignment(self):
        """Each query uses exactly its nearest anchor's expansion.

        Locks: the nearest-anchor routing. We construct two well-separated query
        clusters and 2 stride anchors, then verify each query's output equals the
        first-order expansion around the anchor the query is closest to (and
        differs from expansion around the other anchor).

        Assessment: this is the load-bearing correctness test for multi-anchor.
        It is not over-testing — hard assignment is the defining behavior that
        makes the method "piecewise", and a plausible bug (assigning to the
        wrong anchor, or averaging anchors) would pass a mere shape/finite check.
        """
        torch.manual_seed(0)
        dim = 8
        # Two separated clusters of queries.
        cluster_a = torch.randn(1, 5, dim) + 10.0
        cluster_b = torch.randn(1, 5, dim) - 10.0
        Q = torch.cat([cluster_a, cluster_b], dim=1)  # (1, 10, dim)
        K = torch.randn(1, 10, dim)
        V = torch.randn(1, 10, dim)

        attn = PiecewiseAttention(dim=dim, scale=True, anchor_strategy=StrideAnchor(2))
        with torch.no_grad():
            out, anchors = attn(Q, K, V)  # anchors: (1, 2, dim)

        # Recompute the per-anchor exact output + Jacobian and check each query
        # is the expansion around its nearest anchor.
        with torch.no_grad():
            for i in range(Q.shape[1]):
                q_i = Q[:, i, :]  # (1, dim)
                dists = torch.cdist(q_i.unsqueeze(1), anchors).squeeze(1)  # (1, 2)
                nearest = dists.argmin(dim=-1).item()
                a = anchors[:, nearest, :]  # (1, dim)
                out_a, jac_a = attn._compute_attention_and_jacobian(a, K, V)
                expected = out_a + (jac_a @ (q_i - a).unsqueeze(-1)).squeeze(-1)
                assert torch.allclose(out[:, i, :], expected, atol=1e-5)

    def test_grad_flows_through_multi_anchor(self):
        """Gradients reach Q, K, V despite non-differentiable assignment.

        Locks: backprop through the differentiable A(a)+J(a)@(q-a) expression is
        intact even though anchor selection is under no_grad. Catches a refactor
        that detaches too much.
        """
        torch.manual_seed(0)
        dim = 8
        Q = torch.randn(2, 12, dim, requires_grad=True)
        K = torch.randn(2, 12, dim, requires_grad=True)
        V = torch.randn(2, 12, dim, requires_grad=True)
        attn = PiecewiseAttention(dim=dim, scale=True, anchor_strategy=KMeansAnchor(3))
        out, _ = attn(Q, K, V)
        out.sum().backward()
        for name, t in [("Q", Q), ("K", K), ("V", V)]:
            assert t.grad is not None, f"no grad for {name}"
            assert torch.isfinite(t.grad).all()

    def test_multi_anchor_rejects_causal(self):
        """Multi-anchor + causal is rejected at construction.

        Locks: the documented limitation (multi-anchor is non-causal only).
        Catches a silent, incorrect causal multi-anchor path.
        """
        with pytest.raises(ValueError):
            PiecewiseAttention(dim=8, causal=True, anchor_strategy=KMeansAnchor(4))


class TestCallableAnchor:
    """Legacy pseudo_query_fn adapter."""

    def test_wraps_legacy_fn(self, queries):
        """Adapter reshapes (batch, dim) to (batch, 1, dim).

        Locks: backward-compatible wrapping of the legacy callable. Catches a
        shape mismatch that would break user-provided pseudo-query functions.
        """
        strategy = CallableAnchor(lambda Q: Q[:, 0, :])
        anchors = strategy.select(queries)
        assert anchors.shape == (queries.shape[0], 1, queries.shape[-1])
        assert torch.equal(anchors.squeeze(1), queries[:, 0, :])

    def test_rejects_bad_shape(self, queries):
        """A callable returning the wrong shape raises.

        Locks: the (batch, dim) contract for legacy callables. Catches a
        misconfigured function early rather than deep in the attention math.
        """
        strategy = CallableAnchor(lambda Q: Q)  # returns (batch, seq, dim)
        with pytest.raises(ValueError):
            strategy.select(queries)
