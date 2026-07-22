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


def _reference_multi_anchor_gather(attn, Q, K, V, anchors):
    """Naive multi-anchor apply: build a per-query (d, d) Jacobian and apply it.

    This is a frozen copy of the straightforward, pre-optimization algebra: it
    materializes the per-anchor Jacobian ``(batch, m, dim, dim)``, gathers it to
    a per-query ``(batch, seq_len, dim, dim)`` tensor, and applies
    ``out_i = A(a) + J(a) @ (q_i - a)``. It exists only as an independent
    reference for the fused implementation and is deliberately not optimized.
    """
    _, _, dim = Q.shape
    scale_factor = 1.0 / math.sqrt(dim) if attn.scale else 1.0

    scores = torch.einsum("bmd,bnd->bmn", anchors, K) * scale_factor
    alpha = torch.softmax(scores, dim=-1)
    out_anchor = torch.einsum("bmn,bnd->bmd", alpha, V)
    weighted_k = torch.einsum("bmn,bnd->bmd", alpha, K)
    weighted_outer = torch.einsum("bmn,bnd,bne->bmde", alpha, V, K)
    outer_weighted = out_anchor.unsqueeze(-1) @ weighted_k.unsqueeze(-2)
    jac_anchor = scale_factor * (weighted_outer - outer_weighted)  # (b, m, d, d)

    assign = torch.cdist(Q, anchors).argmin(dim=-1)  # (b, n)
    idx_d = assign.unsqueeze(-1).expand(-1, -1, dim)
    a_sel = torch.gather(anchors, 1, idx_d)
    out_sel = torch.gather(out_anchor, 1, idx_d)
    idx_dd = assign.view(*assign.shape, 1, 1).expand(-1, -1, dim, dim)
    jac_sel = torch.gather(jac_anchor, 1, idx_dd)  # (b, n, d, d)

    delta = (Q - a_sel).unsqueeze(-1)
    return out_sel + (jac_sel @ delta).squeeze(-1)


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

    @pytest.mark.parametrize(
        "b,n,d,m,collapse",
        [
            (3, 40, 16, 4, False),  # ordinary routing -> batched capacity-bucket path
            (1, 20, 4, 8, True),    # m>d + one dominant cluster -> skew-guard fallback
        ],
    )
    def test_fused_matches_gather_reference(self, b, n, d, m, collapse):
        """Fused apply equals a naive per-query-Jacobian gather reference.

        Locks: the fused ``O(seq_len·dim²)`` multi-anchor apply is numerically
        equivalent to the straightforward implementation that materializes a
        ``(batch, m, dim, dim)`` Jacobian, gathers it to ``(batch, seq_len, dim,
        dim)`` per query, and applies it. The refactor changed *how* the
        expansion is computed (distributing the matmul over the rank-1 Jacobian),
        not *what* it computes.

        Catches: any algebra error in the fusion — wrong einsum subscripts, a
        dropped ``scale_factor``, a wrong sign on the rank-1 term, selecting the
        wrong anchor slice, or an off-by-one in the assignment gather. None of
        these would be caught by a shape or finiteness check.

        Both apply schedules are covered: the default capacity-bucket group-by-
        anchor matmul, and the skew-guard fallback (a direct per-query gather-
        matmul) that triggers when ``m · capacity > n · dim`` — reached here by an
        ``m > dim`` head with the queries collapsed onto one dominant cluster so
        one group's capacity approaches the full sequence. The fallback is a pure
        performance switch and must return the identical result.

        Assessment: appropriate, not over-testing — this is the single
        load-bearing test that the optimization preserves output. It overlaps
        ``test_multi_anchor_hard_assignment`` only superficially: that test's
        reference is the exact per-anchor Jacobian path
        (``_compute_attention_and_jacobian``), so it validates *routing*, while
        this one validates the *fusion algebra* against an independent
        gather-based reference. The second shape folds in the otherwise-unexercised
        fallback branch rather than adding a standalone test.
        """
        torch.manual_seed(0)
        if collapse:
            # Nearly identical queries route almost everything to one anchor, so
            # its capacity ~ n and m·capacity > n·dim trips the skew guard.
            Q = 0.001 * torch.randn(b, n, d) + 5.0
        else:
            Q = torch.randn(b, n, d)
        K = torch.randn(b, n, d)
        V = torch.randn(b, n, d)
        attn = PiecewiseAttention(dim=d, scale=True, anchor_strategy=StrideAnchor(m))
        with torch.no_grad():
            out, anchors = attn(Q, K, V)
            ref = _reference_multi_anchor_gather(attn, Q, K, V, anchors)
        assert torch.allclose(out, ref, atol=1e-5)

    @pytest.mark.parametrize("strategy", [StrideAnchor(4), KMeansAnchor(4)])
    def test_multi_anchor_runs_in_float16(self, strategy):
        """Multi-anchor forward runs under float16 on CPU without crashing.

        Locks: the nearest-anchor assignment (``torch.cdist``) does not break
        the multi-anchor path in half precision. ``cdist`` has no half-precision
        kernel on CPU, so the assignment must be computed in a promoted dtype.

        Catches: a hard ``NotImplementedError`` on fp16 for both StrideAnchor
        and KMeansAnchor (``torch.cdist`` has no CPU half kernel).
        Asserted directly (not wrapped in a skip) so the crash cannot be masked;
        the sibling single-anchor fp16 test in ``test_optimization_correctness``
        never exercises the anchor-assignment path, so this is the only guard.

        Assessment: appropriate. It covers a real crash on a supported dtype for
        a headline feature, across both distance-based strategies (which share
        the cdist blocker); one representative shape per strategy suffices.
        """
        torch.manual_seed(0)
        d = 16
        Q = torch.randn(2, 12, d, dtype=torch.float16)
        K = torch.randn(2, 12, d, dtype=torch.float16)
        V = torch.randn(2, 12, d, dtype=torch.float16)
        attn = PiecewiseAttention(dim=d, scale=True, anchor_strategy=strategy)
        out, _ = attn(Q, K, V)
        assert out.shape == Q.shape
        assert out.dtype == torch.float16
        assert torch.isfinite(out).all()

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

    def test_build_topp_reverts_when_diffuse(self):
        """Top-p build reproduces the dense build at p=1 and on diffuse weights.

        Locks the two invariants that make ``build_topp`` a speed-only knob:
          (a) ``build_topp=1.0`` keeps every key, so the truncated build equals
              the exact dense build (up to float reassociation); and
          (b) when the anchor softmax is diffuse (near-uniform), no key can be
              dropped without losing mass, so the nucleus grows back to all keys
              and even an aggressive ``p=0.999`` reverts to the dense output.

        Catches: a truncation that fires when it must not — dropping tail mass on
        a flat distribution (which would silently degrade accuracy exactly where
        the approximation has no headroom), an off-by-one that excludes the
        threshold-crossing key, or a wrong unrenormalized-vs-renormalized sum.

        Assessment: appropriate. This is the load-bearing correctness property of
        the approximation (it must never hurt a diffuse case), tested with one
        p=1 invariant plus the diffuse revert — not per-p over-testing. The
        peaked-regime *speedup* is a benchmark concern, not a unit-test assertion,
        so it is deliberately not asserted here.
        """
        torch.manual_seed(0)
        b, n, d, m = 3, 128, 16, 4
        Q = torch.randn(b, n, d)
        V = torch.randn(b, n, d)

        def run(K, topp):
            attn = PiecewiseAttention(
                dim=d, scale=True, anchor_strategy=StrideAnchor(m), build_topp=topp
            )
            with torch.no_grad():
                out, _ = attn(Q, K, V)
            return out

        # (a) p=1.0 keeps all keys regardless of how peaked the weights are. The
        # top-p path sorts and gathers keys, so the summation order differs from
        # the dense contraction — the mismatch is pure fp32 reassociation, scaled
        # up here by the deliberately large-magnitude peaked weights, so the
        # tolerance is looser than the bit-identical exact-path tests.
        K_peaked = 5.0 * torch.randn(b, n, d)
        assert torch.allclose(run(K_peaked, 1.0), run(K_peaked, None), atol=1e-4, rtol=1e-4)

        # (b) diffuse weights force the nucleus back to the full sequence, so an
        # aggressive p still reverts to the exact dense output.
        K_diffuse = 0.01 * torch.randn(b, n, d)
        assert torch.allclose(run(K_diffuse, 0.999), run(K_diffuse, None), atol=1e-5)

    def test_build_topp_rejects_bad_value(self):
        """build_topp outside (0, 1] is rejected at construction.

        Locks: the documented domain of the knob. Catches a typo'd probability
        (e.g. a percentage like ``99``) silently degrading the build.
        """
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                PiecewiseAttention(dim=8, anchor_strategy=StrideAnchor(4), build_topp=bad)

    def test_build_topp_truncates_to_per_anchor_nucleus(self):
        """Top-p build sums each anchor over exactly its own nucleus of keys.

        The two ``test_build_topp_reverts_when_diffuse`` assertions both land on
        the keep-all boundary (``t == n``): they lock that truncation never fires
        when it must not, but they never exercise the *active* regime the feature
        exists for. This constructs moderately peaked keys so the per-anchor
        nuclei are genuinely smaller than the sequence *and* differ across anchors
        (sizes like 5/7/8 out of 32), then checks the fused factors equal an
        explicit ragged reference that sums each anchor over only its own top keys.

        Locks: the load-bearing algebra of the truncated build — the crossing-key
        rule (``keep[...,1:] = cumulative[...,:-1] < p``, so the key that crosses p
        is kept), the unrenormalized partial sum, and the ``torch.where`` zeroing of
        the surplus slots between a short anchor's nucleus and the shared batched
        cap ``t = max_j |nucleus_j|``.

        Catches: an off-by-one in the crossing key (``<`` vs ``<=``), a tail key
        leaking through unzeroed surplus slots (silent accuracy loss), and a wrong
        advanced-index axis in the ``V``/``K`` gather. None are caught by the
        keep-all boundary tests.

        Assessment: appropriate, not over-testing. This is the single distinguishing
        behavior of the top-p build; the shipped tests only prove the no-op limit,
        so nothing else guards the sparse path.
        """
        torch.manual_seed(3)
        b, n, d, m = 1, 32, 8, 3
        Q = torch.randn(b, n, d)
        V = torch.randn(b, n, d)
        K = 2.0 * torch.randn(b, n, d)  # moderate peaking -> mid-size, uneven nuclei
        p = 0.9

        attn = PiecewiseAttention(
            dim=d, scale=True, anchor_strategy=StrideAnchor(m), build_topp=p
        )
        with torch.no_grad():
            anchors = attn.anchor_strategy.select(Q)
            scale = 1.0 / math.sqrt(d)
            alpha = torch.softmax(
                torch.einsum("bmd,bnd->bmn", anchors, K) * scale, dim=-1
            )
            wk, second_moment = attn._build_jacobian_factors_topp(alpha, V, K)

            # Independent ragged reference: each anchor sums over ONLY the smallest
            # set of its own top keys whose cumulative mass reaches p (crossing key
            # included), with no shared cap and no surplus slots.
            sorted_alpha, sort_idx = torch.sort(alpha, dim=-1, descending=True)
            cumulative = sorted_alpha.cumsum(dim=-1)
            keep = torch.ones_like(sorted_alpha, dtype=torch.bool)
            keep[..., 1:] = cumulative[..., :-1] < p
            counts = keep.sum(dim=-1)  # (b, m)

            # The construction must actually exercise the sparse, uneven regime.
            assert int(counts.max()) < n, "nucleus is keep-all; truncation never fires"
            assert counts.min() != counts.max(), "per-anchor nuclei are homogeneous"

            wk_ref = torch.zeros(b, m, d)
            m_ref = torch.zeros(b, m, d, d)
            for bi in range(b):
                for j in range(m):
                    t_j = int(counts[bi, j])
                    idx = sort_idx[bi, j, :t_j]
                    aw = alpha[bi, j, idx]
                    wk_ref[bi, j] = (aw[:, None] * K[bi, idx]).sum(0)
                    m_ref[bi, j] = torch.einsum("t,td,te->de", aw, V[bi, idx], K[bi, idx])

        assert torch.allclose(wk, wk_ref, atol=1e-5)
        assert torch.allclose(second_moment, m_ref, atol=1e-5)

    def test_grad_flows_through_topp_build(self):
        """Gradients reach Q, K, V through the active top-p build subgraph.

        ``test_grad_flows_through_multi_anchor`` covers only the dense build
        (``build_topp=None``); the ``torch.sort`` + advanced-index gather +
        ``torch.where`` subgraph that the truncated build adds is never
        differentiated there. This runs the sparse path (peaked keys so ``t < n``)
        and checks finite, non-trivial gradients still reach all of Q, K, V.

        Locks: the truncated build stays differentiable end-to-end. Catches a
        future edit that detaches ``alpha_kept`` or gathers keys/values under
        ``no_grad`` — which would silently zero the K/V gradient on the linear
        correction and pass every exactness test.

        Assessment: appropriate — reuses the sibling test's body with peaked keys
        and one config; the gap it fills (grad through the sparse subgraph) is real.
        """
        torch.manual_seed(0)
        dim = 8
        Q = torch.randn(2, 12, dim, requires_grad=True)
        K = (3.0 * torch.randn(2, 12, dim)).requires_grad_(True)  # peaked -> t < n
        V = torch.randn(2, 12, dim, requires_grad=True)
        attn = PiecewiseAttention(
            dim=dim, scale=True, anchor_strategy=KMeansAnchor(3), build_topp=0.9
        )
        out, _ = attn(Q, K, V)
        out.sum().backward()
        for name, t in [("Q", Q), ("K", K), ("V", V)]:
            assert t.grad is not None, f"no grad for {name}"
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0, f"zero grad for {name}"

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
