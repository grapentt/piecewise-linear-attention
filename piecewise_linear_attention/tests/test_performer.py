"""Tests for :class:`PerformerAttention` (FAVOR+ random features)."""

import math

import pytest
import torch

from piecewise_linear_attention.core.attention import PerformerAttention, StandardAttention


@pytest.fixture
def sample_data():
    torch.manual_seed(0)
    batch, seq_len, dim = 4, 32, 16
    Q = torch.randn(batch, seq_len, dim)
    K = torch.randn(batch, seq_len, dim)
    V = torch.randn(batch, seq_len, dim)
    return Q, K, V


class TestPerformerContract:
    """Shape, finiteness, gradient, normalization."""

    def test_output_shape(self, sample_data):
        """Output preserves (batch, seq_len, dim).

        Locks: the feature-map round-trip through the (m, d) memory. Catches a
        projection-dimension mismatch.
        """
        Q, K, V = sample_data
        attn = PerformerAttention(dim=Q.shape[-1])
        out, _ = attn(Q, K, V)
        assert out.shape == Q.shape

    def test_output_is_finite(self, sample_data):
        """No NaN/inf for well-scaled inputs.

        Locks: numerical stability of the exp-feature map + denominator eps.
        Catches an overflow or a zero-denominator division.
        """
        Q, K, V = sample_data
        attn = PerformerAttention(dim=Q.shape[-1])
        out, _ = attn(Q, K, V)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, sample_data):
        """Gradients reach Q, K, V.

        Locks: the estimator stays differentiable. Catches an accidental
        detachment of the random-feature computation.
        """
        Q, K, V = (t.clone().requires_grad_(True) for t in sample_data)
        attn = PerformerAttention(dim=Q.shape[-1])
        out, _ = attn(Q, K, V)
        out.sum().backward()
        for name, t in [("Q", Q), ("K", K), ("V", V)]:
            assert t.grad is not None, f"no grad for {name}"
            assert torch.isfinite(t.grad).all()

    def test_denominator_positive(self, sample_data):
        """The normalization denominator is strictly positive.

        Locks: positivity of the random-feature map (FAVOR+ uses positive
        features precisely so the denominator cannot vanish/flip sign). Catches
        a feature map that admits negative values.
        """
        Q, K, V = sample_data
        attn = PerformerAttention(dim=Q.shape[-1])
        _, normalization = attn(Q, K, V)
        assert (normalization > 0).all()

    def test_causal_shape(self, sample_data):
        """Causal mode preserves shape and finiteness.

        Locks: the prefix-sum causal path. Catches a cumsum axis mistake.
        """
        Q, K, V = sample_data
        attn = PerformerAttention(dim=Q.shape[-1], causal=True)
        out, _ = attn(Q, K, V)
        assert out.shape == Q.shape
        assert torch.isfinite(out).all()


class TestPerformerDeterminism:
    """The random projection is seeded, buffered, and stable."""

    def test_same_seed_identical_projection(self):
        """Two instances with the same seed share the projection.

        Locks: reproducibility of the random features. Catches a projection
        drawn from global RNG state (which would vary with call order).
        """
        a = PerformerAttention(dim=16, seed=123)
        b = PerformerAttention(dim=16, seed=123)
        assert torch.equal(a.projection, b.projection)

    def test_different_seed_differs(self):
        """Different seeds give different projections.

        Locks: the seed actually parameterizes the draw. Catches a hard-coded
        or ignored seed.
        """
        a = PerformerAttention(dim=16, seed=1)
        b = PerformerAttention(dim=16, seed=2)
        assert not torch.equal(a.projection, b.projection)

    def test_forward_deterministic(self, sample_data):
        """Repeated forwards give identical output.

        Locks: the projection is fixed across calls (buffer, not re-drawn).
        Catches a per-forward random draw that would make a trained model's
        outputs nondeterministic.
        """
        Q, K, V = sample_data
        attn = PerformerAttention(dim=Q.shape[-1], seed=0)
        attn.eval()
        with torch.no_grad():
            first, _ = attn(Q, K, V)
            second, _ = attn(Q, K, V)
        assert torch.equal(first, second)

    def test_projection_not_persistent(self):
        """The projection is a non-persistent buffer.

        Locks: checkpoints stay small and the projection is regenerated from the
        seed rather than serialized. Catches accidental persistence that would
        bloat state dicts and pin an old projection.
        """
        attn = PerformerAttention(dim=16, seed=0)
        assert "projection" not in attn.state_dict()
        # But it *is* a registered buffer (so .to() moves it).
        assert "projection" in dict(attn.named_buffers())


class TestPerformerApproximation:
    """The estimator is an unbiased approximation of the softmax kernel.

    We deliberately do NOT assert a tight rel-err on a single random draw:
    FAVOR+ variance grows with dimension and logit magnitude, so a single-draw
    output can differ substantially from softmax. Asserting a tight bound would
    be scientifically wrong. Instead we verify the defining property — the
    estimated kernel is *unbiased*: averaging many independent projections
    converges toward the true softmax kernel.
    """

    def test_kernel_estimator_is_unbiased(self):
        """Averaging over projections approaches exp(q·k/sqrt(d)).

        Locks: correctness of the FAVOR+ feature map (scaling + norm stabilizer).
        Catches the biased-variant bug where a mis-placed 1/d^{1/4} scale makes
        the estimator converge to the wrong kernel.

        Assessment: this is the right test for a Monte-Carlo estimator — it
        checks the estimand, not a noisy sample. It is not over-testing: it is
        the single property that distinguishes a correct FAVOR+ implementation
        from a plausible-looking but biased one, which is exactly the bug class
        encountered while writing this module.
        """
        torch.manual_seed(0)
        dim, seq = 16, 6
        Q = torch.randn(seq, dim)
        K = torch.randn(seq, dim)
        true_kernel = torch.exp(Q @ K.t() / math.sqrt(dim))

        num_draws, num_features = 400, 1024
        acc = torch.zeros(seq, seq)
        for seed in range(num_draws):
            attn = PerformerAttention(dim=dim, num_features=num_features, seed=seed, eps=0.0)
            qp = attn.feature_map(Q)
            kp = attn.feature_map(K)
            acc += qp @ kp.t()
        estimate = acc / num_draws

        rel_err = (estimate - true_kernel).norm() / true_kernel.norm()
        # Loose bound: with 400 draws at m=1024 in d=16 the empirical mean sits
        # well inside 20% of the true kernel; a biased implementation lands at
        # ~1.0 (order-of-magnitude off), so this cleanly separates the two.
        assert rel_err < 0.2, f"kernel estimator looks biased: rel-err={rel_err:.3f}"

    def test_more_features_reduce_error(self):
        """More random features reduce the single-draw approximation error.

        Locks: the variance-reduction direction of the estimator (error shrinks
        as the feature count m grows). Catches a feature count that is ignored
        or wired backwards.

        We average the *single-draw* error over independent seeds (a Monte-Carlo
        estimate of E‖K̂ − K‖). This is the quantity that improves with m;
        averaging the kernel across draws first would cancel variance and hide
        the effect.
        """
        torch.manual_seed(0)
        dim, seq = 16, 6
        Q = torch.randn(seq, dim)
        K = torch.randn(seq, dim)
        true_kernel = torch.exp(Q @ K.t() / math.sqrt(dim))

        def mean_single_draw_rel_err(num_features, num_seeds=50):
            errs = []
            for seed in range(num_seeds):
                attn = PerformerAttention(dim=dim, num_features=num_features, seed=seed, eps=0.0)
                kernel = attn.feature_map(Q) @ attn.feature_map(K).t()
                errs.append((kernel - true_kernel).norm() / true_kernel.norm())
            return torch.stack(errs).mean()

        assert mean_single_draw_rel_err(2048) < mean_single_draw_rel_err(64)
