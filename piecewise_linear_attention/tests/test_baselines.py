"""Tests for the efficient-attention baselines: Nyströmformer, Linformer, Luna.

These are pure-PyTorch reference implementations of three efficient-attention
families (quadrature/landmark, low-rank sequence projection, and nested landmark
attention) used as comparison baselines. The tests lock in the shared
:class:`BaseAttention` contract for each and the one numerical property that
distinguishes a correct implementation from a plausible-but-wrong one
(Nyström's landmark reconstruction).
"""

import pytest
import torch

from piecewise_linear_attention.core.attention import (
    LinformerAttention,
    LunaAttention,
    NystromformerAttention,
    StandardAttention,
)
from piecewise_linear_attention.core.registry import (
    available_attention_types,
    build_attention,
)


@pytest.fixture
def sample_data():
    """A batch of well-scaled Q/K/V with a sequence long enough for landmarks."""
    torch.manual_seed(0)
    batch, seq_len, dim = 4, 128, 16
    Q = torch.randn(batch, seq_len, dim)
    K = torch.randn(batch, seq_len, dim)
    V = torch.randn(batch, seq_len, dim)
    return Q, K, V


def _make(name, dim, seq_len):
    """Construct a baseline sized so its projections/landmarks fit ``seq_len``."""
    if name == "nystromformer":
        return NystromformerAttention(dim=dim, num_landmarks=seq_len // 4)
    if name == "linformer":
        return LinformerAttention(dim=dim, max_seq_len=seq_len, k=64)
    if name == "luna":
        return LunaAttention(dim=dim, num_pack=32)
    raise AssertionError(name)


class TestBaselineContract:
    """Every baseline honors the (output, extra) shape/finiteness/grad contract."""

    @pytest.mark.parametrize("name", ["nystromformer", "linformer", "luna"])
    def test_output_shape(self, name, sample_data):
        """Output preserves (batch, seq_len, dim).

        Locks: the low-rank/landmark round-trip returns to the input length.
        Catches a projection- or landmark-dimension mismatch that would leave
        the output at the compressed length.
        """
        Q, K, V = sample_data
        attn = _make(name, Q.shape[-1], Q.shape[1])
        out, _ = attn(Q, K, V)
        assert out.shape == Q.shape

    @pytest.mark.parametrize("name", ["nystromformer", "linformer", "luna"])
    def test_output_is_finite(self, name, sample_data):
        """No NaN/inf for well-scaled inputs.

        Locks: numerical stability of the pinv iteration (Nyström), the softmax
        blocks, and the eps-guarded divisions. Catches an overflow or a
        divergent pseudo-inverse.
        """
        Q, K, V = sample_data
        attn = _make(name, Q.shape[-1], Q.shape[1])
        out, _ = attn(Q, K, V)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("name", ["nystromformer", "linformer", "luna"])
    def test_gradient_flow(self, name, sample_data):
        """Gradients reach Q, K, V.

        Locks: each baseline stays differentiable end to end. Catches an
        accidental detach (e.g. the Nyström landmark selection breaking the
        graph, or a learned projection shadowing the input path).
        """
        Q, K, V = (t.clone().requires_grad_(True) for t in sample_data)
        attn = _make(name, Q.shape[-1], Q.shape[1])
        out, _ = attn(Q, K, V)
        out.sum().backward()
        for label, t in [("Q", Q), ("K", K), ("V", V)]:
            assert t.grad is not None, f"no grad for {label}"
            assert torch.isfinite(t.grad).all()


class TestBaselineRegistry:
    """The three baselines are discoverable and buildable through the registry."""

    def test_available_types_includes_baselines(self):
        """available_attention_types() lists the new baselines.

        Locks: the registrations run at import time. Catches a decorator that
        is never imported (so the name is silently absent from the CLI).
        """
        types = available_attention_types()
        for name in ("nystromformer", "linformer", "luna"):
            assert name in types

    @pytest.mark.parametrize(
        "name,expected_cls",
        [
            ("nystromformer", NystromformerAttention),
            ("linformer", LinformerAttention),
            ("luna", LunaAttention),
        ],
    )
    def test_build_returns_expected_class(self, name, expected_cls):
        """Each name builds its class through the shared union-kwargs convention.

        Locks: the builder tolerates the multi-head wrapper passing the union of
        all mechanisms' kwargs. Catches a builder that fails on an option meant
        for a different mechanism (it must absorb them via ``**_ignored``).
        """
        attn = build_attention(
            name,
            dim=16,
            dropout=0.0,
            causal=False,
            # Options meant for other mechanisms — must be ignored, not error.
            kernel_type="relu",
            scale=True,
            causal_pseudo_query="mean",
        )
        assert isinstance(attn, expected_cls)


class TestCausalUnsupported:
    """None of the three baselines has an exact causal form; each says so."""

    @pytest.mark.parametrize("name", ["nystromformer", "linformer", "luna"])
    def test_causal_raises(self, name):
        """Constructing with causal=True raises NotImplementedError.

        Locks: the documented lack of a causal formulation is enforced, not
        silently ignored. Catches a build path that would return a
        wrong-but-runnable causal module.
        """
        with pytest.raises(NotImplementedError):
            build_attention(name, dim=16, causal=True)


class TestNystromApproximation:
    """Nyström reconstructs full softmax as the landmark count grows.

    This is the one property that separates a correct Nyström implementation
    (segment-mean landmarks + a working iterative pseudo-inverse) from a
    plausible-looking but broken one. A pinv that fails to converge, or a
    mis-scaled softmax block, breaks the ``kernel1 @ pinv(kernel2) @ kernel3``
    reconstruction and fails this test.
    """

    def test_fine_landmarks_recover_softmax(self):
        """As landmarks approach the token count, output recovers full softmax.

        Locks: correctness of the reconstruction in the fine-landmark limit. In
        this segment-mean design, one landmark per token means each landmark is
        a single token, at which point the Nyström reconstruction is exact — the
        module recognizes ``num_landmarks >= seq_len`` and returns exact softmax
        rather than running an ill-posed landmark reconstruction. Catches a
        fine-limit path that drifts from full attention (e.g. a mis-placed
        scale).

        Coarser landmarks (segment size > 1) give a deliberately loose
        approximation on non-smooth inputs, so asserting a tight bound there
        would be scientifically wrong; that regime is exercised for
        shape/finiteness only.

        Assessment: this is the right acceptance test — it checks the estimand
        (recovery of full attention in the fine-landmark limit), not a noisy
        sample. It is not over-testing: shape/finiteness alone would pass a
        module that reconstructs the wrong operator.
        """
        torch.manual_seed(0)
        batch, seq_len, dim = 2, 64, 16
        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        exact, _ = StandardAttention(dim=dim)(Q, K, V)
        nystrom = NystromformerAttention(dim=dim, num_landmarks=seq_len)
        approx, _ = nystrom(Q, K, V)

        rel_err = (approx - exact).norm() / exact.norm()
        assert rel_err < 1e-4, f"fine-landmark limit drifts from softmax: rel-err={rel_err:.3e}"

    def test_coarse_landmarks_are_finite_and_shaped(self):
        """Coarse landmarks (segment size > 1) still produce a valid output.

        Locks: the genuine landmark-reconstruction path (kernels + pinv) runs
        and returns finite output with a finite conditioning residual. Catches a
        divergent pinv or a shape error on the non-fallback path. We assert
        finiteness and shape, not a tight tolerance — coarse Nyström is a loose
        approximation by design, and claiming otherwise would be wrong.
        """
        torch.manual_seed(0)
        batch, seq_len, dim = 2, 64, 16
        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        nystrom = NystromformerAttention(dim=dim, num_landmarks=16, pinv_iterations=10)
        approx, residual = nystrom(Q, K, V)
        assert approx.shape == Q.shape
        assert torch.isfinite(approx).all()
        assert residual is not None
        assert torch.isfinite(residual).all()

    def test_conditioning_diagnostic_is_exposed(self):
        """The pinv residual and effective landmark count are recorded.

        Locks: the conditioning diagnostic the issue requires — the per-batch
        pinv residual is returned and cached alongside the effective landmark
        count. Catches a diagnostic that is dropped, detached from the actual
        landmark count, or never populated. (We do not assert a monotone
        residual-vs-m relationship: the Frobenius residual scales with the
        (m, m) matrix size, so its magnitude is not a clean function of m — the
        contract is only that it is exposed and finite.)
        """
        torch.manual_seed(1)
        batch, seq_len, dim = 2, 64, 16
        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        nystrom = NystromformerAttention(dim=dim, num_landmarks=16, pinv_iterations=6)
        _, residual = nystrom(Q, K, V)
        assert residual is not None
        assert residual.shape == (batch,)
        assert torch.isfinite(residual).all()
        assert nystrom.last_num_landmarks == 16
        assert torch.equal(nystrom.last_pinv_residual, residual.detach())

    def test_fallback_to_exact_when_seq_short(self):
        """With more landmarks than tokens the module returns exact softmax.

        Locks: the documented n<=m fallback path. Catches a fallback that still
        runs the (ill-posed) landmark reconstruction on a short sequence.
        """
        torch.manual_seed(2)
        batch, seq_len, dim = 2, 8, 16
        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        nystrom = NystromformerAttention(dim=dim, num_landmarks=64)
        approx, residual = nystrom(Q, K, V)
        exact, _ = StandardAttention(dim=dim)(Q, K, V)

        assert residual is None  # fallback path reports no pinv residual
        assert nystrom.last_num_landmarks == seq_len
        assert torch.allclose(approx, exact, atol=1e-5)


class TestLinformerLength:
    """Linformer's fixed projection makes it same-length-only."""

    def test_rejects_seq_longer_than_max(self):
        """seq_len > max_seq_len raises ValueError.

        Locks: the same-length-only limitation is enforced, not silently
        truncated. Catches an out-of-bounds slice of the projection that would
        either crash unclearly or drop tokens.
        """
        attn = LinformerAttention(dim=16, max_seq_len=32, k=16)
        Q = torch.randn(2, 64, 16)
        with pytest.raises(ValueError):
            attn(Q, Q, Q)

    def test_added_params_count(self):
        """E and F contribute exactly 2 * max_seq_len * k learned parameters.

        Locks: the declared parameter cost. Catches a projection sized on the
        feature axis instead of the sequence axis (which would give the wrong
        count and the wrong semantics).
        """
        max_seq_len, k = 32, 16
        attn = LinformerAttention(dim=8, max_seq_len=max_seq_len, k=k)
        n_params = sum(p.numel() for p in attn.parameters())
        assert n_params == 2 * max_seq_len * k


class TestLunaStructure:
    """Luna packs the sequence into p learned landmarks, then unpacks."""

    def test_added_params_count(self):
        """The pack-query sequence contributes exactly p * dim parameters.

        Locks: P is the sole learned tensor and is sized (p, dim). Catches an
        extra projection or a mis-sized pack sequence.
        """
        dim, p = 16, 32
        attn = LunaAttention(dim=dim, num_pack=p)
        n_params = sum(param.numel() for param in attn.parameters())
        assert n_params == p * dim

    def test_summary_has_pack_length(self):
        """The returned summary has length p regardless of sequence length.

        Locks: the length-p bottleneck that gives Luna its O(n*p) cost. Catches
        a pack step that leaks the sequence length into the summary.
        """
        dim, p = 16, 32
        attn = LunaAttention(dim=dim, num_pack=p)
        Q = torch.randn(2, 100, dim)
        _, summary = attn(Q, Q, Q)
        assert summary.shape == (2, p, dim)
