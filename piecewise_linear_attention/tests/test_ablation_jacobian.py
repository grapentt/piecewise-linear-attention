"""Invariant tests for the analytic-Jacobian ablation microbench.

These lock the three properties the ablation's verdict depends on. They assert
structural invariants (constancy, non-triviality, equality of a shared path),
not point-value rel-err numbers, so they stay valid across backends.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from piecewise_linear_attention.core.attention import PiecewiseAttention

# Load the standalone harness script by path (it lives under benchmarks/, not
# the importable package).
_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "harness"
    / "run_ablation_jacobian.py"
)
_spec = importlib.util.spec_from_file_location("run_ablation_jacobian", _SCRIPT)
ablation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ablation)


@pytest.fixture
def sample_data():
    torch.manual_seed(0)
    batch, seq_len, dim = 4, 32, 16
    Q = torch.randn(batch, seq_len, dim)
    K = torch.randn(batch, seq_len, dim)
    V = torch.randn(batch, seq_len, dim)
    return Q, K, V


def test_nystrom_m1_constant_across_positions(sample_data):
    """The single-landmark surrogate outputs the same vector at every position.

    Locks: the m=1 Nyström variant is a degenerate constant-broadcast (the
    documented reason it equals zeroth-order and must not count as verdict
    evidence). Catches a regression that let it vary per query and be mistaken
    for a genuine baseline.
    """
    Q, K, V = sample_data
    out, _ = ablation._NystromM1(Q.shape[-1])(Q, K, V)
    first = out[:, :1, :]
    assert torch.allclose(out, first.expand_as(out), atol=1e-5)


def test_diagonal_only_changes_output(sample_data):
    """diagonal_only=True produces a different output than the full Jacobian.

    Locks: the ablation flag actually drops the mean-field term rather than
    being a silent no-op. Catches a wiring bug where the branch has no effect,
    which would make the rank-1 verdict vacuous.
    """
    Q, K, V = sample_data
    dim = Q.shape[-1]
    full = PiecewiseAttention(dim=dim, scale=True)
    nomf = PiecewiseAttention(dim=dim, scale=True, diagonal_only=True)
    out_full, _ = full(Q, K, V)
    out_nomf, _ = nomf(Q, K, V)
    assert not torch.allclose(out_full, out_nomf, atol=1e-6)


def test_zeroth_order_matches_shared_anchor_output(sample_data):
    """zeroth_order equals the full method's anchor output A(a) broadcast.

    Locks: variant A reuses piecewise_full's exact anchor and A(a) (no
    reimplementation), so the full-vs-zeroth gap isolates only the Jacobian.
    Catches drift between A's anchor path and the shipped one.
    """
    Q, K, V = sample_data
    dim = Q.shape[-1]
    z = ablation._ZerothOrder(dim)
    out_z, _ = z(Q, K, V)

    pa = z._pa
    a = pa.pseudo_query_fn(Q)
    out_a, _ = pa._compute_attention_and_jacobian(a, K, V)
    expected = out_a.unsqueeze(1).expand(-1, Q.shape[1], -1)
    assert torch.allclose(out_z, expected, atol=1e-6)
    # And it is constant across positions (pure anchor broadcast, no Jacobian).
    assert torch.allclose(out_z, out_z[:, :1, :].expand_as(out_z), atol=1e-6)
