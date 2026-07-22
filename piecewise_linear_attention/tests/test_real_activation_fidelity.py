"""Invariant tests for the real-activation fidelity probe.

These lock the two properties the probe's verdict depends on: the per-query
error metric and the verdict fork logic. They run on synthetic tensors and the
public ``build_attention`` contract only — no transformers/datasets/matplotlib —
so they execute under the base test environment (those are benchmark-only
extras). Assertions are on orderings and verdict strings, not point-value
rel-err numbers, so they stay valid across backends.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

# Load the standalone harness script by path (it lives under benchmarks/, not
# the importable package).
_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "harness"
    / "run_real_activation_fidelity.py"
)
_spec = importlib.util.spec_from_file_location("run_real_activation_fidelity", _SCRIPT)
raf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(raf)


def test_per_query_rel_err_zero_and_ordering():
    """Per-query rel-err is ~0 for identical tensors and monotone in quantile.

    Locks: the metric normalizes per query row (over the feature axis) and its
    percentiles are ordered p99 >= p95 >= median >= 0.
    Catches: a broadcast/axis bug or swapped-quantile that would silently defeat
    the p95/p99 tail guards the verdict relies on.
    Over-testing assessment: minimal and warranted — this is the headline metric
    the entire probe rests on; the test asserts structure only, no magnitudes.
    """
    torch.manual_seed(0)
    x = torch.randn(6, 40, 16)
    summ_same, _ = raf.per_query_rel_err(x, x)
    assert summ_same["mean"] < 1e-5
    assert summ_same["p99"] < 1e-5

    approx = x + 0.3 * torch.randn_like(x)
    summ, vec = raf.per_query_rel_err(approx, x)
    assert vec.numel() == 6 * 40  # one entry per query row, not per element
    assert 0.0 <= summ["median"] <= summ["p95"] <= summ["p99"]
    assert all(v == v for v in summ.values())  # finite (no NaN)


def _row(method, seq_len, layer, mean, *, p95=None, p99=None, is_real=False,
         skipped=False):
    """Minimal grid row for verdict tests."""
    return {
        "method": method, "seq_len": seq_len, "layer": layer,
        "rel_err_mean": mean,
        "rel_err_median": mean, "rel_err_p95": p95 if p95 is not None else mean,
        "rel_err_p99": p99 if p99 is not None else mean,
        "is_real": is_real, "skipped": skipped,
    }


def test_verdict_forks_and_proxy_gate():
    """The verdict resolves KILL/SCOPE/PASS branches and gates the proxy.

    Locks: (i) a proxy (is_real=False) stress result can only ever reach
    PASS_PROXY_ONLY, never a bare PASS; the SAME numbers with is_real=True do
    reach PASS; (ii) SCOPE_GROW_M fires when m=1 fails the ceiling but a larger
    m recovers; (iii) SCOPE_RESTRICT_N fires when a shorter length is fine but
    the stress length is not; (iv) never beating any baseline routes to KILL,
    not silently to FAIL.
    Catches: an inverted/short-circuited predicate that always PASSes (the
    rubber-stamp failure mode) and a proxy silently minting a clean pass.
    Over-testing assessment: essential given the probe's mandate that it be able
    to KILL or SCOPE the long-context claim; asserts verdict strings only.
    """
    lengths = [512, 8192]

    # Good numbers under a proxy stress block -> PASS_PROXY_ONLY, not PASS.
    good = [
        _row(raf.M_PW1, 8192, 2, 0.30, p95=0.55, p99=0.70, is_real=False),
        _row(raf.M_NYS64, 8192, 2, 0.45),
        _row(raf.M_NYS256, 8192, 2, 0.42),
        _row(raf.M_PERF256, 8192, 2, 0.60),
        _row(raf.M_PERF512, 8192, 2, 0.58),
    ]
    assert raf.compute_verdict(good, lengths)["verdict"] == "PASS_PROXY_ONLY"

    # Identical numbers but a genuinely native stress block -> PASS.
    native = [dict(r, is_real=True) for r in good]
    assert raf.compute_verdict(native, lengths)["verdict"] == "PASS"

    # m=1 over the ceiling, m=16 recovers and beats baselines -> SCOPE_GROW_M.
    grow = [
        _row(raf.M_PW1, 8192, 2, 0.62, p95=0.80),
        _row(raf.M_PW16, 8192, 2, 0.40),
        _row(raf.M_NYS64, 8192, 2, 0.55),
        _row(raf.M_NYS256, 8192, 2, 0.50),
    ]
    assert raf.compute_verdict(grow, lengths)["verdict"] == "SCOPE_GROW_M"

    # Short length fine (piecewise beats baseline, under ceiling); stress length
    # over the ceiling with no m recovering, though still competitive with the
    # baseline there -> SCOPE_RESTRICT_N (not KILL, since it does beat a baseline).
    restrict = [
        _row(raf.M_PW1, 512, 2, 0.30),
        _row(raf.M_NYS64, 512, 2, 0.40),
        _row(raf.M_PW1, 8192, 2, 0.62, p95=0.85),
        _row(raf.M_PW4, 8192, 2, 0.60),
        _row(raf.M_PW16, 8192, 2, 0.58),
        _row(raf.M_PW64, 8192, 2, 0.56),
        _row(raf.M_NYS64, 8192, 2, 0.61),
    ]
    assert raf.compute_verdict(restrict, lengths)["verdict"] == "SCOPE_RESTRICT_N"

    # Piecewise never beats the baseline at any m, but stays under ceiling ->
    # KILL (not competitive long), not FAIL.
    never = [
        _row(raf.M_PW1, 8192, 2, 0.45),
        _row(raf.M_PW4, 8192, 2, 0.44),
        _row(raf.M_PW16, 8192, 2, 0.43),
        _row(raf.M_PW64, 8192, 2, 0.42),
        _row(raf.M_NYS64, 8192, 2, 0.20),
        _row(raf.M_NYS256, 8192, 2, 0.18),
    ]
    assert raf.compute_verdict(never, lengths)["verdict"] == "KILL"


def test_worst_layer_aggregation_excludes_skipped():
    """Aggregation keys on the worst (highest-error) non-skipped layer.

    Locks: an easy-layer cherry-pick cannot flatter the verdict, and skipped
    (OOM) cells are excluded rather than counted as a zero.
    Catches: a mean-over-layers regression, or a skipped cell being read as a
    valid low-error result (which could fake SCOPE_GROW_M on absent data).
    Over-testing assessment: borderline but kept — it guards two distinct
    cheat vectors (layer cherry-pick, absent-cell recovery) with one small test.
    """
    rows = [
        _row(raf.M_PW1, 8192, 2, 0.30),
        _row(raf.M_PW1, 8192, 3, 0.55),  # worst layer
        _row(raf.M_PW1, 8192, 4, float("nan"), skipped=True),
    ]
    r = raf._worst_layer_row(rows, raf.M_PW1, 8192)
    assert r["layer"] == 3
