#!/usr/bin/env python
"""Analytic-Jacobian ablation microbenchmark.

Isolates *what* buys the approximation accuracy in anchor-Taylor attention:
the analytic Jacobian, or merely the choice of anchor. We measure relative
error to exact softmax on random problems across per-head dimension, query
dispersion, and sequence length, comparing:

- ``zeroth_order``  A(a) only, no Jacobian (constant anchor output broadcast).
- ``piecewise_full`` A(a) + full analytic Jacobian (the method).
- ``no_meanfield``  A(a) + Jacobian with the mean-field term v̄ ⊗ k̄ dropped
                    (retains the full dense uncentered second moment — this is
                    an ablation of the softmax-covariance term, NOT a diagonal
                    approximation).
- ``nystrom_m1_degenerate`` single-landmark surrogate; algebraically equal to
                    ``zeroth_order`` — kept only as a degeneracy sanity line.
- ``stride_anchor`` full Jacobian around the first query instead of the
                    centroid; isolates anchor placement.

The comparisons are paired: within each grid cell every variant and the
softmax reference see the same (Q, K, V) per seed, so per-seed differences
support a signed-rank verdict on whether the Jacobian — and its mean-field
term — are load-bearing. The whole sweep is forward-only and runs on CPU/MPS.

Examples
--------
Smoke test::

    python benchmarks/harness/run_ablation_jacobian.py --smoke

Full run::

    python benchmarks/harness/run_ablation_jacobian.py --out results/ablation_jacobian.json
"""

import argparse
import math
import platform
import statistics
import sys
from pathlib import Path

# Allow running directly from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from piecewise_linear_attention.core.anchors import StrideAnchor
from piecewise_linear_attention.core.attention import PiecewiseAttention, StandardAttention
from piecewise_linear_attention.harness import get_device, set_seed
from piecewise_linear_attention.harness.results import ExperimentResult, save_result

# Guard against running against a stale installed/source copy that predates the
# diagonal_only ablation switch — otherwise the no_meanfield variant would
# silently equal piecewise_full and the verdict would be meaningless.
assert "diagonal_only" in PiecewiseAttention.__init__.__code__.co_varnames, (
    "PiecewiseAttention lacks the 'diagonal_only' parameter; the imported "
    "package is stale relative to this script."
)

# Honest, self-describing variant labels used in JSON keys, legends, stdout.
V_ZEROTH = "zeroth_order"
V_FULL = "piecewise_full"
V_NOMF = "no_meanfield"
V_NYST = "nystrom_m1_degenerate"
V_STRIDE = "stride_anchor"
VARIANT_NAMES = [V_ZEROTH, V_FULL, V_NOMF, V_NYST, V_STRIDE]

# Only cells where the first-order term actually moves the output count toward
# the pass/fail verdict; at tight dispersion Δq→0 so every variant collapses to
# the anchor output regardless of whether the Jacobian matters.
LARGE_DISPERSIONS = (1.0, 2.0)

# Verdict thresholds.
MARGIN_STRONG = 0.30  # relative margin for "B >> A"
MARGIN_KILL = 0.05  # below this at d=128 => Jacobian buys nothing (kill/pivot)
LIFT_RANK1 = 1.10  # no_meanfield/full rel-err ratio for "rank-1 load-bearing"
LIFT_RANK1_AT2 = 1.05  # same, restricted to dispersion 2.0
WINRATE_STRONG = 0.90  # fraction of seeds B must beat A
WINRATE_ORDER = 0.75  # weaker win-rate for placement ordering


def _rel_err(approx, reference):
    return (approx - reference).norm() / reference.norm()


def _make_problem(batch, seq_len, dim, dispersion, generator, device):
    """Queries clustered around a shared center with the given dispersion.

    Small dispersion => queries near each other; large dispersion => spread out
    (the boundary-bias regime where first-order approximations degrade).
    """
    center = torch.randn(batch, 1, dim, generator=generator)
    Q = center + dispersion * torch.randn(batch, seq_len, dim, generator=generator)
    K = torch.randn(batch, seq_len, dim, generator=generator)
    V = torch.randn(batch, seq_len, dim, generator=generator)
    return Q.to(device), K.to(device), V.to(device)


class _ZerothOrder:
    """A(a) broadcast over the sequence, sharing piecewise_full's anchor + A(a).

    Reuses the shipped attention/Jacobian routine and simply discards the
    Jacobian, so the anchor and anchor-output are bit-identical to the full
    method — the full-vs-zeroth difference isolates exactly the Jacobian term.
    """

    def __init__(self, dim):
        self._pa = PiecewiseAttention(dim=dim, scale=True)

    def to(self, device):
        self._pa = self._pa.to(device)
        return self

    def __call__(self, Q, K, V):
        a = self._pa.pseudo_query_fn(Q)  # (batch, dim) centroid
        out0, _ = self._pa._compute_attention_and_jacobian(a, K, V)  # (batch, dim)
        out = out0.unsqueeze(1).expand(-1, Q.shape[1], -1)
        return out, None


class _NystromM1:
    """Single-landmark Nyström surrogate at the query centroid.

    Algebraically equal to ``zeroth_order`` (one landmark collapses the softmax
    to the anchor output broadcast over the sequence). Kept only as a degeneracy
    line/sanity invariant, never as verdict evidence.
    """

    def __init__(self, dim):
        self._scale = 1.0 / math.sqrt(dim)

    def to(self, device):
        return self

    def __call__(self, Q, K, V):
        ell = Q.mean(dim=1, keepdim=True)  # (batch, 1, dim)
        left = torch.softmax((Q @ ell.transpose(-2, -1)) * self._scale, dim=-1)  # (b, n, 1)
        right = torch.softmax((ell @ K.transpose(-2, -1)) * self._scale, dim=-1) @ V  # (b, 1, d)
        return left @ right, None


def _build_variants(dim):
    """name -> duck-typed callable (Q, K, V) -> (out (batch, n, dim), aux)."""
    return {
        V_ZEROTH: _ZerothOrder(dim),
        V_FULL: PiecewiseAttention(dim=dim, scale=True),
        V_NOMF: PiecewiseAttention(dim=dim, scale=True, diagonal_only=True),
        V_NYST: _NystromM1(dim),
        V_STRIDE: PiecewiseAttention(dim=dim, scale=True, anchor_strategy=StrideAnchor(k=1)),
    }


def _paired_stats(diffs):
    """Signed-rank + effect-size summary of paired differences (positive => B better).

    Returns win_rate, mean, a bootstrap-free 95% CI via the normal approx on the
    mean, Wilcoxon-style sign test p-value, and Cohen's dz.
    """
    n = len(diffs)
    mean = statistics.fmean(diffs)
    wins = sum(1 for d in diffs if d > 0)
    win_rate = wins / n
    if n > 1:
        sd = statistics.stdev(diffs)
    else:
        sd = 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
    dz = mean / (sd + 1e-12)
    # Two-sided sign test p-value (exact binomial under H0: p(win)=0.5).
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return {
        "win_rate": win_rate,
        "mean": mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p": p,
        "dz": dz,
    }


def _seed_for(d, n, disp_idx, s):
    """Independent draws across d so 'holds at both d' is a real second check."""
    return 1000 * s + n + disp_idx * 7 + 100000 * (d == 128)


def run_grid(dims, lengths, dispersions, batch, seeds, device):
    rows = []
    for d in dims:
        variants = {k: v.to(device) for k, v in _build_variants(d).items()}
        reference = StandardAttention(dim=d).to(device)
        for n in lengths:
            for di, disp in enumerate(dispersions):
                per_seed = {name: [] for name in variants}
                for s in range(seeds):
                    set_seed(_seed_for(d, n, di, s))
                    gen = torch.Generator().manual_seed(d * 97 + s + 1)
                    Q, K, V = _make_problem(batch, n, d, disp, gen, device)
                    with torch.no_grad():
                        ref, _ = reference(Q, K, V)
                        for name, module in variants.items():
                            approx, _ = module(Q, K, V)
                            per_seed[name].append(_rel_err(approx, ref).item())
                rows.append(
                    {
                        "dim": d,
                        "seq_len": n,
                        "dispersion": disp,
                        "rel_err_mean": {k: statistics.fmean(v) for k, v in per_seed.items()},
                        "rel_err_std": {
                            k: (statistics.stdev(v) if len(v) > 1 else 0.0)
                            for k, v in per_seed.items()
                        },
                        "rel_err_seeds": per_seed,
                    }
                )
                summary = " ".join(
                    f"{k}={statistics.fmean(v):.3f}" for k, v in per_seed.items()
                )
                print(f"  d={d:3d} n={n:5d} disp={disp:.2f}  {summary}")
    return rows


def compute_verdict(rows, dims):
    """Paired verdict on large-dispersion cells; kill/pivot trigger at d=128."""
    metrics = {"per_dim": {}, "tail_error": {}}
    kill = False
    rank1_load_bearing = {}

    for d in dims:
        large_cells = [
            r for r in rows if r["dim"] == d and r["dispersion"] in LARGE_DISPERSIONS
        ]
        cell_results = []
        lifts = []
        lifts_at2 = []
        for r in large_cells:
            full = r["rel_err_seeds"][V_FULL]
            zeroth = r["rel_err_seeds"][V_ZEROTH]
            nomf = r["rel_err_seeds"][V_NOMF]
            stride = r["rel_err_seeds"][V_STRIDE]

            # B >> A : diff = rel_err(A) - rel_err(B), positive => B better.
            d_a = [z - f for z, f in zip(zeroth, full)]
            stats_a = _paired_stats(d_a)
            mean_a = statistics.fmean(zeroth)
            rel_margin = stats_a["mean"] / (mean_a + 1e-12)
            passes_a = (
                stats_a["mean"] > 0
                and rel_margin >= MARGIN_STRONG
                and stats_a["ci_lo"] > 0
                and stats_a["p"] < 0.05
                and stats_a["win_rate"] >= WINRATE_STRONG
            )

            # B > D : placement sensitivity (reported, not part of rank-1 verdict).
            d_d = [sd - f for sd, f in zip(stride, full)]
            stats_d = _paired_stats(d_d)
            passes_order = (
                stats_d["mean"] > 0
                and stats_d["ci_lo"] > 0
                and stats_d["p"] < 0.05
                and stats_d["win_rate"] >= WINRATE_ORDER
            )

            # rank-1 term: g = rel_err(no_meanfield) - rel_err(full).
            g = [nm - f for nm, f in zip(nomf, full)]
            stats_g = _paired_stats(g)
            lift = statistics.fmean(nomf) / (statistics.fmean(full) + 1e-12)
            lifts.append(lift)
            if r["dispersion"] == 2.0:
                lifts_at2.append(lift)

            cell_results.append(
                {
                    "seq_len": r["seq_len"],
                    "dispersion": r["dispersion"],
                    "rel_margin_vs_zeroth": rel_margin,
                    "stats_vs_zeroth": stats_a,
                    "passes_strong": passes_a,
                    "stats_vs_stride": stats_d,
                    "passes_placement": passes_order,
                    "rank1_lift": lift,
                    "stats_rank1_gap": stats_g,
                }
            )

        n_large = len(cell_results)
        n_pass = sum(1 for c in cell_results if c["passes_strong"])
        strong_majority = n_pass > n_large / 2 if n_large else False

        # rank-1 load-bearing verdict for this d.
        gaps_pos = all(c["stats_rank1_gap"]["mean"] > 0 for c in cell_results)
        gaps_sig = all(
            c["stats_rank1_gap"]["ci_lo"] > 0 and c["stats_rank1_gap"]["p"] < 0.05
            for c in cell_results
        )
        lift_ok = (statistics.fmean(lifts) >= LIFT_RANK1) if lifts else False
        lift_at2_ok = (statistics.fmean(lifts_at2) >= LIFT_RANK1_AT2) if lifts_at2 else False
        rank1 = bool(gaps_pos and gaps_sig and lift_ok and lift_at2_ok)
        rank1_load_bearing[d] = rank1

        # Kill/pivot trigger: only meaningful at d=128.
        if d == 128:
            fails_ci = any(c["stats_vs_zeroth"]["ci_lo"] <= 0 for c in cell_results)
            fails_margin = (
                sum(1 for c in cell_results if c["rel_margin_vs_zeroth"] < MARGIN_KILL)
                > n_large / 2
                if n_large
                else False
            )
            if (not strong_majority) and (fails_ci or fails_margin):
                kill = True

        metrics["per_dim"][d] = {
            "cells": cell_results,
            "strong_majority_pass": strong_majority,
            "rank1_load_bearing": rank1,
            "mean_rank1_lift": statistics.fmean(lifts) if lifts else None,
        }

    # tail-error diagnostic on the largest cell.
    big = [r for r in rows if r["dim"] == max(dims) and r["dispersion"] == 2.0]
    if big:
        r = max(big, key=lambda x: x["seq_len"])
        metrics["tail_error"] = {
            "dim": r["dim"],
            "seq_len": r["seq_len"],
            "dispersion": r["dispersion"],
            "full_mean": statistics.fmean(r["rel_err_seeds"][V_FULL]),
            "no_meanfield_mean": statistics.fmean(r["rel_err_seeds"][V_NOMF]),
        }

    overall_pass = all(
        metrics["per_dim"][d]["strong_majority_pass"] for d in dims
    ) and not kill
    metrics["overall_pass"] = overall_pass
    metrics["kill_pivot"] = kill
    metrics["rank1_load_bearing"] = rank1_load_bearing
    return metrics


def make_figures(rows, dims, dispersions, out_path):
    # matplotlib is a benchmark-only dependency; import lazily so the module
    # (and its tested helpers) load without it.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    # F1: rel-err vs dispersion, one column per d, faceted by n, line per variant.
    lengths = sorted({r["seq_len"] for r in rows})
    fig1, axes = plt.subplots(
        len(lengths), len(dims), figsize=(5 * len(dims), 3.2 * len(lengths)), squeeze=False
    )
    for ci, d in enumerate(dims):
        for ri, n in enumerate(lengths):
            ax = axes[ri][ci]
            for name in VARIANT_NAMES:
                xs, ys, es = [], [], []
                for disp in dispersions:
                    cell = next(
                        (
                            r
                            for r in rows
                            if r["dim"] == d and r["seq_len"] == n and r["dispersion"] == disp
                        ),
                        None,
                    )
                    if cell is None:
                        continue
                    xs.append(disp)
                    ys.append(cell["rel_err_mean"][name])
                    es.append(cell["rel_err_std"][name])
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2, label=name)
            ax.set_xscale("log")
            ax.set_title(f"d={d}, n={n}")
            ax.set_xlabel("query dispersion")
            ax.set_ylabel("rel. error to softmax")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7)
    fig1.suptitle("Analytic-Jacobian ablation: relative error vs dispersion")
    fig1.tight_layout()
    p1 = Path(out_path).with_name("ablation_jacobian_error.png")
    fig1.savefig(p1, dpi=120)
    plt.close(fig1)
    figs.append(str(p1))
    return figs


def _print_banner(metrics):
    print("\n" + "=" * 68)
    if metrics["kill_pivot"]:
        print("KILL/PIVOT SIGNAL: the analytic first-order expansion provides no")
        print("numerical benefit over the constant anchor output at d=128.")
        print("(Scope: approximation error on random untrained problems.)")
    elif metrics["overall_pass"]:
        print("PASS: the Jacobian is load-bearing at both d (B >> A).")
    else:
        print("FAIL: 'B >> A' not established at both d (not a kill unless d=128).")
    for d, v in metrics["per_dim"].items():
        lb = v["rank1_load_bearing"]
        lift = v["mean_rank1_lift"]
        lift_s = f"{lift:.3f}" if lift is not None else "n/a"
        print(
            f"  d={d}: strong-pass={v['strong_majority_pass']} "
            f"rank1_load_bearing={lb} mean_lift={lift_s}"
        )
    print("=" * 68)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 4096])
    parser.add_argument("--dispersions", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.dims = [64]
        args.lengths = [64]
        args.dispersions = [0.1, 1.0]
        args.batch = 4
        args.seeds = 2

    device = get_device(args.device)
    print(f"device: {device}")

    rows = run_grid(args.dims, args.lengths, args.dispersions, args.batch, args.seeds, device)
    assert rows, "no grid rows produced"
    assert all(
        e >= 0.0 for r in rows for v in r["rel_err_seeds"].values() for e in v
    ), "negative relative error encountered"

    metrics = compute_verdict(rows, args.dims)
    _print_banner(metrics)

    if args.smoke:
        print("smoke OK")
        return 0

    figures = make_figures(rows, args.dims, args.dispersions, args.out or "ablation_jacobian.json")
    result = ExperimentResult(
        config={
            "benchmark": "ablation_jacobian",
            "dims": args.dims,
            "lengths": args.lengths,
            "dispersions": args.dispersions,
            "batch": args.batch,
            "seeds": args.seeds,
            "variants": VARIANT_NAMES,
        },
        metrics=metrics,
        environment={
            "device": str(device),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "figures": figures,
        },
        history={"grid": rows},
    )
    if args.out:
        path = save_result(result, args.out)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
