#!/usr/bin/env python
"""Plot the m=1 vs m=4 (fresh vs cached) forward-latency comparison.

Answers one question from the anchor-cost investigation: does caching the
k-means anchors (reusing detached centroids across forwards) recover the
step-up in latency from the single mean anchor (``m=1``) to the four-anchor
k-means config (``m=4``)? The clustering that caching amortizes is the dominant
term at low anchor count, so this is where reuse should help most.

Reads a ``run_cuda_scaling.py`` result JSON that includes the three methods
``piecewise_mean`` (m=1), ``piecewise_kmeans_m4`` (m=4, fresh clustering every
forward) and ``piecewise_kmeans_m4_cached{N}`` (m=4, centroids reused, refreshed
every N forwards), and draws a grouped bar chart of forward latency, one group
per sequence length. The residual step that caching does *not* remove is the
``O(m·n·d²)`` second-moment einsum, which reuse cannot touch — so the m=4-cached
bar sitting above m=1 is the einsum floor, and the gap it closes below m=4-fresh
is the recovered clustering.

Examples
--------
::

    python benchmarks/harness/plot_anchor_cache.py \
        results/cuda_scaling_m4_cache_a100.json \
        --out results/anchor_cache/m1_vs_m4_cached.png
"""

import argparse
import json
import sys
from pathlib import Path


def _lat(rows, dim, dtype, n, method):
    """Forward latency (ms) for a non-causal cell, or None if missing/OOM."""
    for r in rows:
        if (
            r["d"] == dim
            and r["dtype"] == dtype
            and not r.get("causal", False)
            and r["n"] == n
            and r["method"] == method
        ):
            return r["latency_fwd_ms"] if r["status"] == "ok" else None
    return None


def _cached_method_name(methods):
    """The single ``piecewise_kmeans_m4_cached{N}`` method present, or None."""
    for m in methods:
        if m.startswith("piecewise_kmeans_m4_cached"):
            return m
    return None


def render(data, out_path, dim=None, dtype=None):
    config = data.get("config", {})
    rows = data.get("history", {}).get("rows", [])
    dim = dim or config.get("dims", [128])[-1]
    dtype = dtype or (
        "bf16" if "bf16" in config.get("dtypes", []) else config.get("dtypes", ["fp32"])[0]
    )
    cached_name = _cached_method_name(config.get("methods", []))
    if cached_name is None:
        raise SystemExit(
            "no piecewise_kmeans_m4_cached{N} method in the result; "
            "rerun the sweep including the cached variant"
        )
    refresh = cached_name.split("_cached", 1)[1]

    ns = sorted(
        {
            r["n"]
            for r in rows
            if r["d"] == dim and r["dtype"] == dtype and not r.get("causal", False)
        }
    )
    series = {
        "m=1 (mean anchor)": ("piecewise_mean", "#2ca02c"),
        "m=4 (fresh k-means)": ("piecewise_kmeans_m4", "#d62728"),
        f"m=4 (cached, refresh {refresh})": (cached_name, "#1f77b4"),
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_groups = len(ns)
    bar_w = 0.26
    x = range(n_groups)

    for i, (label, (method, color)) in enumerate(series.items()):
        vals = [_lat(rows, dim, dtype, n, method) for n in ns]
        positions = [xi + (i - 1) * bar_w for xi in x]
        # Missing cells (OOM) plot as zero-height with a hatch so they are visible.
        heights = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(
            positions, heights, bar_w, label=label, color=color, edgecolor="black", linewidth=0.5
        )
        for rect, v in zip(bars, vals):
            if v is None:
                continue
            ax.annotate(
                f"{v:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, v),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    # Annotate, per length, how much of the m1->m4 jump caching recovered.
    for xi, n in zip(x, ns):
        r1 = _lat(rows, dim, dtype, n, "piecewise_mean")
        r4 = _lat(rows, dim, dtype, n, "piecewise_kmeans_m4")
        r4c = _lat(rows, dim, dtype, n, cached_name)
        if None not in (r1, r4, r4c) and r4 > r1:
            recovered = (r4 - r4c) / (r4 - r1) * 100.0
            ax.annotate(
                f"cache recovers\n{recovered:.0f}% of jump",
                xy=(xi, r4),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#444",
                fontstyle="italic",
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"n={n}" for n in ns])
    # Headroom so the per-group recovery annotation never collides with the title.
    ymax = max(
        (r["latency_fwd_ms"] or 0.0)
        for r in rows
        if r["d"] == dim and r["dtype"] == dtype and not r.get("causal", False)
    )
    ax.set_ylim(top=ymax * 1.25)
    ax.set_ylabel("forward latency  (ms, lower = better)")
    ax.set_title(
        f"Does anchor caching overcome the m=1 → m=4 jump?\n"
        f"d={dim}, {dtype}, forward pass — cached recovers the recomputed "
        f"clustering, not the O(m·n·d²) einsum"
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json")
    parser.add_argument("--out", default="results/anchor_cache/m1_vs_m4_cached.png")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.result_json).read_text())
    print(f"wrote {render(data, args.out, args.dim, args.dtype)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
