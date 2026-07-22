#!/usr/bin/env python
"""Speed-vs-quality figure generator.

Reads the committed evidence JSON (``results/scaling.json``,
``results/microbench.json``, ``results/recall.json``) and renders a single
two-panel figure that places each attention mechanism by *speed* and by how
well it *performs*:

- Left panel: forward-pass speed vs softmax approximation fidelity. Fidelity is
  ``1 - relative_error`` against exact softmax on the microbenchmark (higher is
  a closer match). This answers "how faithfully does the linear-time method
  reproduce softmax?".
- Right panel: forward-pass speed vs associative-recall accuracy (8-seed mean,
  error bars = std). This answers "does the method solve the task?".

In both panels the x-axis is forward wall-time (ms, log scale) so *faster is to
the left*, and the top-left corner is the desirable region (fast and accurate).
The single mean-centroid anchor sits there in both panels; linear attention
sits at the bottom of the recall panel (it fails the task).

Speed is taken at a single representative sequence length (default 4096, the
longest measured) so every method is compared at one operating point; pass
``--speed-len`` to change it.

Examples
--------
::

    python benchmarks/harness/plot_results.py --out results/speed_vs_quality.png
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt

# Canonical method identity: display label, color, marker. The three benchmarks
# use slightly different keys for the same mechanism (e.g. the single mean anchor
# is ``piecewise_mean`` in scaling/microbench but ``piecewise`` in recall, and
# linear is ``linear``/``linear_elu``); the aliases map every key to one canon.
METHODS = {
    "standard": {"label": "softmax (exact)", "color": "#444444", "marker": "s"},
    "performer": {"label": "Performer", "color": "#d62728", "marker": "o"},
    "linear": {"label": "linear", "color": "#9467bd", "marker": "v"},
    "piecewise_mean": {"label": "piecewise (mean, m=1)", "color": "#1f77b4", "marker": "*"},
    "piecewise_kmeans": {"label": "piecewise (k-means, m=4)", "color": "#2ca02c", "marker": "D"},
    "piecewise_stride": {"label": "piecewise (stride, m=4)", "color": "#ff7f0e", "marker": "^"},
}
ALIASES = {"piecewise": "piecewise_mean", "linear_elu": "linear"}


def _canon(name):
    return ALIASES.get(name, name)


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _speed_by_method(scaling, speed_len):
    """Forward wall-time (ms) per method at the chosen sequence length."""
    rows = scaling["history"]["grid"]
    row = next((r for r in rows if r["seq_len"] == speed_len), None)
    if row is None:  # fall back to the longest measured length
        row = max(rows, key=lambda r: r["seq_len"])
    return {_canon(k): v for k, v in row["wall_ms"].items()}, row["seq_len"]


def _fidelity_by_method(microbench):
    """Softmax approximation fidelity (1 - mean rel-err over the grid) per method.

    Exact softmax is not in the microbenchmark (it is the reference), so it is
    added at fidelity 1.0 by definition.
    """
    rows = microbench["history"]["grid"]
    accum = {}
    for row in rows:
        for name, err in row["rel_err"].items():
            accum.setdefault(_canon(name), []).append(err)
    fidelity = {m: 1.0 - st.mean(errs) for m, errs in accum.items()}
    fidelity["standard"] = 1.0
    return fidelity


def _recall_by_method(recall):
    """Final recall accuracy (mean, std) across seeds per method."""
    accum = {}
    for run in recall["history"]["runs"]:
        m = _canon(run["config"]["attention_type"])
        accum.setdefault(m, []).append(run["metrics"]["accuracy"])
    return {m: (st.mean(a), st.pstdev(a) if len(a) > 1 else 0.0) for m, a in accum.items()}


def _scatter(ax, x_by_method, y_by_method, yerr_by_method=None):
    for m, style in METHODS.items():
        if m not in x_by_method or m not in y_by_method:
            continue
        x = x_by_method[m]
        y = y_by_method[m]
        yerr = (yerr_by_method or {}).get(m)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=style["marker"],
            color=style["color"],
            markersize=15 if style["marker"] == "*" else 9,
            capsize=3,
            label=style["label"],
            linestyle="none",
        )
        ax.annotate(
            style["label"],
            (x, y),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
            color=style["color"],
        )
    ax.set_xscale("log")  # smaller ms (faster) is naturally to the left
    ax.grid(True, which="both", alpha=0.25)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--speed-len", type=int, default=4096)
    parser.add_argument("--out", default="results/speed_vs_quality.png")
    args = parser.parse_args(argv)

    rdir = Path(args.results_dir)
    scaling = _load(rdir / "scaling.json")
    microbench = _load(rdir / "microbench.json")
    recall = _load(rdir / "recall.json")

    speed, speed_len = _speed_by_method(scaling, args.speed_len)
    fidelity = _fidelity_by_method(microbench)
    recall_acc = _recall_by_method(recall)
    recall_mean = {m: v[0] for m, v in recall_acc.items()}
    recall_std = {m: v[1] for m, v in recall_acc.items()}

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5))

    _scatter(ax_l, speed, fidelity)
    ax_l.set_xlabel(f"forward time at n={speed_len} (ms, log — faster ←)")
    ax_l.set_ylabel("softmax approximation fidelity (1 − rel-err)")
    ax_l.set_title("Speed vs approximation fidelity")

    _scatter(ax_r, speed, recall_mean, recall_std)
    ax_r.set_xlabel(f"forward time at n={speed_len} (ms, log — faster ←)")
    ax_r.set_ylabel("recall accuracy (8-seed mean ± std)")
    ax_r.set_title("Speed vs recall accuracy")
    ax_r.axhline(
        recall.get("metrics", {}).get("random_baseline", 0.0625),
        color="gray",
        linestyle="--",
        linewidth=1,
        label="chance",
    )

    fig.suptitle(
        "Fast AND faithful: the single mean-centroid anchor sits in the top-left " "of both panels",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
