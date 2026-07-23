#!/usr/bin/env python
"""Render the anchor-count sweep as two figures: accuracy-vs-``m`` and speed-vs-``m``.

Reads the ``cuda_scaling`` result JSON produced by ``run_cuda_scaling.py`` with an
anchor sweep (``--anchors``) and real-activation accuracy (``--acc-real
--acc-layers``), and produces:

- **Accuracy** (``*_accuracy.png``): x = anchor count ``m``, y = output relative
  error to exact softmax on real trained-model activations, one curve per captured
  layer (early/mid/late), with a horizontal reference line per baseline (its own
  real-activation rel-err). Lower is better; ``m`` is the piecewise dial.
- **Speed** (``*_speed.png``): x = anchor count ``m``, y = forward latency (ms), one
  curve per sequence length ``n`` (colour-coded), with a horizontal reference line
  per baseline per ``n`` (so each baseline contributes one line at each ``n``).
  Points that ran out of memory are omitted from the curve and annotated as an OOM
  ceiling marker so the fit boundary is visible rather than silently dropped.

Both figures fix a single per-head ``d`` (annotated); if the sweep covered more than
one ``d`` a separate pair of figures is written per ``d``. The non-anchor baselines
have no ``m`` axis, so they appear only as horizontal references.

Examples
--------
::

    python benchmarks/harness/plot_anchor_sweep.py \
        results/cuda_scaling_anchors_a100.json --out-prefix results/anchor_sweep

    # schema smoke (no GPU, uses the crafted sample):
    python benchmarks/harness/plot_anchor_sweep.py --self-check
"""

import argparse
import json
import sys
from pathlib import Path

# Non-anchor baselines drawn as horizontal reference lines. ``standard`` and
# ``sdpa_math`` are the exact-softmax references (rel-err 0); they are the OOM
# marker on the speed plot rather than a latency line where they do not fit.
_BASELINES = ["performer", "nystromformer", "linformer", "luna", "linear",
              "standard", "sdpa_math"]
_SOFTMAX = {"standard", "sdpa_math"}


def _anchor_count(method):
    """``m`` for a piecewise method name (``piecewise_mean`` -> 1), else None."""
    if method == "piecewise_mean":
        return 1
    prefix = "piecewise_kmeans_m"
    if method.startswith(prefix):
        try:
            return int(method[len(prefix):])
        except ValueError:
            return None
    return None


def _speed_dtype(config):
    """Latency dtype to plot: bf16 if swept (inference precision), else the first."""
    dtypes = config.get("dtypes", ["bf16"])
    return "bf16" if "bf16" in dtypes else dtypes[0]


def _collect_speed(rows, dim, dtype):
    """Non-causal forward latency, keyed for the speed plot.

    Returns ``(anchor_lat, baseline_lat, oom_by_n)`` where ``anchor_lat[n]`` is a
    sorted ``[(m, ms), ...]`` for the piecewise dial, ``baseline_lat[method][n]`` is
    a scalar ms, and ``oom_by_n[n]`` is the smallest ``m`` that OOMed (or None)."""
    anchor_lat = {}
    baseline_lat = {}
    oom_m = {}
    for r in rows:
        if r["d"] != dim or r["dtype"] != dtype or r.get("causal", False):
            continue
        n = r["n"]
        m = _anchor_count(r["method"])
        if m is not None:
            if r["status"] == "ok" and r["latency_fwd_ms"] is not None:
                anchor_lat.setdefault(n, []).append((m, r["latency_fwd_ms"]))
            elif r["status"] == "oom":
                if n not in oom_m or m < oom_m[n]:
                    oom_m[n] = m
        elif r["method"] in _BASELINES and r["status"] == "ok" \
                and r["latency_fwd_ms"] is not None:
            baseline_lat.setdefault(r["method"], {})[n] = r["latency_fwd_ms"]
    for n in anchor_lat:
        anchor_lat[n].sort()
    return anchor_lat, baseline_lat, oom_m


def _collect_accuracy(accuracy_block):
    """Per-layer accuracy for the accuracy plot.

    Returns ``(anchor_acc, baseline_acc)`` where ``anchor_acc[layer]`` is a sorted
    ``[(m, rel_err), ...]`` and ``baseline_acc[method]`` is the mean rel-err across
    layers (a single reference line per baseline; the layer spread lives in the
    anchor curves)."""
    anchor_acc = {}
    baseline_vals = {}
    for pl in accuracy_block.get("per_layer", []):
        layer = pl["layer"]
        for ar in pl.get("rows", []):
            if ar.get("rel_err") is None:
                continue
            m = _anchor_count(ar["method"])
            if m is not None:
                anchor_acc.setdefault(layer, []).append((m, ar["rel_err"]))
            elif ar["method"] in _BASELINES and ar["method"] not in _SOFTMAX:
                baseline_vals.setdefault(ar["method"], []).append(ar["rel_err"])
    for layer in anchor_acc:
        anchor_acc[layer].sort()
    baseline_acc = {mth: sum(v) / len(v) for mth, v in baseline_vals.items() if v}
    return anchor_acc, baseline_acc


def _plot_accuracy(anchor_acc, baseline_acc, dim, model_id, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")
    layers = sorted(anchor_acc)
    for i, layer in enumerate(layers):
        ms = [m for m, _ in anchor_acc[layer]]
        errs = [e for _, e in anchor_acc[layer]]
        colour = cmap(0.15 + 0.7 * (i / max(1, len(layers) - 1)))
        ax.plot(ms, errs, "-o", color=colour, label=f"piecewise, layer {layer}",
                linewidth=2, markersize=5)

    base_cmap = plt.get_cmap("tab10")
    for j, (mth, err) in enumerate(sorted(baseline_acc.items())):
        ax.axhline(err, color=base_cmap(j % 10), linestyle="--", linewidth=1.3,
                   alpha=0.8, label=f"{mth} (baseline)")

    ax.set_xlabel("anchor count  m")
    ax.set_ylabel("relative error to exact softmax  (lower = better)")
    ax.set_title(f"Accuracy vs anchor count — real activations "
                 f"({model_id}), d={dim}")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_speed(anchor_lat, baseline_lat, oom_m, dim, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ns = sorted(anchor_lat)
    n_cmap = {n: c for n, c in zip(ns, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])}

    for n in ns:
        ms = [m for m, _ in anchor_lat[n]]
        lat = [ms_ for _, ms_ in anchor_lat[n]]
        ax.plot(ms, lat, "-o", color=n_cmap[n], linewidth=2, markersize=5,
                label=f"piecewise, n={n}")
        # OOM ceiling marker: the smallest m that failed at this n.
        if n in oom_m:
            ax.axvline(oom_m[n], color=n_cmap[n], linestyle=":", linewidth=1.2,
                       alpha=0.7)
            top = max(lat) if lat else 1.0
            ax.annotate(f"OOM ≥ m={oom_m[n]} (n={n})",
                        xy=(oom_m[n], top), fontsize=7, color=n_cmap[n],
                        rotation=90, va="top", ha="right")

    # Baseline horizontal lines: one per baseline per n.
    base_cmap = plt.get_cmap("tab10")
    for j, mth in enumerate(sorted(baseline_lat)):
        for n in ns:
            if n in baseline_lat[mth]:
                ax.axhline(baseline_lat[mth][n], color=base_cmap(j % 10),
                           linestyle="--" if n == ns[0] else ":", linewidth=1.1,
                           alpha=0.7,
                           label=f"{mth} (n={n})")

    ax.set_xlabel("anchor count  m")
    ax.set_ylabel("forward latency  (ms, lower = better)")
    ax.set_title(f"Speed vs anchor count — d={dim}, "
                 f"batch·heads folded, forward pass")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render(data, out_prefix):
    config = data.get("config", {})
    hist = data.get("history", {})
    rows = hist.get("rows", [])
    accuracy = hist.get("accuracy", {})
    dims = config.get("dims", [128])
    speed_dtype = _speed_dtype(config)
    written = []
    for dim in dims:
        anchor_lat, baseline_lat, oom_m = _collect_speed(rows, dim, speed_dtype)
        acc_block = accuracy.get(str(dim), {})
        model_id = acc_block.get("model_id", config.get("acc_model", "real"))
        anchor_acc, baseline_acc = _collect_accuracy(acc_block)

        if anchor_acc:
            p = _plot_accuracy(anchor_acc, baseline_acc, dim, model_id,
                               f"{out_prefix}_d{dim}_accuracy.png")
            written.append(p)
        else:
            print(f"d={dim}: no accuracy data; skipping accuracy plot")
        if anchor_lat:
            p = _plot_speed(anchor_lat, baseline_lat, oom_m, dim,
                            f"{out_prefix}_d{dim}_speed.png")
            written.append(p)
        else:
            print(f"d={dim}: no speed data; skipping speed plot")
    return written


def _self_check():
    """Render the crafted schema sample so the parser/plot paths are exercised
    without a GPU run. Writes to a temp prefix and asserts both figures appear."""
    sample = Path("/tmp/plot_schema_sample.json")
    if not sample.exists():
        print("self-check needs /tmp/plot_schema_sample.json (crafted sample)")
        return 1
    data = json.loads(sample.read_text())
    written = render(data, "/tmp/plot_selfcheck")
    assert any(p.endswith("_accuracy.png") for p in written), "no accuracy figure"
    assert any(p.endswith("_speed.png") for p in written), "no speed figure"
    for p in written:
        assert Path(p).stat().st_size > 0, f"empty figure {p}"
    print("self-check OK:", ", ".join(written))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", nargs="?", help="cuda_scaling result JSON")
    parser.add_argument("--out-prefix", default="results/anchor_sweep",
                        help="output path prefix; figures get _d{d}_{accuracy,speed}.png")
    parser.add_argument("--self-check", action="store_true",
                        help="render the crafted schema sample (no GPU) and assert")
    args = parser.parse_args(argv)

    if args.self_check:
        return _self_check()
    if not args.result_json:
        parser.error("result_json is required unless --self-check")
    data = json.loads(Path(args.result_json).read_text())
    written = render(data, args.out_prefix)
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
