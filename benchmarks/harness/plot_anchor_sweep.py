#!/usr/bin/env python
"""Render the anchor-count sweep as accuracy, speed, and Pareto figures.

Reads the ``cuda_scaling`` result JSON produced by ``run_cuda_scaling.py`` with an
anchor sweep (``--anchors``) and real-activation accuracy (``--acc-real
--acc-layers``), and writes figures under ``{prefix}/d{d}/`` in descriptive
subfolders:

- **Accuracy**, one figure *per layer* (``accuracy_per_layer/accuracy_layer{L}.png``):
  x = anchor count ``m``, y = output relative error to exact softmax on real
  trained-model activations, with a horizontal reference line per baseline at *that
  layer's own* rel-err (baselines vary by layer, so each layer gets its own lines
  rather than one cross-layer average). Lower is better; ``m`` is the piecewise dial.
- **Speed** (``speed/speed_vs_anchors.png``): x = anchor count ``m``, y = forward
  latency (ms), one curve per sequence length ``n`` (colour-coded), with a horizontal
  reference line per baseline per ``n``. OOM points are omitted from the curve and
  annotated as an OOM ceiling marker so the fit boundary is visible.
- **Pareto** (``pareto/pareto_m1_layer{L}.png``): the single-anchor ``m=1`` config
  and every baseline plotted as points in the speed × accuracy plane (x = latency,
  y = rel-err at that layer), so the both-axes trade-off is visible in one figure.
  Down-and-left is better; the non-dominated frontier is highlighted.

Figures fix a single per-head ``d`` (annotated); a separate ``d{d}`` tree is written
per ``d`` if the sweep covered more than one. Non-anchor baselines have no ``m`` axis,
so on the anchor/speed plots they appear only as horizontal references.

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
# Baselines whose defining projection is a *learned* parameter (trained end-to-end in
# their papers) but is left at random init in this untrained accuracy pass, so their
# rel-err reflects a random compression rather than the method. Flagged in the legend
# and excluded from the headline accuracy claim; the training-free baselines
# (performer, nystromformer) are the fair accuracy peers.
_UNTRAINED = {"linformer", "luna"}


def _baseline_label(method, value):
    """Legend label for a baseline: value plus an ``(untrained)`` flag for the
    learned-projection baselines scored at random init."""
    suffix = "  [untrained]" if method in _UNTRAINED else ""
    return f"{method}  ({value:.4f}){suffix}"


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
    """Per-layer accuracy for the accuracy plots.

    Returns ``(anchor_acc, baseline_acc)`` where ``anchor_acc[layer]`` is a sorted
    ``[(m, rel_err), ...]`` for the piecewise dial and ``baseline_acc[layer][method]``
    is that layer's own rel-err. Baselines are kept *per layer* (not averaged) so
    each layer's figure shows the baselines' true values at that layer — the spread
    across layers is real (e.g. Nyström 0.071 @ L6 vs 0.037 @ L18) and averaging it
    into one line would misstate the comparison."""
    anchor_acc = {}
    baseline_acc = {}
    for pl in accuracy_block.get("per_layer", []):
        layer = pl["layer"]
        for ar in pl.get("rows", []):
            if ar.get("rel_err") is None:
                continue
            m = _anchor_count(ar["method"])
            if m is not None:
                anchor_acc.setdefault(layer, []).append((m, ar["rel_err"]))
            elif ar["method"] in _BASELINES and ar["method"] not in _SOFTMAX:
                baseline_acc.setdefault(layer, {})[ar["method"]] = ar["rel_err"]
    for layer in anchor_acc:
        anchor_acc[layer].sort()
    return anchor_acc, baseline_acc


def _plot_accuracy_layer(layer, anchor_pts, baselines, dim, model_id, out_path):
    """One accuracy figure for a single layer: the piecewise dial (x=m) plus a
    horizontal line for each baseline at *its own* rel-err for this layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ms = [m for m, _ in anchor_pts]
    errs = [e for _, e in anchor_pts]
    ax.plot(ms, errs, "-o", color="#1f3b73", linewidth=2.2, markersize=6,
            label="piecewise (anchor-Taylor)")

    base_cmap = plt.get_cmap("tab10")
    for j, mth in enumerate(sorted(baselines)):
        ax.axhline(baselines[mth], color=base_cmap(j % 10), linestyle="--",
                   linewidth=1.4, alpha=0.85,
                   label=_baseline_label(mth, baselines[mth]))

    ax.set_xlabel("anchor count  m")
    ax.set_ylabel("relative error to exact softmax  (lower = better)")
    ax.set_title(f"Accuracy vs anchor count — layer {layer}\n"
                 f"real activations ({model_id}), d={dim}")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
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


def _plot_pareto_m1(m1_lat, m1_err, baseline_lat, baseline_err, layer, n,
                    dim, model_id, out_path):
    """Speed × accuracy scatter for the single-anchor ``m=1`` config vs baselines.

    Each method is one point: x = forward latency (ms) at length ``n``, y = rel-err
    to softmax at ``layer``. Down-and-left is better. The non-dominated frontier
    (no other point is <= on both axes) is drawn as a connecting line so the
    trade-off m=1 sits on is explicit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = []  # (label, latency, rel_err, is_piecewise)
    if m1_lat is not None and m1_err is not None:
        pts.append(("piecewise m=1", m1_lat, m1_err, True))
    for mth in sorted(baseline_err):
        if mth in baseline_lat and baseline_lat[mth] is not None:
            pts.append((mth, baseline_lat[mth], baseline_err[mth], False))
    if not pts:
        return None

    # Non-dominated frontier: a point is on it if no other point is <= on both axes.
    def dominated(p, others):
        return any(o[1] <= p[1] and o[2] <= p[2] and (o[1] < p[1] or o[2] < p[2])
                   for o in others if o is not p)
    frontier = sorted((p for p in pts if not dominated(p, pts)),
                      key=lambda p: p[1])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if len(frontier) > 1:
        ax.plot([p[1] for p in frontier], [p[2] for p in frontier],
                "-", color="0.6", linewidth=1.2, alpha=0.8, zorder=1,
                label="non-dominated frontier")
    for label, lat, err, is_pw in pts:
        ax.scatter([lat], [err], s=140 if is_pw else 80,
                   marker="*" if is_pw else "o",
                   color="#d62728" if is_pw else "#1f77b4",
                   edgecolor="black", linewidth=0.8, zorder=3)
        tag = "  [untrained]" if label in _UNTRAINED else ""
        ax.annotate(f"  {label}{tag}", xy=(lat, err), fontsize=8,
                    va="center", ha="left",
                    fontweight="bold" if is_pw else "normal")

    ax.set_xlabel(f"forward latency at n={n}  (ms, lower = better)")
    ax.set_ylabel(f"rel-err to softmax, layer {layer}  (lower = better)")
    ax.set_title(f"Speed × accuracy — piecewise m=1 vs baselines\n"
                 f"layer {layer}, d={dim}, real activations ({model_id})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
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
    out_root = Path(out_prefix)
    written = []
    for dim in dims:
        anchor_lat, baseline_lat, oom_m = _collect_speed(rows, dim, speed_dtype)
        acc_block = accuracy.get(str(dim), {})
        model_id = acc_block.get("model_id", config.get("acc_model", "real"))
        anchor_acc, baseline_acc = _collect_accuracy(acc_block)

        # Accuracy: one figure per layer, under a descriptive subfolder.
        if anchor_acc:
            acc_dir = out_root / f"d{dim}" / "accuracy_per_layer"
            acc_dir.mkdir(parents=True, exist_ok=True)
            for layer in sorted(anchor_acc):
                p = _plot_accuracy_layer(
                    layer, anchor_acc[layer], baseline_acc.get(layer, {}),
                    dim, model_id, str(acc_dir / f"accuracy_layer{layer}.png"))
                written.append(p)
        else:
            print(f"d={dim}: no accuracy data; skipping accuracy plots")

        # Speed: single figure (two n-curves), under its own subfolder.
        if anchor_lat:
            speed_dir = out_root / f"d{dim}" / "speed"
            speed_dir.mkdir(parents=True, exist_ok=True)
            p = _plot_speed(anchor_lat, baseline_lat, oom_m, dim,
                            str(speed_dir / "speed_vs_anchors.png"))
            written.append(p)
        else:
            print(f"d={dim}: no speed data; skipping speed plot")

        # Pareto: m=1 speed × accuracy vs baselines, one figure per layer. Uses the
        # smallest swept n (where softmax/baselines are most likely to have run).
        if anchor_acc and anchor_lat:
            pareto_n = min(anchor_lat)
            m1_lat = next((ms for m, ms in anchor_lat[pareto_n] if m == 1), None)
            base_lat_n = {mth: lat[pareto_n] for mth, lat in baseline_lat.items()
                          if pareto_n in lat}
            pareto_dir = out_root / f"d{dim}" / "pareto"
            pareto_dir.mkdir(parents=True, exist_ok=True)
            for layer in sorted(anchor_acc):
                m1_err = next((e for m, e in anchor_acc[layer] if m == 1), None)
                p = _plot_pareto_m1(
                    m1_lat, m1_err, base_lat_n, baseline_acc.get(layer, {}),
                    layer, pareto_n, dim, model_id,
                    str(pareto_dir / f"pareto_m1_layer{layer}.png"))
                if p:
                    written.append(p)
    return written


def _self_check():
    """Render the crafted schema sample so the parser/plot paths are exercised
    without a GPU run. Writes under a temp prefix and asserts figures appear."""
    sample = Path("/tmp/plot_schema_sample.json")
    if not sample.exists():
        print("self-check needs /tmp/plot_schema_sample.json (crafted sample)")
        return 1
    data = json.loads(sample.read_text())
    written = render(data, "/tmp/plot_selfcheck")
    assert any("accuracy_layer" in p for p in written), "no per-layer accuracy figure"
    assert any(p.endswith("speed_vs_anchors.png") for p in written), "no speed figure"
    for p in written:
        assert Path(p).stat().st_size > 0, f"empty figure {p}"
    print("self-check OK:", ", ".join(written))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", nargs="?", help="cuda_scaling result JSON")
    parser.add_argument("--out-prefix", default="results/anchor_sweep",
                        help="output root; figures written under "
                             "{prefix}/d{d}/accuracy_per_layer/ and {prefix}/d{d}/speed/")
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
