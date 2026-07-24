#!/usr/bin/env python
"""Measure anchor-reuse speedup and accuracy cost for multi-anchor piecewise.

Op-level profiling (``run_multianchor_profile.py``) shows the recomputed k-means
Lloyd loop is the single largest term in the ``piecewise_kmeans`` forward at low
anchor count. Anchor selection runs under ``no_grad`` and returns detached
centroids, so the anchors can be reused across forwards without touching the
gradient path. This harness quantifies the two sides of that trade-off:

- **speed** — mean forward wall-clock over a multi-step loop with (a) fresh
  clustering every step (``refresh_every=1``), (b) fewer Lloyd iterations, and
  (c) cached centroids refreshed every ``N`` steps (``CachedAnchor``).
- **accuracy** — relative error of each configuration's output to *exact softmax*
  attention, averaged over the loop. The queries drift slightly each step to
  emulate the changing minibatches of training, which is when a stale cache is
  most exposed.

The headline number is the accuracy *penalty* of reuse: fresh-clustering error
vs cached error, both measured against softmax. Because the anchors only choose
where the first-order expansion is centered, this penalty is expected to be
small even when the cached centroids drift from the fresh ones.

Examples
--------
Smoke test::

    python benchmarks/harness/run_anchor_reuse.py --smoke

Full run::

    python benchmarks/harness/run_anchor_reuse.py --out results/anchor_reuse.json
"""

import argparse
import platform
import sys
import time
from pathlib import Path

# Allow running directly from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from piecewise_linear_attention.core.anchors import CachedAnchor, KMeansAnchor
from piecewise_linear_attention.core.attention import PiecewiseAttention, StandardAttention
from piecewise_linear_attention.harness import set_seed
from piecewise_linear_attention.harness.results import ExperimentResult, save_result


def _drifting_inputs(batch, seq_len, dim, steps, drift, generator):
    """A base (Q, K, V) plus a per-step Q that drifts to emulate training.

    K and V are held fixed across steps so the measured accuracy change is
    attributable to the query drift the cached anchors fail to track, not to a
    changing attention target.
    """
    base_Q = torch.randn(batch, seq_len, dim, generator=generator)
    K = torch.randn(batch, seq_len, dim, generator=generator)
    V = torch.randn(batch, seq_len, dim, generator=generator)
    step_Q = [
        base_Q + drift * s * torch.randn(batch, seq_len, dim, generator=generator)
        for s in range(steps)
    ]
    return step_Q, K, V


def _run_loop(module, step_Q, K, V, reference_std):
    """Mean per-step forward ms and mean rel-err to exact softmax over the loop."""
    times, errs = [], []
    with torch.no_grad():
        # Warmup a couple of steps (also primes any anchor cache).
        for Q in step_Q[: min(2, len(step_Q))]:
            module(Q, K, V)
        for Q in step_Q:
            ref = reference_std(Q, K, V)[0]
            start = time.perf_counter()
            out = module(Q, K, V)[0]
            times.append((time.perf_counter() - start) * 1000.0)
            errs.append(((out - ref).norm() / ref.norm()).item())
    return sum(times) / len(times), sum(errs) / len(errs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=64, help="B*H flattened batch")
    parser.add_argument("--num-anchors", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--drift", type=float, default=0.05, help="per-step query drift scale")
    parser.add_argument(
        "--refresh", type=int, nargs="+", default=[4, 8, 16], help="cache refresh cadences to test"
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.dim = 16
        args.seq_len = 32
        args.batch = 2
        args.num_anchors = 4
        args.steps = 4
        args.refresh = [2]

    device = torch.device("cpu")
    set_seed(0)
    gen = torch.Generator().manual_seed(0)
    step_Q, K, V = _drifting_inputs(args.batch, args.seq_len, args.dim, args.steps, args.drift, gen)
    std = StandardAttention(dim=args.dim, scale=True)

    def kmeans_module(iters, refresh_every):
        base = KMeansAnchor(k=args.num_anchors, iters=iters)
        strat = CachedAnchor(base, refresh_every=refresh_every) if refresh_every > 1 else base
        return PiecewiseAttention(dim=args.dim, scale=True, anchor_strategy=strat)

    configs = {
        "fresh_iters3": kmeans_module(3, 1),  # baseline: recompute every step
        "fresh_iters1": kmeans_module(1, 1),  # variant (i): fewer Lloyd iters
        "fresh_iters0": kmeans_module(0, 1),  # stride-init only (no Lloyd)
    }
    for r in args.refresh:
        configs[f"cached_iters3_refresh{r}"] = kmeans_module(3, r)  # variant (ii)/(iii)

    rows = {}
    baseline_ms = baseline_err = None
    for name, module in configs.items():
        ms, err = _run_loop(module, step_Q, K, V, std)
        rows[name] = {"mean_ms": round(ms, 4), "mean_relerr_vs_softmax": round(err, 6)}
        if name == "fresh_iters3":
            baseline_ms, baseline_err = ms, err
        print(f"  {name:26s}  {ms:8.3f} ms   relerr={err:.5f}")

    # Speedup and accuracy penalty of each config vs the fresh-every-step baseline.
    summary = {}
    for name, r in rows.items():
        if name == "fresh_iters3":
            continue
        summary[name] = {
            "speedup_vs_fresh": round(baseline_ms / r["mean_ms"], 3),
            "relerr_penalty_abs": round(r["mean_relerr_vs_softmax"] - baseline_err, 6),
            "relerr_penalty_pct": round(
                (r["mean_relerr_vs_softmax"] / baseline_err - 1.0) * 100.0, 3
            ),
        }
    for name, s in summary.items():
        print(
            f"  {name:26s}  {s['speedup_vs_fresh']:.2f}x   "
            f"accuracy penalty {s['relerr_penalty_pct']:+.2f}%"
        )

    result = ExperimentResult(
        config={
            "benchmark": "anchor_reuse",
            "purpose": (
                "speedup and accuracy cost of anchor caching (CachedAnchor) and "
                "fewer Lloyd iters for piecewise_kmeans, over a drifting-input loop; "
                "accuracy is rel-err to exact softmax, penalty is vs fresh-every-step"
            ),
            "dim": args.dim,
            "seq_len": args.seq_len,
            "batch_bh": args.batch,
            "num_anchors": args.num_anchors,
            "steps": args.steps,
            "drift": args.drift,
            "refresh": args.refresh,
        },
        metrics={"summary_vs_fresh": summary},
        environment={
            "device": str(device),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "num_threads": torch.get_num_threads(),
        },
        history={"configs": rows},
    )

    if args.out:
        path = save_result(result, args.out)
        print(f"wrote {path}")

    if args.smoke:
        assert rows and summary, "no measurements produced"
        print("smoke OK")

    return result


if __name__ == "__main__":
    main(sys.argv[1:])
