#!/usr/bin/env python
"""Wall-clock scaling benchmark.

Measures forward-pass wall-clock as a function of sequence length for each
attention mechanism. Unlike the approximation-quality microbenchmark, this
isolates *speed*: it shows where the linear-in-``n`` methods overtake softmax's
quadratic cost.

The multi-anchor piecewise apply is ``O(batch · m · n · d²)`` — linear in the
sequence length ``n`` with a small constant ``m`` (the anchor count) — versus
softmax's ``O(batch · n² · d)``. This benchmark demonstrates the crossover where
the piecewise wall-clock drops below softmax, and reports the Performer and
linear-attention curves for a head-to-head comparison of the linear-time
methods.

Timing on accelerators is asynchronous, so every measured region is bracketed by
a device synchronize; results are averaged over repeated forwards after warmup.

Examples
--------
Smoke test::

    python benchmarks/harness/run_scaling.py --smoke

Full run::

    python benchmarks/harness/run_scaling.py --lengths 128 256 512 1024 \
        --out results/scaling.json
"""

import argparse
import platform
import sys
import time
from pathlib import Path

# Allow running directly from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from piecewise_linear_attention.core.registry import build_attention
from piecewise_linear_attention.harness import get_device, set_seed
from piecewise_linear_attention.harness.results import ExperimentResult, save_result


def _sync(device):
    """Block until queued device work has finished (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _build_methods(dim, num_anchors):
    """Return name -> attention module for the scaling comparison.

    ``standard`` is the quadratic baseline; ``linear``, ``performer`` and the
    piecewise variants are the linear-in-``n`` methods under comparison. All are
    non-causal.

    ``piecewise_mean`` is the single mean-centroid anchor (the fastest piecewise
    config and — because the centroid minimizes the mean squared query-to-anchor
    distance that drives first-order error — also the most accurate first-order
    one). ``piecewise_stride`` uses ``num_anchors`` positionally-placed anchors,
    which is both slower and less accurate; it is kept as a baseline.
    """
    common = dict(dim=dim, dropout=0.0, causal=False)
    return {
        "standard": build_attention("standard", **common),
        "linear": build_attention("linear", **common),
        "performer": build_attention("performer", **common),
        "piecewise_mean": build_attention("piecewise", **common),
        "piecewise_stride": build_attention("piecewise_stride", num_anchors=num_anchors, **common),
    }


def _time_forward(module, Q, K, V, device, warmup, repeats):
    """Mean seconds per forward pass, measured with device synchronization."""
    with torch.no_grad():
        for _ in range(warmup):
            module(Q, K, V)
        _sync(device)
        start = time.perf_counter()
        for _ in range(repeats):
            module(Q, K, V)
        _sync(device)
        elapsed = time.perf_counter() - start
    return elapsed / repeats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--batch", type=int, default=256, help="B*H flattened batch")
    parser.add_argument("--num-anchors", type=int, default=4)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.batch = 2
        args.dim = 16
        args.lengths = [16]
        args.warmup = 1
        args.repeats = 2
        args.num_anchors = 2

    device = get_device(args.device)
    print(f"device: {device}")
    methods = {k: v.to(device) for k, v in _build_methods(args.dim, args.num_anchors).items()}

    rows = []
    for seq_len in args.lengths:
        set_seed(seq_len)
        Q = torch.randn(args.batch, seq_len, args.dim, device=device)
        K = torch.randn(args.batch, seq_len, args.dim, device=device)
        V = torch.randn(args.batch, seq_len, args.dim, device=device)
        wall_ms = {}
        for name, module in methods.items():
            seconds = _time_forward(module, Q, K, V, device, args.warmup, args.repeats)
            wall_ms[name] = seconds * 1000.0
        rows.append({"seq_len": seq_len, "wall_ms": wall_ms})
        summary = " ".join(f"{n}={wall_ms[n]:.2f}" for n in methods)
        print(f"  n={seq_len:5d}  {summary}")

    # First length (if any) at which piecewise_stride is faster than softmax.
    crossover = next(
        (
            row["seq_len"]
            for row in rows
            if row["wall_ms"]["piecewise_stride"] < row["wall_ms"]["standard"]
        ),
        None,
    )
    if crossover is not None:
        print(f"piecewise_stride overtakes softmax at n={crossover}")
    else:
        print("piecewise_stride did not overtake softmax in the tested range")

    # Speed of the single mean-anchor piecewise relative to Performer, per length
    # (>1 means piecewise_mean is faster). This is the head-to-head that matters:
    # the fastest piecewise config vs the strongest linear-time softmax estimator.
    speedup_vs_performer = {
        row["seq_len"]: row["wall_ms"]["performer"] / row["wall_ms"]["piecewise_mean"]
        for row in rows
    }
    for seq_len, ratio in speedup_vs_performer.items():
        print(f"  n={seq_len:5d}  piecewise_mean is {ratio:.2f}x performer")

    result = ExperimentResult(
        config={
            "benchmark": "scaling",
            "dim": args.dim,
            "batch": args.batch,
            "num_anchors": args.num_anchors,
            "lengths": args.lengths,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        metrics={
            "crossover_n": crossover,
            "speedup_vs_performer": speedup_vs_performer,
        },
        environment={
            "device": str(device),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        history={"grid": rows},
    )

    if args.out:
        path = save_result(result, args.out)
        print(f"wrote {path}")

    if args.smoke:
        assert rows, "no measurements produced"
        for row in rows:
            for name, value in row["wall_ms"].items():
                assert value >= 0.0, name
        print("smoke OK")

    return result


if __name__ == "__main__":
    main(sys.argv[1:])
