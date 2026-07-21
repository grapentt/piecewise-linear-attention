#!/usr/bin/env python
"""Approximation-quality microbenchmark.

Measures how closely each linear-time mechanism approximates exact softmax
attention as a function of (a) query dispersion and (b) sequence length, with no
training involved. This isolates the approximation gap that motivates the method:
anchor-Taylor tracks softmax best when queries are clustered, and multiple
anchors extend that regime.

The metric is relative error ‖approx − softmax‖ / ‖softmax‖ on the attention
output, averaged over random problem instances.

Examples
--------
Smoke test::

    python benchmarks/harness/run_microbench.py --smoke

Full run::

    python benchmarks/harness/run_microbench.py --out results/microbench.json
"""

import argparse
import platform
import sys
from pathlib import Path

# Allow running directly from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from piecewise_linear_attention.core.anchors import KMeansAnchor
from piecewise_linear_attention.core.attention import (
    LinearAttention,
    PerformerAttention,
    PiecewiseAttention,
    StandardAttention,
)
from piecewise_linear_attention.harness import get_device, set_seed
from piecewise_linear_attention.harness.results import ExperimentResult, save_result


def _build_methods(dim, num_anchors):
    """Return name -> attention module (all non-causal)."""
    return {
        "linear_elu": LinearAttention(dim=dim, kernel_type="elu"),
        "performer": PerformerAttention(dim=dim, seed=0),
        "piecewise_mean": PiecewiseAttention(dim=dim, scale=True),
        "piecewise_kmeans": PiecewiseAttention(
            dim=dim, scale=True, anchor_strategy=KMeansAnchor(k=num_anchors, iters=3)
        ),
    }


def _rel_err(approx, reference):
    return (approx - reference).norm() / reference.norm()


def _make_problem(batch, seq_len, dim, dispersion, generator, device):
    """Queries clustered around a shared center with given dispersion.

    Small dispersion => queries near each other (favorable for anchor-Taylor);
    large dispersion => spread out (the boundary-bias regime where all
    first-order/kernel approximations degrade).
    """
    center = torch.randn(batch, 1, dim, generator=generator)
    Q = center + dispersion * torch.randn(batch, seq_len, dim, generator=generator)
    K = torch.randn(batch, seq_len, dim, generator=generator)
    V = torch.randn(batch, seq_len, dim, generator=generator)
    return Q.to(device), K.to(device), V.to(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num-anchors", type=int, default=4)
    parser.add_argument("--dispersions", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.batch = 4
        args.dispersions = [0.1, 1.0]
        args.lengths = [16]
        args.trials = 2
        args.num_anchors = 2

    device = get_device(args.device)
    print(f"device: {device}")
    softmax = StandardAttention(dim=args.dim).to(device)
    methods = {k: v.to(device) for k, v in _build_methods(args.dim, args.num_anchors).items()}

    rows = []
    with torch.no_grad():
        for seq_len in args.lengths:
            for dispersion in args.dispersions:
                accum = {name: 0.0 for name in methods}
                for trial in range(args.trials):
                    set_seed(1000 * trial + seq_len)
                    gen = torch.Generator().manual_seed(trial + 1)
                    Q, K, V = _make_problem(args.batch, seq_len, args.dim, dispersion, gen, device)
                    reference, _ = softmax(Q, K, V)
                    for name, module in methods.items():
                        approx, _ = module(Q, K, V)
                        accum[name] += _rel_err(approx, reference).item()
                row = {
                    "seq_len": seq_len,
                    "dispersion": dispersion,
                    "rel_err": {n: accum[n] / args.trials for n in methods},
                }
                rows.append(row)
                summary = " ".join(f"{n}={row['rel_err'][n]:.3f}" for n in methods)
                print(f"  n={seq_len:4d} disp={dispersion:.2f}  {summary}")

    result = ExperimentResult(
        config={
            "benchmark": "microbench",
            "dim": args.dim,
            "num_anchors": args.num_anchors,
            "dispersions": args.dispersions,
            "lengths": args.lengths,
            "trials": args.trials,
        },
        metrics={},
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
            for name, value in row["rel_err"].items():
                assert value >= 0.0, name
        print("smoke OK")

    return result


if __name__ == "__main__":
    main(sys.argv[1:])
