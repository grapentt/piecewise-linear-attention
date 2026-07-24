#!/usr/bin/env python
"""Op-level CPU profile of the multi-anchor piecewise apply.

Explains *why* ``piecewise_kmeans`` with ``m > 1`` anchors is slower than dense
softmax attention, by attributing the forward wall-clock to the individual
operators inside :meth:`PiecewiseAttention._compute_multi_anchor` and
:meth:`KMeansAnchor.select`. It complements ``results/anchor_cost_sweep.json``
(which reports only per-method totals) with a per-op breakdown at the same
shapes, so the totals here can be cross-checked against that committed file.

Two views are produced at each ``(dim, seq_len, num_anchors)`` cell:

1. **Isolated-op timing** — each suspect operator is timed on its own with the
   same shapes it sees inside the forward, warmed and repeated with
   ``time.perf_counter``. The suspects are:

     - ``kmeans_lloyd``      the k-means Lloyd loop in ``KMeansAnchor.select``
                             (``cdist`` + ``scatter_add_`` per iteration), which
                             is recomputed on *every* forward and is independent
                             of the einsum below.
     - ``kmeans_cdist_iter`` a single Lloyd ``cdist`` (the loop's inner cost),
                             so the loop cost can be split from the assignment.
     - ``second_moment``     the ``einsum("bmn,bnd,bne->bmde")`` that builds the
                             ``(b, m, d, d)`` Jacobian operator — the term whose
                             cost is ``O(b·m·n·d²)``, linear in ``m`` and
                             quadratic in ``d``.
     - ``assign_cdist``      the nearest-anchor ``cdist(Q, anchors).argmin`` that
                             routes each query to its anchor.
     - ``fp32_cast``         ``Q.float()`` alone, to size the half→float upcast
                             that both k-means and the assignment perform (on CPU
                             everything is already fp32, so this is expected to be
                             negligible and is measured to confirm it).
     - ``scores_softmax``    the anchor scores + softmax that precede the build.

2. **torch.profiler self-time** — the same forward wrapped in
   ``torch.profiler.profile(record_shapes=True)``; the top aten operators by
   total CPU self-time are recorded so the isolated-op attribution can be
   sanity-checked against the profiler's own accounting.

The verdict this feeds: at small ``m`` the *clustering* (recomputed k-means)
dominates, while at larger ``m`` the ``second_moment`` einsum overtakes it and
scales with ``m·d²``. The fp32 cast is not a factor on CPU. See the PR write-up
for the full attribution.

Examples
--------
Smoke test (seconds)::

    python benchmarks/harness/run_multianchor_profile.py --smoke

Full run at the anchor_cost_sweep shapes::

    python benchmarks/harness/run_multianchor_profile.py \
        --dims 64 128 --lengths 512 2048 --anchors 1 4 16 \
        --out results/multianchor_profile.json
"""

import argparse
import math
import platform
import sys
import time
import warnings
from pathlib import Path

# Allow running directly from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

# A single profiled forward is exactly one profiler cycle, so torch's
# cross-cycle-retention notice does not apply to this usage; silence it so the
# op breakdown output (and the test-suite smoke run) stays clean.
warnings.filterwarnings("ignore", message="Profiler clears events")

from piecewise_linear_attention.core.anchors import KMeansAnchor
from piecewise_linear_attention.core.registry import build_attention
from piecewise_linear_attention.harness import set_seed
from piecewise_linear_attention.harness.results import ExperimentResult, save_result


def _time(fn, warmup, repeats):
    """Mean milliseconds per call of ``fn`` under ``no_grad`` after warmup."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        elapsed = time.perf_counter() - start
    return elapsed / repeats * 1000.0


def _isolated_ops(Q, K, V, m, kmeans_iters, warmup, repeats):
    """Time each suspect operator in isolation at the given shapes.

    Returns a ``name -> milliseconds`` dict. The operators mirror exactly what
    ``KMeansAnchor.select`` and ``_compute_multi_anchor`` execute, so the sum of
    the dominant terms approximates the measured forward total.
    """
    b, n, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    # Stride-initialized anchors (the k-means Lloyd init) so the build ops see a
    # realistic (b, m, d) operand without paying for clustering here.
    idx = torch.linspace(0, n - 1, m, device=Q.device).round().long()
    anchors = Q[:, idx, :].clone()
    scores = torch.einsum("bmd,bnd->bmn", anchors, K) * scale
    alpha = torch.softmax(scores, dim=-1)

    def kmeans_lloyd():
        # Full Lloyd loop, matching KMeansAnchor.select (fp32, iters passes).
        Qf = Q.float()
        c = Qf[:, idx, :].clone()
        for _ in range(kmeans_iters):
            dists = torch.cdist(Qf, c)
            assign = dists.argmin(dim=-1)
            nc = torch.zeros_like(c)
            cnt = torch.zeros(b, m, device=Q.device)
            nc.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, d), Qf)
            cnt.scatter_add_(1, assign, torch.ones(b, n, device=Q.device))
            ne = cnt > 0
            cnt = cnt.clamp(min=1).unsqueeze(-1)
            c = torch.where(ne.unsqueeze(-1), nc / cnt, c)
        return c

    ops = {
        # Recomputed-every-forward clustering (m-nearly-independent, d-bound cdist).
        "kmeans_lloyd": kmeans_lloyd,
        "kmeans_cdist_iter": lambda: torch.cdist(Q.float(), anchors.float()),
        # The O(b*m*n*d^2) Jacobian-operator build (grows with m and d^2).
        "second_moment": lambda: torch.einsum("bmn,bnd,bne->bmde", alpha, V, K),
        # Nearest-anchor routing.
        "assign_cdist": lambda: torch.cdist(Q.float(), anchors.float()).argmin(dim=-1),
        # The half->float upcast in isolation (negligible on CPU; measured to show it).
        "fp32_cast": lambda: Q.float(),
        # Anchor scores + softmax preceding the build.
        "scores_softmax": lambda: torch.softmax(
            torch.einsum("bmd,bnd->bmn", anchors, K) * scale, dim=-1
        ),
    }
    return {name: _time(fn, warmup, repeats) for name, fn in ops.items()}


def _profiler_self_time(module, Q, K, V, top_k=12):
    """Return the top aten ops by total CPU self-time for one profiled forward.

    Uses ``record_shapes=True`` so the shapes are captured; only aggregate
    self-time is returned here (a compact list of ``{op, self_ms, shapes}``).
    """
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad():
        module(Q, K, V)  # one warmup so lazy allocations don't skew the profile
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            module(Q, K, V)
    events = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
    rows = []
    for e in events[:top_k]:
        rows.append(
            {
                "op": e.key,
                "self_ms": round(e.self_cpu_time_total / 1000.0, 4),
                "count": e.count,
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048])
    parser.add_argument(
        "--anchors",
        type=int,
        nargs="+",
        default=[1, 4, 16],
        help="anchor counts; 1 uses the single mean-anchor piecewise path",
    )
    parser.add_argument("--batch", type=int, default=64, help="B*H flattened batch")
    parser.add_argument("--kmeans-iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.dims = [16]
        args.lengths = [32]
        args.anchors = [1, 4]
        args.batch = 2
        args.warmup = 1
        args.repeats = 2

    device = torch.device("cpu")  # this investigation is CPU-only by design
    print(f"device: {device}  torch: {torch.__version__}  threads: {torch.get_num_threads()}")

    grid = []
    for dim in args.dims:
        for seq_len in args.lengths:
            set_seed(seq_len + dim)
            Q = torch.randn(args.batch, seq_len, dim)
            K = torch.randn(args.batch, seq_len, dim)
            V = torch.randn(args.batch, seq_len, dim)

            std = build_attention("standard", dim=dim, dropout=0.0, causal=False)
            std_ms = _time(lambda: std(Q, K, V), args.warmup, args.repeats)

            for m in args.anchors:
                if m == 1:
                    module = build_attention("piecewise", dim=dim, dropout=0.0, causal=False)
                else:
                    module = build_attention(
                        "piecewise_kmeans",
                        dim=dim,
                        dropout=0.0,
                        causal=False,
                        num_anchors=m,
                        kmeans_iters=args.kmeans_iters,
                    )
                total_ms = _time(lambda: module(Q, K, V), args.warmup, args.repeats)
                ops = (
                    _isolated_ops(Q, K, V, m, args.kmeans_iters, args.warmup, args.repeats)
                    if m > 1
                    else {}
                )
                prof_rows = _profiler_self_time(module, Q, K, V)
                cell = {
                    "dim": dim,
                    "seq_len": seq_len,
                    "num_anchors": m,
                    "wall_ms": {
                        "standard": round(std_ms, 4),
                        "piecewise": round(total_ms, 4),
                    },
                    "isolated_ops_ms": {k: round(v, 4) for k, v in ops.items()},
                    "profiler_self_ms": prof_rows,
                }
                grid.append(cell)
                dom = max(ops, key=ops.get) if ops else "n/a (single-anchor)"
                print(
                    f"  d={dim:3d} n={seq_len:5d} m={m:2d}  "
                    f"total={total_ms:8.3f}  std={std_ms:8.3f}  dominant_op={dom}"
                )

    # Attribution metric: fraction of the multi-anchor total explained by the
    # clustering vs the second-moment einsum, at each cell. This is the number
    # the verdict leans on -- it shows the crossover from clustering-bound (low
    # m) to einsum-bound (high m).
    attribution = {}
    for cell in grid:
        m = cell["num_anchors"]
        if m == 1:
            continue
        ops = cell["isolated_ops_ms"]
        total = cell["wall_ms"]["piecewise"]
        key = f"d{cell['dim']}_n{cell['seq_len']}_m{m}"
        attribution[key] = {
            "kmeans_frac": round(ops["kmeans_lloyd"] / total, 3),
            "einsum_frac": round(ops["second_moment"] / total, 3),
            "assign_frac": round(ops["assign_cdist"] / total, 3),
            "einsum_over_kmeans": round(ops["second_moment"] / ops["kmeans_lloyd"], 3),
        }

    result = ExperimentResult(
        config={
            "benchmark": "multianchor_profile",
            "purpose": (
                "op-level CPU attribution of the piecewise_kmeans (m>1) forward: "
                "splits wall-clock across the recomputed k-means Lloyd loop, the "
                "O(b*m*n*d^2) second-moment einsum, the nearest-anchor cdist, and "
                "the fp32 cast, at the shapes of results/anchor_cost_sweep.json"
            ),
            "batch_bh": args.batch,
            "dims": args.dims,
            "lengths": args.lengths,
            "anchors": args.anchors,
            "kmeans_iters": args.kmeans_iters,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        metrics={"attribution": attribution},
        environment={
            "device": str(device),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "num_threads": torch.get_num_threads(),
        },
        history={"grid": grid},
    )

    if args.out:
        path = save_result(result, args.out)
        print(f"wrote {path}")

    if args.smoke:
        assert grid, "no measurements produced"
        for cell in grid:
            assert cell["wall_ms"]["piecewise"] >= 0.0
        print("smoke OK")

    return result


if __name__ == "__main__":
    main(sys.argv[1:])
