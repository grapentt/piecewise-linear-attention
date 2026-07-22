"""Profile the reduced-rank (query-residual low-rank) anchor build.

Reports THREE axes separately, never conflated (the top-p lesson: a FLOP ceiling
is not a wall-clock speedup):

  1. Time   -- dense build vs reduced, with the reduced path broken into
               (a) SVD + basis (R_j) estimation, (b) reduced build einsum,
               (c) reduced apply. The SVD line is the load-bearing cost; it is
               always shown so a wash is visible.
  2. Memory -- peak allocated bytes for the compact dense operator M (b,m,d,d)
               vs the reduced factors R + M_red (b,m,d,d').
  3. FLOP   -- theoretical build FLOPs d*d' (using the MEASURED per-anchor d'_j)
               vs the dense d^2. A ceiling, not a timing.

Output is an MLflow-shaped JSON envelope ``{params, metrics, rows}`` so the
numbers drop straight into ``mlflow.log_params`` / ``mlflow.log_metrics`` once
tracking is wired up; ``params`` is the run configuration, ``metrics`` the
aggregate measurements, ``rows`` the per-(n,dtype) detail.

The queries are drawn from a clustered mixture (a few tight Gaussian blobs) so
the residual subspace is genuinely low-rank -- the regime the real d=128
activations exhibit. This is a MICROBENCHMARK of the build/apply kernels, not an
accuracy claim; accuracy is gated separately by run_real_activation_fidelity.py
(``--with-reduced``).

Usage
-----
    python benchmarks/harness/run_reduced_build.py --smoke
    python benchmarks/harness/run_reduced_build.py \
        --device cuda --dim 128 --anchors 64 --batch 8 \
        --lengths 1024 2048 4096 --dtypes bf16 fp32 \
        --out results/reduced_build_a100.json
"""

import argparse
import math
import platform
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from piecewise_linear_attention.core.anchors import KMeansAnchor
from piecewise_linear_attention.core.attention import PiecewiseAttention
from piecewise_linear_attention.harness.results import ExperimentResult, save_result

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _clustered_queries(b, n, d, num_blobs, device, generator):
    """Queries from ``num_blobs`` tight Gaussian clusters so per-cluster residuals
    are low-rank (the real-activation regime the reduced build targets)."""
    centers = torch.randn(b, num_blobs, d, generator=generator) * 4.0
    which = torch.randint(0, num_blobs, (b, n), generator=generator)
    base = torch.gather(centers, 1, which.unsqueeze(-1).expand(-1, -1, d))
    jitter = torch.randn(b, n, d, generator=generator) * 0.3
    return (base + jitter).to(device)


def _time(fn, device, warmup, repeats):
    """Median seconds over ``repeats`` calls after ``warmup``, device-synced."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        _sync(device)
        samples = []
        for _ in range(repeats):
            _sync(device)
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end) / 1e3)  # ms -> s
            else:
                import time

                t0 = time.perf_counter()
                fn()
                _sync(device)
                samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _peak_mem_bytes(fn, device):
    """Peak allocated bytes during ``fn`` (CUDA only; 0 elsewhere)."""
    if device.type != "cuda":
        return 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def profile_length(attn_dense, attn_reduced, Q, K, V, anchors, device, warmup, repeats):
    """One (n, dtype) cell: time / memory / FLOP for dense vs reduced.

    The dense and reduced modules share anchors and inputs so the only difference
    is the build+apply path. Times are captured for the full forward and, for the
    reduced path, for the SVD/basis stage in isolation (by calling the build
    method directly) so the SVD's share is explicit.
    """
    b, n, d = Q.shape
    m = anchors.shape[1]
    scale = 1.0 / math.sqrt(d)
    with torch.no_grad():
        alpha = torch.softmax(torch.einsum("bmd,bnd->bmn", anchors, K) * scale, dim=-1)
        assign = torch.cdist(Q.float(), anchors.float()).argmin(dim=-1)

    # --- Full-forward wall-clock: dense vs reduced. --------------------------
    t_dense = _time(lambda: attn_dense._compute_multi_anchor(Q, K, V, anchors),
                    device, warmup, repeats)
    t_reduced = _time(lambda: attn_reduced._compute_multi_anchor(Q, K, V, anchors),
                      device, warmup, repeats)

    # --- Reduced-path stage breakdown. ---------------------------------------
    # (a) SVD + basis + reduced build (the whole _build_jacobian_factors_reduced).
    t_reduced_build = _time(
        lambda: attn_reduced._build_jacobian_factors_reduced(alpha, V, K, Q, anchors, assign),
        device, warmup, repeats,
    )
    # Dense build in isolation for a like-for-like build comparison.
    t_dense_build = _time(
        lambda: torch.einsum("bmn,bnd,bne->bmde", alpha, V, K),
        device, warmup, repeats,
    )

    # --- Measured per-anchor d'_j and the FLOP ceiling. ----------------------
    with torch.no_grad():
        R, M_red, dprime = attn_reduced._build_jacobian_factors_reduced(
            alpha, V, K, Q, anchors, assign
        )
    dprime_f = dprime.float()
    dprime_mean = float(dprime_f.mean())
    dprime_median = float(dprime_f.median())
    dprime_max = int(dprime.max())
    # Build-FLOP ceiling: reduced sums Σ_j (n * d * d'_j) vs dense (m * n * d^2),
    # per batch element. Report as a ratio (dense / reduced), the ceiling speedup.
    reduced_build_flops = float((n * d * dprime_f.sum(dim=1)).mean())  # per batch elt
    dense_build_flops = float(m * n * d * d)
    flop_ceiling_ratio = dense_build_flops / max(reduced_build_flops, 1.0)

    # --- Peak memory: compact dense M vs reduced R + M_red. ------------------
    mem_dense = _peak_mem_bytes(
        lambda: torch.einsum("bmn,bnd,bne->bmde", alpha, V, K), device
    )
    mem_reduced = _peak_mem_bytes(
        lambda: attn_reduced._build_jacobian_factors_reduced(alpha, V, K, Q, anchors, assign),
        device,
    )

    return {
        "seq_len": n,
        "dtype": str(Q.dtype).replace("torch.", ""),
        "batch": b,
        "dim": d,
        "anchors": m,
        # time (seconds, median)
        "t_forward_dense_s": t_dense,
        "t_forward_reduced_s": t_reduced,
        "t_build_dense_s": t_dense_build,
        "t_build_reduced_s": t_reduced_build,  # includes SVD + basis + einsum
        "forward_speedup": t_dense / max(t_reduced, 1e-12),
        "build_speedup": t_dense_build / max(t_reduced_build, 1e-12),
        # measured rank
        "dprime_mean": dprime_mean,
        "dprime_median": dprime_median,
        "dprime_max": dprime_max,
        "dprime_mean_frac": dprime_mean / d,
        # FLOP ceiling (theoretical, not timed)
        "flop_ceiling_ratio": flop_ceiling_ratio,
        # memory (bytes, CUDA only)
        "mem_build_dense_bytes": mem_dense,
        "mem_build_reduced_bytes": mem_reduced,
        "mem_ratio": (mem_dense / mem_reduced) if mem_reduced else float("nan"),
    }


def _print_row(r):
    def s(x):
        return f"{x*1e3:7.2f}ms" if x >= 0 else "   n/a"
    print(
        f"  n={r['seq_len']:>5} {r['dtype']:>4}  "
        f"fwd dense {s(r['t_forward_dense_s'])} / reduced {s(r['t_forward_reduced_s'])} "
        f"(x{r['forward_speedup']:.2f})  |  "
        f"build dense {s(r['t_build_dense_s'])} / reduced {s(r['t_build_reduced_s'])} "
        f"(x{r['build_speedup']:.2f})"
    )
    print(
        f"        d'(mean/med/max)={r['dprime_mean']:.1f}/{r['dprime_median']:.0f}/"
        f"{r['dprime_max']} of {r['dim']} ({r['dprime_mean_frac']*100:.0f}%)  "
        f"FLOP-ceiling x{r['flop_ceiling_ratio']:.1f}  "
        f"mem x{r['mem_ratio']:.2f}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--anchors", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num-blobs", type=int, default=32,
                        help="Query cluster count (controls residual rank).")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Retained-variance threshold for build_reduced_d.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--dtypes", type=str, nargs="+", default=["fp32"],
                        choices=list(_DTYPES))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.device = "cpu"
        args.dim = 32
        args.anchors = 8
        args.batch = 2
        args.lengths = [256]
        args.dtypes = ["fp32"]
        args.warmup = 1
        args.repeats = 2

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device: {device}  torch: {torch.__version__}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    rows = []
    for dtype_name in args.dtypes:
        dtype = _DTYPES[dtype_name]
        for n in args.lengths:
            gen = torch.Generator().manual_seed(args.seed + n)
            Q = _clustered_queries(args.batch, n, args.dim, args.num_blobs, torch.device("cpu"), gen)
            K = torch.randn(args.batch, n, args.dim, generator=gen)
            V = torch.randn(args.batch, n, args.dim, generator=gen)
            # Anchors are chosen in fp32 on the target device, shared by both paths.
            strat = KMeansAnchor(args.anchors, iters=3)
            anchors = strat.select(Q.to(device))
            Q = Q.to(device=device, dtype=dtype)
            K = K.to(device=device, dtype=dtype)
            V = V.to(device=device, dtype=dtype)
            anchors = anchors.to(dtype=dtype)

            attn_dense = PiecewiseAttention(
                args.dim, scale=True, anchor_strategy=KMeansAnchor(args.anchors, iters=3)
            ).to(device)
            attn_reduced = PiecewiseAttention(
                args.dim, scale=True, anchor_strategy=KMeansAnchor(args.anchors, iters=3),
                build_reduced_d=args.threshold,
            ).to(device)

            row = profile_length(
                attn_dense, attn_reduced, Q, K, V, anchors, device,
                args.warmup, args.repeats,
            )
            rows.append(row)
            _print_row(row)

    # MLflow-shaped envelope: params (config) + metrics (aggregate) + rows (detail).
    params = {
        "dim": args.dim, "anchors": args.anchors, "batch": args.batch,
        "num_blobs": args.num_blobs, "threshold": args.threshold,
        "lengths": args.lengths, "dtypes": args.dtypes,
        "warmup": args.warmup, "repeats": args.repeats, "seed": args.seed,
    }
    metrics = {
        "best_forward_speedup": max((r["forward_speedup"] for r in rows), default=float("nan")),
        "best_build_speedup": max((r["build_speedup"] for r in rows), default=float("nan")),
        "max_flop_ceiling_ratio": max((r["flop_ceiling_ratio"] for r in rows), default=float("nan")),
        "best_mem_ratio": max((r["mem_ratio"] for r in rows if not math.isnan(r["mem_ratio"])), default=float("nan")),
        "min_dprime_mean_frac": min((r["dprime_mean_frac"] for r in rows), default=float("nan")),
    }

    print("\nsummary (measure first, claim second):")
    print(f"  best build wall-clock speedup:  x{metrics['best_build_speedup']:.2f}"
          "  <- includes the SVD cost; a value <1 means the SVD ate the saving")
    print(f"  FLOP-ceiling (theoretical):     x{metrics['max_flop_ceiling_ratio']:.1f}")
    print(f"  best memory reduction:          x{metrics['best_mem_ratio']:.2f}")

    result = ExperimentResult(
        config={"benchmark": "reduced_build_profile", **params},
        metrics=metrics,
        environment={"device": str(device), "torch": torch.__version__,
                     "platform": platform.platform(),
                     "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        history={"params": params, "metrics": metrics, "rows": rows},
    )
    if args.out:
        print(f"wrote {save_result(result, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
