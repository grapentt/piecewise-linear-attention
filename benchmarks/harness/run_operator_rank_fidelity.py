#!/usr/bin/env python
"""Operator-rank truncation fidelity probe: does truncating ``M_j`` directly
(rather than its query-residual input) preserve *delivered* accuracy?

The query-residual reduced build was measured (``run_reduced_build.py`` +
``run_jacobian_spectrum.py``) to be accurate but not a wall-clock win: even with a
free basis, the reduced apply keeps the full ``m·n·d·d'`` factor, so its FLOP
ceiling is only ~1.3-1.6x. The one structurally different lever is that the
*operator* ``M_j`` has a smaller effective rank than the query residuals it acts
on -- ``M_j`` could be truncated to rank ``r`` *directly*, cutting build to
``n·d·r`` on both axes and giving a ~2.4-3.2x ceiling. That lever is only real if
truncating ``M_j`` to its top-``r`` singular subspace preserves the *delivered*
output, i.e. the quantity that actually reaches the attention output:

    err_j = mean_i  ‖(M_j − M_r) · Δq_i‖ / ‖M_j · Δq_i‖

over the real query residuals ``Δq_i = q_i − a_j`` in cluster ``j``. This is
NOT the same as ``M_j``'s own singular-value energy: a low-energy singular
direction of ``M_j`` can align with a dominant ``Δq`` direction and inflate the
delivered error. No prior probe measures this coupling -- the spectrum probe
measured operator-side singular *values* (energy), not the error *delivered*
through the real residuals. This probe measures exactly the delivered error, so
it decides whether the direct-``M`` truncation idea is worth implementing before
any GPU kernel is written.

The build, anchors, and hard assignment replicate ``PiecewiseAttention`` exactly
(k-means query centroids, ``argmin_j ‖q_i − a_j‖``) so the clusters match the
production forward. Diagnostic-only: no reduced build path is exercised.

GATE: the direct-``M`` lever is worth building iff the WORST occupied cluster's
``err_j`` at ``r <= 52`` beats the Nyström L=256 real-activation floor
(``rel-err ≈ 0.0285`` at the worst layer). If ``err_j`` exceeds ~0.05 the
operator-energy truncation is confirmed to diverge from delivered accuracy and the
direct-``M`` idea is dead; the reduced build stays a frozen-anchor / memory result.

Examples
--------
Smoke (synthetic tensors, no model, no deps)::

    python benchmarks/harness/run_operator_rank_fidelity.py --smoke

Full run (real gpt-neo d=128, worst layer L6 and controls)::

    python benchmarks/harness/run_operator_rank_fidelity.py \
        --device cuda --model EleutherAI/gpt-neo-1.3B --layers 6 12 18 \
        --anchors 64 --ranks 36 40 50 52 --out results/operator_rank_fidelity_d128.json
"""

import argparse
import math
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from piecewise_linear_attention.core.anchors import KMeansAnchor
from piecewise_linear_attention.harness.results import ExperimentResult, save_result

# Occupied clusters below this many members are skipped for the aggregate
# (mirrors the reduced build's _MIN_CLUSTER self-guard: too few residuals to
# make a rank claim meaningful).
_MIN_CLUSTER = 8


def _relerr(approx, exact, eps=1e-9):
    """Per-vector relative L2 error, averaged; ``exact`` is (cnt, dim)."""
    num = torch.linalg.norm(approx - exact, dim=-1)
    den = torch.linalg.norm(exact, dim=-1).clamp_min(eps)
    return float((num / den).mean())


def _measure_layer(Q, K, V, m, ranks, scale, device):
    """One (head-folded) block: build exact M_j per occupied cluster, truncate to
    each rank r, measure delivered (M_j − M_r)·Δq error over the real residuals.

    Q, K, V are (heads, n, dim). Heads are treated as the batch axis, matching the
    harness's per-head convention. Returns per-rank aggregates across all occupied
    clusters of all heads (worst and median), plus the operator-energy rank for
    reference so the energy-vs-delivered divergence is visible.
    """
    heads, n, dim = Q.shape
    # Per-head anchors and hard assignment, exactly as PiecewiseAttention does it.
    strat = KMeansAnchor(m, iters=3)
    anchors = strat.select(Q.float())  # (heads, m, dim)
    with torch.no_grad():
        assign = torch.cdist(Q.float(), anchors.float()).argmin(dim=-1)  # (heads, n)

    # Per-rank accumulators over occupied clusters.
    per_rank_err = {r: [] for r in ranks}
    energy_rank_99 = []  # operator-energy rank@99% per cluster (for divergence view)
    cluster_sizes = []

    Qf, Kf, Vf, Af = Q.float(), K.float(), V.float(), anchors.float()
    for h in range(heads):
        for j in range(m):
            mask = assign[h] == j
            cnt = int(mask.sum().item())
            if cnt < _MIN_CLUSTER:
                continue
            cluster_sizes.append(cnt)
            dq = Qf[h][mask] - Af[h, j]  # (cnt, dim) -- the REAL residuals
            # Exact per-anchor second moment M_j = Σ_n α_jn v_n k_nᵀ, built from
            # ALL keys under this anchor's softmax weights (the production build).
            a_scores = (Af[h, j] @ Kf[h].T) * scale  # (n,)
            alpha = torch.softmax(a_scores, dim=-1)  # (n,)
            M = torch.einsum("n,nd,ne->de", alpha, Vf[h], Kf[h])  # (dim, dim)

            exact_apply = dq @ M.T  # (cnt, dim) == M_j · Δq for each residual

            # Single SVD of M_j; every rank r is a prefix truncation of it.
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)  # (d,d)(d,)(d,d)
            energy = S.double() ** 2
            cum = (torch.cumsum(energy, 0) / energy.sum().clamp_min(1e-12))
            energy_rank_99.append(int((cum < 0.99).sum().item()) + 1)

            for r in ranks:
                rr = min(r, S.numel())
                M_r = (U[:, :rr] * S[:rr]) @ Vh[:rr]  # rank-rr truncation of M_j
                approx_apply = dq @ M_r.T
                per_rank_err[r].append(_relerr(approx_apply, exact_apply))

    def agg(vals):
        if not vals:
            return {"worst": float("nan"), "median": float("nan"), "count": 0}
        t = torch.tensor(vals)
        return {"worst": float(t.max()), "median": float(t.median()),
                "count": len(vals)}

    rows = []
    for r in ranks:
        a = agg(per_rank_err[r])
        rows.append({"rank": r, "rank_frac": r / dim,
                     "delivered_relerr_worst": a["worst"],
                     "delivered_relerr_median": a["median"],
                     "n_clusters": a["count"]})
    er = agg([float(x) for x in energy_rank_99])
    return {
        "dim": dim, "heads": heads, "seq_len": n, "anchors": m,
        "n_occupied_clusters": len(cluster_sizes),
        "cluster_size_median": float(torch.tensor(cluster_sizes, dtype=torch.float).median())
            if cluster_sizes else float("nan"),
        "operator_energy_rank99_worst": er["worst"],
        "operator_energy_rank99_median": er["median"],
        "per_rank": rows,
    }


def _synthetic_block(heads, n, dim, seed=0):
    """Clustered synthetic Q (low-rank residuals) + random K/V, for --smoke."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(heads, 4, dim, generator=g) * 4.0
    which = torch.randint(0, 4, (heads, n), generator=g)
    base = torch.gather(centers, 1, which.unsqueeze(-1).expand(-1, -1, dim))
    Q = base + torch.randn(heads, n, dim, generator=g) * 0.3
    K = torch.randn(heads, n, dim, generator=g)
    V = torch.randn(heads, n, dim, generator=g)
    return Q, K, V


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18])
    parser.add_argument("--anchors", type=int, default=64)
    parser.add_argument("--ranks", type=int, nargs="+", default=[36, 40, 50, 52])
    parser.add_argument("--nystrom-floor", type=float, default=0.0285,
                        help="Real-activation Nyström L256 worst-layer rel-err (the gate).")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    print(f"device: {device}  torch: {torch.__version__}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    blocks = []  # list of (label, Q, K, V)
    if args.smoke:
        Q, K, V = _synthetic_block(heads=4, n=256, dim=32)
        blocks.append(("smoke_synthetic", Q.to(device), K.to(device), V.to(device)))
        args.anchors = 8
        args.ranks = [4, 8, 16]
    else:
        from benchmarks.harness.run_jacobian_spectrum import capture_decoder_d128
        for layer in args.layers:
            blk = capture_decoder_d128(args.model, corpus=None, layer=layer, device=device)
            if blk is None:
                print(f"SKIP_NO_SOURCE: {args.model} L{layer}")
                continue
            blocks.append((f"{args.model}_L{layer}",
                           blk["Q"], blk["K"], blk["V"]))

    if not blocks:
        print("no blocks captured; nothing to measure")
        return 1

    rows = []
    for label, Q, K, V in blocks:
        dim = Q.shape[-1]
        scale = 1.0 / math.sqrt(dim)
        res = _measure_layer(Q, K, V, args.anchors, args.ranks, scale, device)
        res["label"] = label
        rows.append(res)
        print(f"\n{label}  d={res['dim']} n={res['seq_len']} m={res['anchors']}  "
              f"({res['n_occupied_clusters']} occupied clusters, "
              f"op-energy rank@99 worst/med={res['operator_energy_rank99_worst']:.0f}/"
              f"{res['operator_energy_rank99_median']:.0f})")
        for pr in res["per_rank"]:
            verdict = ("BEATS Nyström" if pr["delivered_relerr_worst"] < args.nystrom_floor
                       else "loses")
            print(f"  r={pr['rank']:>3} ({pr['rank_frac']*100:.0f}%)  "
                  f"delivered rel-err worst/med="
                  f"{pr['delivered_relerr_worst']:.4f}/{pr['delivered_relerr_median']:.4f}"
                  f"  [{verdict}]")

    # Program-level verdict: the smallest rank that beats the Nyström floor at the
    # worst layer's worst cluster. If none, the direct-M lever is dead.
    worst_by_rank = {}
    for r in args.ranks:
        worst = max((pr["delivered_relerr_worst"]
                     for res in rows for pr in res["per_rank"] if pr["rank"] == r),
                    default=float("nan"))
        worst_by_rank[r] = worst
    passing = [r for r, w in worst_by_rank.items()
               if not math.isnan(w) and w < args.nystrom_floor]
    min_passing_rank = min(passing) if passing else None

    print("\nverdict (direct-M operator-rank truncation):")
    for r in args.ranks:
        print(f"  worst-cluster delivered rel-err @ r={r}: {worst_by_rank[r]:.4f}"
              f"  ({'PASS' if (r in passing) else 'fail'} vs {args.nystrom_floor})")
    if min_passing_rank is not None:
        print(f"  => GREENLIGHT direct-M at r={min_passing_rank} "
              f"(frac {min_passing_rank / rows[0]['dim']:.2f}); real >2x ceiling is reachable")
    else:
        print(f"  => KILL direct-M: operator-energy truncation diverges from delivered "
              f"accuracy at every tested r; reduced build stays a frozen-anchor/memory result")

    metrics = {
        "min_passing_rank": (min_passing_rank if min_passing_rank is not None else -1),
        "nystrom_floor": args.nystrom_floor,
        **{f"worst_relerr_r{r}": worst_by_rank[r] for r in args.ranks},
    }
    params = {"model": args.model, "layers": args.layers, "anchors": args.anchors,
              "ranks": args.ranks, "nystrom_floor": args.nystrom_floor}
    result = ExperimentResult(
        config={"benchmark": "operator_rank_fidelity", **params},
        metrics=metrics,
        environment={"device": str(device), "torch": torch.__version__,
                     "platform": platform.platform(),
                     "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        history={"params": params, "metrics": metrics, "rows": rows},
    )
    if args.out:
        print(f"\nwrote {save_result(result, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
