#!/usr/bin/env python
"""Jacobian d²-spectrum probe (issue #9): is the anchor-Taylor Jacobian factor
buildable/appliable in sub-``d²`` cost without losing the m=64 fidelity win?

The dominant build cost of multi-anchor anchor-Taylor attention is the per-anchor
second-moment operator ``M_j = Σ_n α_jn v_n k_nᵀ`` — a full ``d×d`` matrix, so the
cost carries a ``d²`` factor that dominates at realistic per-head ``d`` and is
untouched by every ``n``-axis sparsity trick. This probe measures whether ``d²``
can be cut, on **real trained-model activations** (the only distribution whose
verdict counts), by testing the two structurally-distinct truncation axes:

QUERY-RESIDUAL side (the load-bearing candidate)
    ``M_j`` is only ever consumed as ``M_j @ Δq`` with ``Δq = q_i − a_j``. k-means
    makes tight clusters, so ``Δq`` should live in a low-dimensional within-cluster
    subspace *by construction*. If the stacked per-cluster residuals ``{Δq}`` have
    effective rank ``d' ≪ d``, building ``M_j`` only on the ``d'`` key-dims ``Δq``
    excites (``M_j^red = Σ_n α_jn v_n (Rᵀk_n)ᵀ``, then ``M_j^red (Rᵀ Δq)``) is
    near-lossless and cuts ``d² → d·d'``. This axis is the one the settled top-p
    nucleus result does NOT rule out (that ruled out the *key* side).

KEY/OPERATOR side (the control the nucleus result argues against)
    The spectrum of ``M_j`` itself. Every competing structured-Jacobian idea
    (Nyström-landmark, shared basis, sketch, DPLR, Kronecker, Monarch) truncates
    this side, which the nucleus probe found broad on real activations. Measured
    here stratified by anchor breadth so the two verdicts are decided in one pass.

The probe reuses the fidelity harness's real-activation capture (native n=512
blocks — the trustworthy peakedness signal) and replicates the attention module's
own hard assignment ``argmin_j ‖q_i − a_j‖`` verbatim, so clusters match the
production forward pass exactly. It is diagnostic-only: no reduced build is run
here; it produces the cumulative-variance-vs-rank curves that gate whether writing
the reduced build is worth it.

GATE (from issue #9): greenlight the query-residual reduced build iff, at m=64,
the WORST-anchor ``d'@95%`` variance is a clear fraction of ``d`` (and shrinks as
``m`` grows / as ``d`` grows). Report worst-anchor, not just median — the top-p
lesson is that the max, not the mean, sets the shared cost.

Sources (native d, native n=512): ``sentence-transformers/all-MiniLM-L6-v2``
(d=32) and ``bert-base-uncased`` (d=64). Two real per-head dimensions let the
``d'/d`` trend-vs-d be read, the closest available proxy for the paper's d=128.

Examples
--------
Smoke (synthetic tensors, no model, no deps)::

    python benchmarks/harness/run_jacobian_spectrum.py --smoke

Full run::

    python benchmarks/harness/run_jacobian_spectrum.py \
        --out results/jacobian_spectrum.json --device cpu
"""

import argparse
import json
import math
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

# Reuse the fidelity harness's real-activation capture verbatim so the probe
# reads the identical native n=512 trained-model Q/K/V blocks.
from benchmarks.harness.run_real_activation_fidelity import (
    EPS,
    load_real_activations,
)

# Variance thresholds whose crossing rank d' is reported.
VAR_THRESHOLDS = (0.90, 0.95, 0.99)
# Anchor counts to sweep: the gate needs the d'-vs-m trend (does tighter
# clustering shrink the residual rank?), so span the fidelity regimes.
ANCHOR_COUNTS = (16, 64)
# Native block only — the tiled proxy flattens the attention landscape and is
# not a trustworthy peakedness/rank signal (same caveat as the nucleus probe).
NATIVE_LEN = 512


def assign_to_anchors(Q, anchors):
    """Replicate ``_compute_multi_anchor``'s hard assignment exactly:
    ``argmin_j ‖q_i − a_j‖`` in float32 (cdist has no fp16 kernel on some
    backends). Returns ``(batch, seq_len)`` int64 anchor indices."""
    return torch.cdist(Q.float(), anchors.float()).argmin(dim=-1)


def kmeans_anchors(Q, k, iters=3):
    """Deterministic k-means centroids matching ``KMeansAnchor.select``
    (linspace init, ``iters`` Lloyd steps, empty clusters keep their centroid).
    Reproduced here rather than imported so the probe is a self-contained
    reference for what the module does."""
    batch, seq_len, dim = Q.shape
    k = min(k, seq_len)
    Qf = Q.float()
    idx = torch.linspace(0, seq_len - 1, k, device=Q.device).round().long()
    centroids = Qf[:, idx, :].clone()
    for _ in range(iters):
        dists = torch.cdist(Qf, centroids)
        assign = dists.argmin(dim=-1)
        new_c = torch.zeros_like(centroids)
        counts = torch.zeros(batch, k, device=Q.device)
        new_c.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, dim), Qf)
        counts.scatter_add_(1, assign, torch.ones(batch, seq_len, device=Q.device))
        nonempty = counts > 0
        counts = counts.clamp(min=1).unsqueeze(-1)
        averaged = new_c / counts
        centroids = torch.where(nonempty.unsqueeze(-1), averaged, centroids)
    return centroids.detach()


def _rank_for_variance(singular_values, thresholds=VAR_THRESHOLDS):
    """Smallest rank whose cumulative *variance* (singular value squared) share
    reaches each threshold. Returns dict threshold -> rank (int).

    Variance, not singular-value, share: the residual energy that a rank-``r``
    projection ``R Rᵀ`` preserves is ``Σ_{i<r} σ_i²`` (Eckart–Young), which is the
    quantity that bounds the truncation error propagated through ``M_j``."""
    energy = singular_values.double() ** 2
    total = energy.sum().clamp_min(EPS)
    cum = torch.cumsum(energy, dim=0) / total
    out = {}
    for th in thresholds:
        # Smallest r with cum[r-1] >= th; searchsorted on the cumulative curve.
        r = int(torch.searchsorted(cum, torch.tensor(th, dtype=cum.dtype)).item()) + 1
        out[th] = min(r, singular_values.numel())
    return out


def query_residual_spectrum(Q, anchors, min_cluster=8):
    """Per-cluster SVD of the stacked query residuals ``Δq = q_i − a_j``.

    For every (sample, anchor) cluster with at least ``min_cluster`` members,
    center-free SVD of the residual matrix (residuals are already anchor-relative,
    so no re-centering) gives the effective dimensionality of the subspace ``Δq``
    occupies. This is the axis the reduced build would truncate; the reported ``d'``
    is the rank a query-PCA basis ``R`` would need.

    Returns per-cluster ``d'`` at each variance threshold, as ``(num_clusters,)``
    int tensors keyed by threshold, plus the raw cluster sizes. Aggregated
    worst/median/mean across clusters is left to the caller (worst matters most).
    """
    batch, seq_len, dim = Q.shape
    assign = assign_to_anchors(Q, anchors)  # (batch, seq_len)
    m = anchors.shape[1]
    per_thresh = {th: [] for th in VAR_THRESHOLDS}
    sizes = []
    frac_energy_topk = []  # fraction of residual variance in the top-min(m,dim)/... dirs
    for b in range(batch):
        for j in range(m):
            mask = assign[b] == j
            cnt = int(mask.sum().item())
            if cnt < min_cluster:
                continue
            dq = (Q[b][mask].float() - anchors[b, j].float())  # (cnt, dim)
            # Economy SVD; singular values only (subspace *size*, not the basis).
            try:
                svals = torch.linalg.svdvals(dq)  # (min(cnt, dim),)
            except Exception:
                continue
            ranks = _rank_for_variance(svals)
            for th in VAR_THRESHOLDS:
                per_thresh[th].append(ranks[th])
            sizes.append(cnt)
    return per_thresh, sizes


def operator_spectrum(Q, K, V, anchors, min_cluster=8):
    """Effective rank of the per-anchor operator ``M_j = Σ_n α_jn v_n k_nᵀ`` itself
    (the key/operator-side control). Built exactly as the module does — per-anchor
    softmax over ALL keys, ``scale = 1/√d`` — then SVD of the ``d×d`` operator.

    Also records each anchor's nucleus fraction (share of keys reaching p=0.99
    cumulative α mass) so the operator rank can be stratified by anchor breadth:
    the hypothesis under test is that broad anchors (large nucleus) have high
    operator rank, which is what would sink key-side low-rank on real data."""
    batch, seq_len, dim = Q.shape
    m = anchors.shape[1]
    scale = 1.0 / math.sqrt(dim)
    Kf, Vf, Af = K.float(), V.float(), anchors.float()
    scores = torch.einsum("bmd,bnd->bmn", Af, Kf) * scale  # (b, m, n)
    alpha = torch.softmax(scores, dim=-1)  # (b, m, n)
    # M_j = Σ_n α_jn v_n k_nᵀ, built the fused way (no (b,m,n,d) intermediate).
    M = torch.einsum("bmn,bnd,bne->bmde", alpha, Vf, Kf)  # (b, m, d, d)

    # Per-anchor nucleus fraction at p=0.99 (breadth stratifier).
    sorted_alpha, _ = torch.sort(alpha, dim=-1, descending=True)
    cumulative = sorted_alpha.cumsum(dim=-1)
    keep = torch.ones_like(sorted_alpha, dtype=torch.bool)
    keep[..., 1:] = cumulative[..., :-1] < 0.99
    nucleus_frac = keep.sum(dim=-1).float() / seq_len  # (b, m)

    per_thresh = {th: [] for th in VAR_THRESHOLDS}
    breadth = []
    for b in range(batch):
        for j in range(m):
            try:
                svals = torch.linalg.svdvals(M[b, j])  # (d,)
            except Exception:
                continue
            ranks = _rank_for_variance(svals)
            for th in VAR_THRESHOLDS:
                per_thresh[th].append(ranks[th])
            breadth.append(float(nucleus_frac[b, j]))
    return per_thresh, breadth


def _agg(values):
    """worst (max), p95, median, mean, and count for a list of per-cluster ranks."""
    if not values:
        return {"worst": float("nan"), "p95": float("nan"),
                "median": float("nan"), "mean": float("nan"), "count": 0}
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "worst": float(t.max()),
        "p95": float(torch.quantile(t, 0.95)),
        "median": float(t.median()),
        "mean": float(t.mean()),
        "count": int(t.numel()),
    }


def probe_block(Q, K, V, dim, anchor_counts=ANCHOR_COUNTS):
    """Run both spectra for each anchor count on one captured block."""
    rows = []
    for m in anchor_counts:
        anchors = kmeans_anchors(Q, m)
        qr_thresh, qr_sizes = query_residual_spectrum(Q, anchors)
        op_thresh, op_breadth = operator_spectrum(Q, K, V, anchors)
        row = {"m": m, "dim": dim}
        for th in VAR_THRESHOLDS:
            tag = f"{int(th * 100)}"
            qr = _agg(qr_thresh[th])
            op = _agg(op_thresh[th])
            # Query-residual d' and its FRACTION of d (the gate quantity).
            row[f"qres_dprime{tag}_worst"] = qr["worst"]
            row[f"qres_dprime{tag}_p95"] = qr["p95"]
            row[f"qres_dprime{tag}_median"] = qr["median"]
            row[f"qres_dprime{tag}_worst_frac"] = qr["worst"] / dim if qr["count"] else float("nan")
            row[f"qres_dprime{tag}_median_frac"] = qr["median"] / dim if qr["count"] else float("nan")
            # Operator (key-side) effective rank and its fraction of d.
            row[f"op_rank{tag}_worst"] = op["worst"]
            row[f"op_rank{tag}_median"] = op["median"]
            row[f"op_rank{tag}_worst_frac"] = op["worst"] / dim if op["count"] else float("nan")
            row[f"op_rank{tag}_median_frac"] = op["median"] / dim if op["count"] else float("nan")
        row["qres_num_clusters"] = _agg(qr_thresh[VAR_THRESHOLDS[0]])["count"]
        row["op_num_anchors"] = _agg(op_thresh[VAR_THRESHOLDS[0]])["count"]
        # Operator rank of the BROADEST anchor (key-side worst case, stratified).
        if op_breadth:
            bmax_idx = int(torch.tensor(op_breadth).argmax())
            row["op_broadest_nucleus_frac"] = float(op_breadth[bmax_idx])
            row["op_broadest_rank95"] = float(op_thresh[0.95][bmax_idx]) if op_thresh[0.95] else float("nan")
        rows.append(row)
    return rows


def _synthetic_blocks(device):
    """Smoke blocks: two tight clusters (low residual rank by construction) at
    d=32 and d=64, so the query-residual d' should come out small — a sanity
    check that the SVD pipeline detects low-rank residuals."""
    torch.manual_seed(0)
    blocks = {}
    for dim in (32, 64):
        bh, n = 4, 512
        # Two well-separated centers; within-cluster spread confined to a
        # 4-dim subspace so the true residual rank is ~4 ≪ dim.
        centers = torch.randn(2, dim) * 5.0
        assign = torch.randint(0, 2, (bh, n))
        low = torch.randn(bh, n, 4)
        basis = torch.randn(4, dim)
        Q = centers[assign] + low @ basis * 0.3
        K = torch.randn(bh, n, dim)
        V = torch.randn(bh, n, dim)
        blocks[dim] = (Q.to(device), K.to(device), V.to(device))
    return blocks


def run_smoke(device):
    print("SMOKE: synthetic two-cluster residuals (true rank ~4)")
    blocks = _synthetic_blocks(device)
    out = []
    for dim, (Q, K, V) in blocks.items():
        for row in probe_block(Q, K, V, dim):
            out.append(row)
            print(f"  d={dim} m={row['m']:3d}  "
                  f"qres_d'@95 worst={row['qres_dprime95_worst']:.0f} "
                  f"median={row['qres_dprime95_median']:.0f} "
                  f"(frac worst={row['qres_dprime95_worst_frac']:.2f})  "
                  f"op_rank@95 worst={row['op_rank95_worst']:.0f} "
                  f"(frac {row['op_rank95_worst_frac']:.2f})")
    return out


def capture_decoder_d128(model_id, corpus, layer, device, n_target=512, max_docs=16):
    """Capture per-head Q/K/V from a real decoder LM whose head_dim is 128.

    The BERT-encoder loader in the fidelity harness tops out at head_dim=64 (no
    standard encoder is wider), but the paper's target per-head dim is 128. A
    decoder LM such as ``EleutherAI/gpt-neo-1.3B`` (hidden 2048 / 16 heads = 128)
    is a genuine trained model at that width, so its activations are a real
    d=128 signal for the rank probe. This is deliberately standalone (does not
    touch the BERT-specific ``load_real_activations``): it hooks the layer's
    separate ``q_proj``/``k_proj``/``v_proj`` outputs and folds heads into the
    batch axis, matching the harness's per-head convention.

    Caveats recorded on the returned block: the model is a *decoder* (attention
    is causal in the LM), but this probe measures only the *geometry* of Q/K/V —
    the query-residual subspace and the operator spectrum — which does not depend
    on the attention mask. It is a single native forward over real text of
    ``n_target`` tokens (no cross-document tiling). Returned as a one-entry list of
    ``(Q, K, V, dim)`` with provenance ``native_decoder``.
    """
    import os

    # The BERT loader may have set HF_HUB_OFFLINE=1 during its own attempts;
    # this decoder capture needs the hub, so force online explicitly (not
    # setdefault, which cannot override an already-set value).
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import datasets
    except Exception:
        return None

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model = model.eval().to(device)

    # Locate the transformer block list and the requested layer's attention.
    base = getattr(model, "transformer", getattr(model, "model", model))
    blocks_attr = getattr(base, "h", getattr(base, "layers", None))
    n_layers = len(blocks_attr)
    layer = max(0, min(layer, n_layers - 1))
    attn = blocks_attr[layer].attn
    inner = getattr(attn, "attention", attn)  # gpt-neo wraps in .attention
    num_heads = getattr(inner, "num_heads", None)
    head_dim = getattr(inner, "head_dim", None)

    # Real text: concatenate sentences until the token budget is reached.
    try:
        ds = datasets.load_dataset("opus_books", "de-en", split="train")
        texts = [ex["translation"]["en"] for ex in ds.select(range(2000))]
    except Exception:
        return None
    buf = ""
    for t in texts:
        buf += (" " + t)
        if len(tok(buf)["input_ids"]) >= n_target:
            break
    ids = tok(buf, return_tensors="pt", truncation=True,
              max_length=n_target)["input_ids"].to(device)

    captured = {}

    def mk(name):
        def hook(_m, _in, out):
            captured[name] = out.detach()
        return hook

    hs = [inner.q_proj.register_forward_hook(mk("q")),
          inner.k_proj.register_forward_hook(mk("k")),
          inner.v_proj.register_forward_hook(mk("v"))]
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()

    def fold(x):  # (1, n, embed) -> (num_heads, n, head_dim)
        n = x.shape[1]
        return x.view(1, n, num_heads, head_dim).permute(0, 2, 1, 3).reshape(
            num_heads, n, head_dim)

    Q, K, V = fold(captured["q"]), fold(captured["k"]), fold(captured["v"])
    return {"Q": Q, "K": K, "V": V, "dim": head_dim, "layer": layer,
            "seq_len": Q.shape[1], "provenance": "native_decoder",
            "model_id": model_id, "num_heads": num_heads}


def run_real(models, corpus, layers, device, d128_model=None):
    all_rows = []
    for model_id in models:
        blocks = load_real_activations(
            model_id, corpus, [NATIVE_LEN], layers, device
        )
        if not blocks:
            print(f"SKIP_NO_SOURCE: {model_id}")
            continue
        for key, block in blocks.items():
            if key == "_meta":
                continue
            seq_len, layer = key
            if block["provenance"] != "native_concat":
                continue  # native only
            Q, K, V = block["Q"].to(device), block["K"].to(device), block["V"].to(device)
            dim = Q.shape[-1]
            print(f"\n{model_id} L{layer} n={seq_len} d={dim} (bh={Q.shape[0]})")
            for row in probe_block(Q, K, V, dim):
                row.update(model_id=model_id, layer=layer, seq_len=seq_len,
                           provenance=block["provenance"])
                all_rows.append(row)
                print(f"  m={row['m']:3d}  "
                      f"qres_d'@95 worst={row['qres_dprime95_worst']:.0f}/{dim} "
                      f"(frac {row['qres_dprime95_worst_frac']:.2f})  "
                      f"median={row['qres_dprime95_median']:.0f} "
                      f"(frac {row['qres_dprime95_median_frac']:.2f})  |  "
                      f"op_rank@95 worst={row['op_rank95_worst']:.0f}/{dim} "
                      f"(frac {row['op_rank95_worst_frac']:.2f}) "
                      f"broadest_nuc={row.get('op_broadest_nucleus_frac', float('nan')):.2f}")

    # Real d=128 signal from a decoder LM (the paper's target per-head dim, not
    # reachable via any BERT encoder). Treated as native_decoder provenance.
    if d128_model:
        for layer in layers:
            blk = capture_decoder_d128(d128_model, corpus, layer, device)
            if blk is None:
                print(f"SKIP_NO_SOURCE (d128): {d128_model}")
                break
            Q, K, V = blk["Q"].to(device), blk["K"].to(device), blk["V"].to(device)
            dim = blk["dim"]
            print(f"\n{d128_model} L{blk['layer']} n={blk['seq_len']} d={dim} "
                  f"(bh={Q.shape[0]}) [decoder, causal LM — geometry probe]")
            for row in probe_block(Q, K, V, dim):
                row.update(model_id=d128_model, layer=blk["layer"],
                           seq_len=blk["seq_len"], provenance="native_decoder")
                all_rows.append(row)
                print(f"  m={row['m']:3d}  "
                      f"qres_d'@95 worst={row['qres_dprime95_worst']:.0f}/{dim} "
                      f"(frac {row['qres_dprime95_worst_frac']:.2f})  "
                      f"median={row['qres_dprime95_median']:.0f} "
                      f"(frac {row['qres_dprime95_median_frac']:.2f})  |  "
                      f"op_rank@95 worst={row['op_rank95_worst']:.0f}/{dim} "
                      f"(frac {row['op_rank95_worst_frac']:.2f}) "
                      f"broadest_nuc={row.get('op_broadest_nucleus_frac', float('nan')):.2f}")
    return all_rows


def _print_verdict(rows):
    """Read the gate: query-residual d'@95% worst-fraction at m=64, and its trend
    vs m and vs d. Print an honest greenlight / key-side-negative call."""
    print("\n" + "=" * 72)
    print("VERDICT — query-residual PCA d² lever (issue #9)")
    print("=" * 72)
    real = [r for r in rows if r.get("provenance") in ("native_concat", "native_decoder")]
    if not real:
        print("  no native real-activation rows — verdict deferred")
        return
    by_dm = {}
    for r in real:
        by_dm.setdefault((r["dim"], r["m"]), []).append(r)

    print("  query-residual d'@95% (WORST anchor, the gate quantity) — frac of d:")
    for (d, m) in sorted(by_dm):
        fr = [r["qres_dprime95_worst_frac"] for r in by_dm[(d, m)]]
        med = [r["qres_dprime95_median_frac"] for r in by_dm[(d, m)]]
        fr = [x for x in fr if x == x]
        med = [x for x in med if x == x]
        if fr:
            print(f"    d={d:3d} m={m:3d}:  worst {min(fr):.2f}-{max(fr):.2f}   "
                  f"median {min(med):.2f}-{max(med):.2f}")
    print("  operator (key-side) rank@95% (WORST anchor) — frac of d:")
    for (d, m) in sorted(by_dm):
        fr = [r["op_rank95_worst_frac"] for r in by_dm[(d, m)] if r["op_rank95_worst_frac"] == r["op_rank95_worst_frac"]]
        if fr:
            print(f"    d={d:3d} m={m:3d}:  worst {min(fr):.2f}-{max(fr):.2f}")

    # Gate call at the largest available d, m=64.
    dmax = max(d for (d, m) in by_dm)
    key = (dmax, 64)
    if key in by_dm:
        worst = [r["qres_dprime95_worst_frac"] for r in by_dm[key]
                 if r["qres_dprime95_worst_frac"] == r["qres_dprime95_worst_frac"]]
        if worst:
            w = max(worst)
            print(f"\n  GATE @ d={dmax}, m=64: worst-anchor query-residual d'@95% = "
                  f"{w:.2f}·d")
            if w <= 0.5:
                print("  -> query side is LOW RANK: greenlight building the reduced "
                      "M_j^red and measuring end-to-end rel-err (target <=0.24).")
            else:
                print("  -> query side is NOT clearly low rank at this d; check the "
                      "d'/d trend vs d (needs d=128) before greenlight/kill.")
    print("  NOTE: native d reaches only 64 here (bert-base); the paper target is "
          "d=128.\n  Read the d'/d TREND across d=32->64 and m=16->64, not the "
          "absolute frac alone.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--models", nargs="+",
                    default=["sentence-transformers/all-MiniLM-L6-v2",
                             "bert-base-uncased"])
    ap.add_argument("--corpus", type=str, default="opus_books")
    ap.add_argument("--layers", nargs="+", type=int, default=[3, 6])
    ap.add_argument("--d128-model", type=str, default=None,
                    help="Real decoder LM with head_dim=128 for the d=128 rank "
                         "signal (e.g. EleutherAI/gpt-neo-1.3B).")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    if args.smoke:
        rows = run_smoke(device)
        _print_verdict([{**r, "provenance": "native_concat"} for r in rows])
        payload = {"mode": "smoke", "rows": rows}
    else:
        rows = run_real(args.models, args.corpus, args.layers, device,
                        d128_model=args.d128_model)
        _print_verdict(rows)
        payload = {
            "mode": "real",
            "platform": platform.platform(),
            "torch": torch.__version__,
            "var_thresholds": list(VAR_THRESHOLDS),
            "anchor_counts": list(ANCHOR_COUNTS),
            "rows": rows,
        }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
