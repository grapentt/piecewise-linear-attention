#!/usr/bin/env python
"""Real-activation approximation-fidelity vs sequence length.

All prior fidelity numbers for anchor-Taylor attention were measured on short,
low-dispersion synthetic Gaussian queries. The long-context claim rests on a
distribution we had never tested: the non-Gaussian query clouds a *trained*
model actually produces. This probe measures relative error to exact softmax on
per-head Q/K/V captured from genuinely pretrained encoders, as a function of
sequence length and of the measured query dispersion E‖q − a‖².

Honesty note on sequence length
-------------------------------
No offline encoder here exceeds 512 positions, so only n=512 activations are a
native forward pass of the model (``native_concat``: consecutive real sentences
from one source, concatenated to 512 unpadded tokens). Longer lengths are a
labeled ``tiled_proxy``: per-head Q/K/V from several *distinct* source
documents' native blocks, concatenated along the sequence axis. This is a
cross-document mixture, NOT a coherent long document — it raises query
dispersion but also flattens the attention landscape (a query over keys from
unrelated documents sees near-uniform softmax), a regime that happens to favor a
first-order approximation. The verdict therefore treats long-n results as
proxy-only: a proxy can trigger a scope reduction or a kill, but can never mint
an unqualified pass. Every row carries an ``n_provenance`` and ``is_real`` flag.

Comparators (all consume the same captured (B·H, n, d) tensors)
---------------------------------------------------------------
- ``piecewise_mean`` (m=1) and ``piecewise_kmeans_m{4,16,64}`` — anchor-Taylor.
- ``performer_M{256,512}`` and ``nystrom_L{64,256}`` — efficient-attention
  baselines. Piecewise is scored against the *best* (lowest-error) baseline.
- exact softmax reference, computed with query-row chunking at long n so the
  (B·H, n, n) score matrix is never fully materialized.

The probe is inference-only and runs on CPU/MPS. Reaching the activation source
requires optional extras (``transformers``, ``datasets``); if they or the model
weights are unavailable offline the probe records a ``SKIP_NO_SOURCE`` verdict
rather than fabricating activations.

Examples
--------
Smoke test (synthetic tensors, no model download, no optional deps)::

    python benchmarks/harness/run_real_activation_fidelity.py --smoke

Full run::

    python benchmarks/harness/run_real_activation_fidelity.py \\
        --out results/real_activation_fidelity.json --device cpu
"""

import argparse
import math
import platform
import statistics
import sys
from pathlib import Path

# Allow running directly from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from piecewise_linear_attention.core.attention import NystromformerAttention
from piecewise_linear_attention.core.registry import ATTENTION_REGISTRY, build_attention
from piecewise_linear_attention.harness import get_device, set_seed
from piecewise_linear_attention.harness.results import ExperimentResult, save_result

# Guard against a stale package that predates the baselines this probe compares
# against; without them the sweep would silently drop comparators.
assert "piecewise_kmeans" in ATTENTION_REGISTRY and "nystromformer" in ATTENTION_REGISTRY, (
    "attention registry is stale relative to this script (missing "
    "piecewise_kmeans / nystromformer)."
)

# --- Method labels (honest, self-describing; used as JSON keys and legends). --
M_PW1 = "piecewise_mean"
M_PW4 = "piecewise_kmeans_m4"
M_PW16 = "piecewise_kmeans_m16"
M_PW64 = "piecewise_kmeans_m64"
M_PERF256 = "performer_M256"
M_PERF512 = "performer_M512"
M_NYS64 = "nystrom_L64"
M_NYS256 = "nystrom_L256"

PIECEWISE_METHODS = [M_PW1, M_PW4, M_PW16, M_PW64]
PIECEWISE_M = {M_PW1: 1, M_PW4: 4, M_PW16: 16, M_PW64: 64}
BASELINE_METHODS = [M_PERF256, M_PERF512, M_NYS64, M_NYS256]
NYSTROM_LANDMARKS = {M_NYS64: 64, M_NYS256: 256}
METHOD_NAMES = PIECEWISE_METHODS + BASELINE_METHODS

# Provenance of each sequence length. Only 512 is a native forward pass; longer
# lengths are cross-document tilings, disclosed as proxy.
NATIVE_LEN = 512

# --- Verdict thresholds. ------------------------------------------------------
CEIL = 0.5  # acceptance ceiling on piecewise-mean mean rel-err
TAIL_CEIL = 0.75  # ceiling on the p95 per-query tail
MARGIN = 0.05  # relative margin piecewise must beat the best baseline by
KILL_P99 = 1.0  # p99 blowup that exposes a mean-only "pass"
EPS = 1e-8

# The long length whose result the verdict keys on (the stress case).
STRESS_LEN = 8192

VERDICT_ORDER = [
    "SKIP_NO_SOURCE",
    "KILL",
    "SCOPE_GROW_M",
    "SCOPE_RESTRICT_N",
    "PASS",
    "PASS_PROXY_ONLY",
    "FAIL",
]


# --- Pure metric helpers (no optional deps; unit-tested). ---------------------
def per_query_rel_err(approx, ref, eps=EPS):
    """Per-query relative error summary of two (B·H, n, d) tensors.

    Relative error is computed per query row (norm over the feature axis), never
    as a single whole-tensor Frobenius scalar, so a per-query tail cannot hide
    behind a benign mean. Returns ``(summary_dict, raw_vector)`` where the
    summary has mean / median / p95 / p99. Quantiles are computed on CPU
    (``torch.quantile`` is fragile on MPS).
    """
    num = (approx - ref).norm(dim=-1)  # (B·H, n)
    den = ref.norm(dim=-1).clamp_min(eps)  # (B·H, n)
    e = (num / den).reshape(-1).detach().float().cpu()
    return (
        {
            "mean": float(e.mean()),
            "median": float(e.median()),
            "p95": float(torch.quantile(e, 0.95)),
            "p99": float(torch.quantile(e, 0.99)),
        },
        e,
    )


def block_dispersion(Q):
    """Per-query squared distance to the mean anchor a = mean_i q_i.

    This is exactly the m=1 anchor of ``piecewise_mean``, so the returned
    dispersion is the quantity the first-order error bound is written against.
    Returns ``(per_query_vector (B·H·n,), scalar_mean)``.
    """
    a = Q.mean(dim=1, keepdim=True)  # (B·H, 1, d)
    disp = ((Q - a) ** 2).sum(dim=-1).reshape(-1).detach().float().cpu()  # (B·H·n,)
    return disp, float(disp.mean())


def chunked_softmax_reference(Q, K, V, row_chunk=1024):
    """Exact scaled-dot-product softmax attention, tiled over query rows.

    Materializes at most a (B·H, row_chunk, n) score block instead of the full
    (B·H, n, n) matrix, so the exact reference stays in memory at long n on CPU.
    Kept in the harness rather than added to the library so no shipped module
    changes.
    """
    bh, n, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(Q)
    for start in range(0, n, row_chunk):
        stop = min(start + row_chunk, n)
        scores = torch.matmul(Q[:, start:stop] * scale, K.transpose(-2, -1))  # (bh, c, n)
        attn = torch.softmax(scores, dim=-1)
        out[:, start:stop] = torch.matmul(attn, V)
    return out


# Cumulative-mass thresholds and pooled summary stats for the nucleus measure.
NUCLEUS_PS = (0.9, 0.99, 0.999)
NUCLEUS_TAGS = {0.9: "p90", 0.99: "p99", 0.999: "p999"}
NUCLEUS_STATS = ("maxfrac", "p95frac", "medfrac", "meanfrac")


def measure_nucleus_sizes(anchors, K, ps=NUCLEUS_PS):
    """Per-anchor nucleus fractions at each threshold p, mirroring the shipped
    top-p build rule exactly (``PiecewiseAttention._build_jacobian_factors_topp``).

    Recomputes the per-anchor softmax the same way ``_compute_multi_anchor`` does
    (``scores = einsum('bmd,bnd->bmn', anchors, K) / sqrt(dim)``; softmax over
    keys), then counts, per anchor, the smallest set of top-weighted keys whose
    cumulative softmax mass reaches ``p`` -- *including* the key that crosses the
    threshold (keep where the *preceding* cumsum ``< p``). The first key is always
    kept, so the count is in ``[1, n]``. Weights are not renormalized (irrelevant
    to the count). This is the size ``t`` the top-p build's batched cap would use.

    All math is float32 to match the library's fp32 discipline and to keep the
    ``< p`` boundary deterministic at the crossing key. No gather/matmul is done --
    only the counts that would determine the build's ``t``.

    Args:
        anchors: (b, m, d) anchor tensor (k-means centroids, or the m=1 mean).
        K:       (b, n, d) key tensor.
        ps:      iterable of cumulative-mass thresholds.

    Returns:
        dict[float, torch.Tensor]: ``p -> (b, m)`` float32 CPU tensor of nucleus
        *fractions* (count / n), one entry per (batch, anchor).
    """
    anchors = anchors.float()
    Kf = K.float()
    dim = anchors.shape[-1]
    scale = 1.0 / math.sqrt(dim)
    scores = torch.einsum("bmd,bnd->bmn", anchors, Kf) * scale  # (b, m, n)
    alpha = torch.softmax(scores, dim=-1)  # (b, m, n)
    n = alpha.shape[-1]

    sorted_alpha, _ = torch.sort(alpha, dim=-1, descending=True)  # (b, m, n)
    cumulative = sorted_alpha.cumsum(dim=-1)
    out = {}
    for p in ps:
        keep = torch.ones_like(sorted_alpha, dtype=torch.bool)
        keep[..., 1:] = cumulative[..., :-1] < p  # library rule; crossing key kept
        counts = keep.sum(dim=-1)  # (b, m), in [1, n]
        out[p] = (counts.float() / n).detach().cpu()  # (b, m) fractions on CPU
    return out


def excess_kurtosis(Q):
    """Mean per-feature excess kurtosis of the query cloud (report-only).

    A near-zero value flags an approximately-Gaussian cloud — the distribution
    this probe exists to escape. Diagnostic only; it does not gate the verdict.
    """
    x = Q.reshape(-1, Q.shape[-1]).detach().float().cpu()
    mu = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False).clamp_min(EPS)
    m4 = ((x - mu) ** 4).mean(dim=0)
    return float((m4 / var**2 - 3.0).mean())


# --- Comparators. -------------------------------------------------------------
def build_methods(dim, device):
    """name -> attention module on ``device`` for per-head dimension ``dim``.

    Piecewise m=1 is plain ``piecewise`` (mean anchor); m>1 routes through
    ``piecewise_kmeans`` with the requested anchor count. Performer feature
    counts and Nyström landmark counts are set explicitly (the registry defaults
    are far smaller and would hobble the baselines).
    """
    methods = {
        M_PW1: build_attention("piecewise", dim=dim),
        M_PW4: build_attention("piecewise_kmeans", dim=dim, num_anchors=4, kmeans_iters=3),
        M_PW16: build_attention("piecewise_kmeans", dim=dim, num_anchors=16, kmeans_iters=3),
        M_PW64: build_attention("piecewise_kmeans", dim=dim, num_anchors=64, kmeans_iters=3),
        M_PERF256: build_attention("performer", dim=dim, num_features=256, performer_seed=0),
        M_PERF512: build_attention("performer", dim=dim, num_features=512, performer_seed=0),
        M_NYS64: build_attention("nystromformer", dim=dim, num_landmarks=64, pinv_iterations=6),
        M_NYS256: build_attention("nystromformer", dim=dim, num_landmarks=256, pinv_iterations=6),
    }
    return {k: v.to(device) for k, v in methods.items()}


# Piecewise m>=16 materializes (B·H, m, n, d) intermediates; force a single
# head at long n so the sweep stays within CPU/MPS memory.
def _batch_heads_cap(method, seq_len, default_cap):
    if PIECEWISE_M.get(method, 1) >= 16 and seq_len >= 2048:
        return 1
    return default_cap


# --- Activation source (optional deps; isolated so import failure => SKIP). ---
def load_real_activations(model_id, corpus, lengths, layers, device, max_docs=64):
    """Capture per-head Q/K/V from a pretrained encoder over a real corpus.

    Returns ``blocks[(seq_len, layer)] = {"Q","K","V","provenance","is_real",
    "n_effective","source_indices","dispersion","per_span_dispersion_mean",
    "kurtosis"}`` or ``None`` if the optional deps / weights / corpus are not
    available offline (the caller maps ``None`` to SKIP_NO_SOURCE).

    n=512 blocks are a native forward pass (consecutive real sentences from one
    source concatenated to 512 unpadded tokens). Longer blocks tile per-head
    Q/K/V from distinct source documents along the sequence axis — a disclosed
    cross-document proxy.
    """
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoModel, AutoTokenizer
        import datasets
    except Exception:  # pragma: no cover - exercised only without extras
        return None

    try:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception:
            # Some cached tokenizers need optional converters (e.g. protobuf).
            # These BERT-family encoders share the bert-base WordPiece vocab, so
            # a cached bert-base tokenizer is an exact stand-in for capturing
            # activations (only the token ids matter, and the vocab is identical).
            tok = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        try:
            model = AutoModel.from_pretrained(model_id, local_files_only=True)
        except Exception:
            # Some cached configs (e.g. bert-tiny) omit the `model_type` key that
            # AutoModel dispatches on; fall back to the concrete BERT class.
            from transformers import BertModel

            model = BertModel.from_pretrained(model_id, local_files_only=True)
        model = model.eval().to(device)
    except Exception:  # pragma: no cover
        return None

    # Discover the self-attention layout from the model rather than hardcoding.
    # Clamp requested layers to those the model actually has (small models such
    # as bert-tiny have only 2 layers), keeping mid layers where possible.
    try:
        n_model_layers = len(model.encoder.layer)
        eff_layers = [L for L in layers if 0 <= L < n_model_layers]
        if not eff_layers:
            # Fall back to the deepest available layer(s) (still "mid" for a
            # 2-layer model) rather than dropping the source entirely.
            eff_layers = sorted({max(0, n_model_layers - 2), n_model_layers - 1})
        self_attns = [model.encoder.layer[L].attention.self for L in eff_layers]
        num_heads = self_attns[0].num_attention_heads
        head_dim = self_attns[0].attention_head_size
    except Exception:  # pragma: no cover
        return None
    layers = eff_layers

    # A short, topically diverse text pool. opus_books gives longer, more varied
    # sentences than a sentiment corpus, so tiled spans are less self-similar.
    try:
        if corpus == "opus_books":
            ds = datasets.load_dataset(
                "opus_books", "de-en", split="train", streaming=False
            )
            texts = [ex["translation"]["en"] for ex in ds.select(range(2000))]
        else:
            ds = datasets.load_dataset("glue", "sst2", split="train")
            texts = [ex["sentence"] for ex in ds.select(range(4000))]
    except Exception:  # pragma: no cover
        return None

    # Build documents: concatenate consecutive sentences until each reaches the
    # native token budget, so a "document" is a real block of contiguous text.
    def make_documents(n_target, want):
        docs, buf = [], ""
        for t in texts:
            buf = (buf + " " + t).strip()
            ids = tok(buf, add_special_tokens=True, truncation=False)["input_ids"]
            if len(ids) >= n_target:
                docs.append(buf)
                buf = ""
                if len(docs) >= want:
                    break
        return docs

    def capture(text_batch, per_layer):
        """Run one forward pass; fill per_layer[layer_idx] with folded Q/K/V."""
        enc = tok(
            text_batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=NATIVE_LEN,
        ).to(device)
        mask = enc["attention_mask"]  # (B, n)
        captured = {}

        def mk_hook(idx, sa):
            def hook(_mod, args, kwargs):
                hs = kwargs.get("hidden_states", args[0] if args else None)
                B, n, _ = hs.shape

                def fold(proj):
                    return (
                        proj(hs)
                        .view(B, n, num_heads, head_dim)
                        .transpose(1, 2)
                        .reshape(B * num_heads, n, head_dim)
                    )

                captured[idx] = (fold(sa.query), fold(sa.key), fold(sa.value))

            return hook

        handles = [
            sa.register_forward_pre_hook(mk_hook(i, sa), with_kwargs=True)
            for i, sa in enumerate(self_attns)
        ]
        with torch.no_grad():
            model(**enc)
        for h in handles:
            h.remove()

        # Drop padded positions per sequence, keep only real-token rows. The
        # per-head fold repeats the (B, n) mask across heads.
        keep = mask.reshape(B_times := mask.shape[0], -1)  # (B, n)
        for idx in captured:
            Q, K, V = captured[idx]
            head_mask = keep.repeat_interleave(num_heads, dim=0).bool()  # (B*H, n)
            # Flatten to (sum_real, d) then regroup by sequence for tiling.
            per_layer.setdefault(idx, [])
            per_layer[idx].append((Q, K, V, head_mask))

    # --- native n=512 block: one document, first real span, unpadded. --------
    native_docs = make_documents(NATIVE_LEN, want=max_docs)
    if len(native_docs) < 1:
        return None

    blocks = {}

    # Native block per layer: take the first document, keep exactly its real
    # tokens truncated/asserted to NATIVE_LEN.
    per_layer_native = {}
    capture([native_docs[0]], per_layer_native)
    for li, layer in enumerate(layers):
        Q, K, V, head_mask = per_layer_native[li][0]
        # Single document => single (B=1) sequence; select its real rows.
        row = head_mask[0]  # (n,) shared across this doc's heads within the head block
        # head_mask rows for this doc's H heads are identical (same seq mask).
        H = num_heads
        Qb = Q[:H][:, row]  # (H, n_eff, d)
        Kb = K[:H][:, row]
        Vb = V[:H][:, row]
        n_eff = int(row.sum())
        disp_vec, disp_mean = block_dispersion(Qb)
        blocks[(NATIVE_LEN, layer)] = {
            "Q": Qb, "K": Kb, "V": Vb,
            "provenance": "native_concat", "is_real": True,
            "n_effective": n_eff, "source_indices": [0],
            "dispersion": disp_mean, "per_span_dispersion_mean": disp_mean,
            "kurtosis": excess_kurtosis(Qb),
        }

    # --- tiled proxy blocks for n>512: distinct documents per span. ----------
    for seq_len in [n for n in lengths if n > NATIVE_LEN]:
        n_spans = seq_len // NATIVE_LEN
        docs = make_documents(NATIVE_LEN, want=n_spans)
        if len(docs) < n_spans:
            # Not enough distinct source material for an honest tiling: skip.
            continue
        per_layer_tiled = {}
        capture(docs[:n_spans], per_layer_tiled)
        for li, layer in enumerate(layers):
            Q, K, V, head_mask = per_layer_tiled[li][0]
            H = num_heads
            span_Q, span_K, span_V, per_span_disp = [], [], [], []
            for s in range(n_spans):
                rows = head_mask[s * H]  # this span's seq mask
                span_Q.append(Q[s * H:(s + 1) * H][:, rows])
                span_K.append(K[s * H:(s + 1) * H][:, rows])
                span_V.append(V[s * H:(s + 1) * H][:, rows])
                _, dmean = block_dispersion(span_Q[-1])
                per_span_disp.append(dmean)
            Qb = torch.cat(span_Q, dim=1)  # (H, sum_real, d)
            Kb = torch.cat(span_K, dim=1)
            Vb = torch.cat(span_V, dim=1)
            disp_vec, disp_mean = block_dispersion(Qb)
            blocks[(seq_len, layer)] = {
                "Q": Qb, "K": Kb, "V": Vb,
                "provenance": "tiled_proxy", "is_real": False,
                "n_effective": int(Qb.shape[1]), "source_indices": list(range(n_spans)),
                "dispersion": disp_mean,
                "per_span_dispersion_mean": statistics.fmean(per_span_disp),
                "kurtosis": excess_kurtosis(Qb),
            }

    blocks["_meta"] = {"num_heads": num_heads, "head_dim": head_dim, "layers": layers}
    return blocks


def _reshape_equivalence_ok(block, dim, device, tol=1e-3):
    """Sanity: captured per-head Q/K/V run through exact softmax are finite and
    match a chunked recompute — a scrambled permute would diverge here.
    """
    Q, K, V = block["Q"].to(device), block["K"].to(device), block["V"].to(device)
    ref_a = chunked_softmax_reference(Q, K, V, row_chunk=256)
    ref_b = build_attention("standard", dim=dim).to(device)(Q, K, V)[0]
    return torch.isfinite(ref_a).all() and (ref_a - ref_b).norm() / (ref_b.norm() + EPS) < tol


# --- Grid + verdict. ----------------------------------------------------------
def run_grid(blocks, model_id, dim, layers, lengths, device, batch_heads_cap):
    rows = []
    keys = [(n, L) for n in lengths for L in layers if (n, L) in blocks]
    for (seq_len, layer) in keys:
        block = blocks[(seq_len, layer)]
        Qf, Kf, Vf = block["Q"], block["K"], block["V"]
        for method in METHOD_NAMES:
            cap = _batch_heads_cap(method, seq_len, batch_heads_cap)
            bh = min(cap, Qf.shape[0])
            Q, K, V = Qf[:bh].to(device), Kf[:bh].to(device), Vf[:bh].to(device)
            module = build_methods(dim, device)[method]
            skipped = False
            try:
                with torch.no_grad():
                    ref = chunked_softmax_reference(Q, K, V)
                    approx, _ = module(Q, K, V)
                summary, _ = per_query_rel_err(approx, ref)
                # Baseline-fairness asserts: no exact-softmax fallback, requested
                # landmarks actually used.
                if method in NYSTROM_LANDMARKS:
                    L = NYSTROM_LANDMARKS[method]
                    assert seq_len > L, f"{method}: n={seq_len} <= L={L} (would hit exact fallback)"
                    assert isinstance(module, NystromformerAttention)
                    assert module.last_num_landmarks == L, (
                        f"{method}: used {module.last_num_landmarks} landmarks, requested {L}"
                    )
                fro = float((approx - ref).norm() / (ref.norm() + EPS))
            except (RuntimeError, MemoryError) as exc:  # pragma: no cover - OOM path
                skipped = True
                summary = {"mean": float("nan"), "median": float("nan"),
                           "p95": float("nan"), "p99": float("nan")}
                fro = float("nan")
                print(f"  SKIP {method} n={seq_len} L{layer}: {type(exc).__name__}")

            # Top-10% dispersion bucket rel-err (per-query, on the same rows).
            top10 = float("nan")
            if not skipped:
                disp_vec, _ = block_dispersion(Q)
                num = (approx - ref).norm(dim=-1).reshape(-1).detach().float().cpu()
                den = ref.norm(dim=-1).reshape(-1).clamp_min(EPS).detach().float().cpu()
                e = num / den
                k = max(1, int(0.10 * e.numel()))
                idx = torch.topk(disp_vec, k).indices
                top10 = float(e[idx].mean())

            # Nucleus-size measurement (piecewise only; purely observational).
            # Measures the per-anchor softmax nucleus that the top-p build would
            # truncate to -- the sequence-axis shrink ceiling for the build cost.
            # Needs only anchors + K (no second module forward): k-means
            # anchor_strategy.select is deterministic and reproduces the module's
            # own anchors; the m=1 "piecewise" path has no strategy (its anchor is
            # the query mean), and its dense forward does not use build_topp, so
            # its nucleus is diagnostic-only (nucleus_applies_to_build=0).
            nucleus_fields = {f"nucleus_{t}_{s}": float("nan")
                              for t in NUCLEUS_TAGS.values() for s in NUCLEUS_STATS}
            nucleus_fields.update(nucleus_n_used=float("nan"),
                                  nucleus_m_effective=float("nan"),
                                  nucleus_n_anchors=float("nan"),
                                  nucleus_applies_to_build=float("nan"))
            if method in PIECEWISE_METHODS and not skipped:
                try:
                    with torch.no_grad():
                        strat = getattr(module, "anchor_strategy", None)
                        if strat is None:  # m=1 plain piecewise (mean anchor)
                            anchors = Q.mean(dim=1, keepdim=True)  # (bh, 1, d)
                            applies = 0.0
                        else:  # k-means; deterministic re-select matches the module
                            anchors = strat.select(Q)  # (bh, m, d)
                            applies = 1.0
                        fracs = measure_nucleus_sizes(anchors, K)  # {p: (bh, m) cpu}
                    m_eff = anchors.shape[1]
                    nucleus_fields["nucleus_n_used"] = float(K.shape[1])
                    nucleus_fields["nucleus_m_effective"] = float(m_eff)
                    nucleus_fields["nucleus_n_anchors"] = float(bh * m_eff)
                    nucleus_fields["nucleus_applies_to_build"] = applies
                    for p, f in fracs.items():
                        flat = f.reshape(-1)  # already CPU fp32
                        tag = NUCLEUS_TAGS[p]
                        nucleus_fields[f"nucleus_{tag}_maxfrac"] = float(flat.max())
                        nucleus_fields[f"nucleus_{tag}_p95frac"] = float(torch.quantile(flat, 0.95))
                        nucleus_fields[f"nucleus_{tag}_medfrac"] = float(flat.median())
                        nucleus_fields[f"nucleus_{tag}_meanfrac"] = float(flat.mean())
                except (RuntimeError, MemoryError) as exc:  # keep the rel-err row
                    print(f"  NUCLEUS-SKIP {method} n={seq_len} L{layer}: {type(exc).__name__}")

            rows.append({
                "model_id": model_id, "attn_dim": dim,
                "method": method, "seq_len": seq_len, "layer": layer,
                "n_provenance": block["provenance"], "is_real": block["is_real"],
                "n_effective": block["n_effective"],
                "tiled_source_indices": block["source_indices"],
                "measured_dispersion": block["dispersion"],
                "per_span_dispersion_mean": block["per_span_dispersion_mean"],
                "query_excess_kurtosis": block["kurtosis"],
                "batch_heads": bh, "skipped": skipped,
                "rel_err_mean": summary["mean"], "rel_err_median": summary["median"],
                "rel_err_p95": summary["p95"], "rel_err_p99": summary["p99"],
                "rel_err_frobenius": fro, "rel_err_top10pct_dispersion_bucket": top10,
                "last_num_landmarks": NYSTROM_LANDMARKS.get(method),
                **nucleus_fields,
            })
            tag = "SKIP" if skipped else f"mean={summary['mean']:.3f} p95={summary['p95']:.3f}"
            print(f"  n={seq_len:5d} L{layer} {method:20s} {tag}")
    return rows


def _worst_layer_row(rows, method, seq_len):
    """The row for (method, seq_len) on the worst (highest mean) mid layer.

    Aggregating on the worst layer, never the mean, prevents an easy-layer
    cherry-pick from flattering the verdict. Skipped cells are excluded.
    """
    cands = [r for r in rows if r["method"] == method and r["seq_len"] == seq_len
             and not r["skipped"] and not math.isnan(r["rel_err_mean"])]
    if not cands:
        return None
    return max(cands, key=lambda r: r["rel_err_mean"])


def _best_baseline_mean(rows, seq_len):
    vals = []
    for m in BASELINE_METHODS:
        r = _worst_layer_row(rows, m, seq_len)
        if r is not None:
            vals.append(r["rel_err_mean"])
    return min(vals) if vals else float("nan")


def compute_verdict(rows, lengths):
    """Resolve PASS_PROXY_ONLY / SCOPE_* / KILL / FAIL / SKIP on the stress len.

    A bare PASS requires a *native* (is_real) stress-length block, which the
    offline pipeline never produces — so the strongest honest positive outcome
    is PASS_PROXY_ONLY. KILL and both SCOPE reductions are genuinely reachable.
    """
    if not rows:
        return {"verdict": "SKIP_NO_SOURCE"}

    stress = STRESS_LEN if STRESS_LEN in lengths else max(lengths)
    # Prefer a length <= stress that is native for the restrict-n check.
    restrict_len = max([n for n in lengths if n < stress], default=None)

    def pw(method, seq_len):
        r = _worst_layer_row(rows, method, seq_len)
        return r["rel_err_mean"] if r else float("nan")

    def pw_tail(method, seq_len, key):
        r = _worst_layer_row(rows, method, seq_len)
        return r[key] if r else float("nan")

    pw1_stress = pw(M_PW1, stress)
    pw1_p95 = pw_tail(M_PW1, stress, "rel_err_p95")
    pw1_p99 = pw_tail(M_PW1, stress, "rel_err_p99")
    best_base = _best_baseline_mean(rows, stress)
    stress_is_real = any(
        r["is_real"] for r in rows if r["seq_len"] == stress and not r["skipped"]
    )

    def finite(x):
        return isinstance(x, float) and not math.isnan(x)

    beats = finite(pw1_stress) and finite(best_base) and pw1_stress <= best_base * (1 - MARGIN)
    under_ceil = finite(pw1_stress) and pw1_stress <= CEIL
    tail_ok = finite(pw1_p95) and pw1_p95 <= TAIL_CEIL
    numbers_hold = beats and under_ceil and tail_ok

    # SCOPE_GROW_M: m=1 fails the ceiling but some larger non-skipped m recovers.
    grow_m = False
    if finite(pw1_stress) and pw1_stress > CEIL:
        for m in (M_PW4, M_PW16, M_PW64):
            v = pw(m, stress)
            if finite(v) and v <= CEIL and finite(best_base) and v <= best_base:
                grow_m = True
                break

    # SCOPE_RESTRICT_N: a shorter length is fine but the stress length is not.
    restrict_n = False
    if restrict_len is not None:
        pw1_short = pw(M_PW1, restrict_len)
        base_short = _best_baseline_mean(rows, restrict_len)
        short_ok = (finite(pw1_short) and pw1_short <= CEIL
                    and finite(base_short) and pw1_short <= base_short)
        stress_bad = (not under_ceil) or (not tail_ok) or (not beats)
        restrict_n = short_ok and stress_bad

    # KILL conditions. `never_beats` (dominated by a baseline at every m) and
    # `tail_cheat` (mean sneaks under the ceiling but the p99 tail blows up) are
    # unconditional — neither is rescued by restricting the length. Every m
    # exceeding the ceiling is a KILL only if no shorter length rescues the claim
    # (otherwise it is a SCOPE_RESTRICT_N) and no larger m recovers (SCOPE_GROW_M).
    pw_all = {m: pw(m, stress) for m in PIECEWISE_METHODS}
    finite_pw = {m: v for m, v in pw_all.items() if finite(v)}
    all_over_ceil = bool(finite_pw) and all(v > CEIL for v in finite_pw.values())
    never_beats = (finite(best_base) and bool(finite_pw)
                   and all(v > best_base for v in finite_pw.values()))
    tail_cheat = finite(pw1_p99) and pw1_p99 > KILL_P99 and under_ceil
    kill_unconditional = never_beats or tail_cheat
    kill_ceiling = all_over_ceil and not (grow_m or restrict_n)

    if kill_unconditional or kill_ceiling:
        verdict = "KILL"
    elif grow_m:
        verdict = "SCOPE_GROW_M"
    elif restrict_n:
        verdict = "SCOPE_RESTRICT_N"
    elif numbers_hold and stress_is_real:
        verdict = "PASS"
    elif numbers_hold:
        verdict = "PASS_PROXY_ONLY"
    else:
        verdict = "FAIL"

    worst = _worst_layer_row(rows, M_PW1, stress)
    return {
        "verdict": verdict,
        "stress_len": stress,
        "pw_mean_at_stress": pw1_stress,
        "pw_p95_at_stress": pw1_p95,
        "pw_p99_at_stress": pw1_p99,
        "best_baseline_mean_at_stress": best_base,
        "beats_baselines_at_stress": bool(beats),
        "exceeds_ceil_at_stress": bool(not under_ceil),
        "stress_is_real": bool(stress_is_real),
        "worst_layer": worst["layer"] if worst else None,
    }


def make_figures(rows, out_path):
    # matplotlib is a benchmark-only dependency; import lazily so the module
    # (and its tested helpers) load without it.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    lengths = sorted({r["seq_len"] for r in rows})

    # F1: rel-err (median + p95 whisker) vs sequence length, one line per method.
    fig1, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHOD_NAMES:
        xs, ys, es = [], [], []
        for n in lengths:
            r = _worst_layer_row(rows, method, n)
            if r is None:
                continue
            xs.append(n)
            ys.append(r["rel_err_median"])
            es.append(max(0.0, r["rel_err_p95"] - r["rel_err_median"]))
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2, label=method)
    ax.axhline(CEIL, ls="--", color="k", lw=1, label=f"ceiling {CEIL}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length (n>512 = cross-document tiled proxy)")
    ax.set_ylabel("per-query rel. error to softmax (median, p95 whisker)")
    ax.set_title("Real-activation fidelity vs length (worst mid layer)")
    ax.legend(fontsize=7)
    fig1.tight_layout()
    p1 = Path(out_path).with_name("rel_err_vs_n.png")
    fig1.savefig(p1, dpi=120)
    plt.close(fig1)
    figs.append(str(p1))

    # F2: rel-err vs measured dispersion, one point per (method, length).
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for method in METHOD_NAMES:
        xs, ys = [], []
        for n in lengths:
            r = _worst_layer_row(rows, method, n)
            if r is None:
                continue
            xs.append(r["measured_dispersion"])
            ys.append(r["rel_err_mean"])
        if xs:
            ax2.scatter(xs, ys, label=method)
    ax2.axhline(CEIL, ls="--", color="k", lw=1)
    ax2.set_xlabel("measured query dispersion  E‖q − a‖²")
    ax2.set_ylabel("per-query rel. error to softmax (mean)")
    ax2.set_title("Real-activation fidelity vs dispersion (cross-document proxy at n>512)")
    ax2.legend(fontsize=7)
    fig2.tight_layout()
    p2 = Path(out_path).with_name("rel_err_vs_dispersion.png")
    fig2.savefig(p2, dpi=120)
    plt.close(fig2)
    figs.append(str(p2))
    return figs


def _nucleus_summary(rows):
    """Roll up the per-row nucleus measurement into an honest report block.

    The shipped top-p build uses a single batched cap ``t = max_j |nucleus_j|``,
    so the sequence-axis build shrink is bounded by ``1 / maxfrac`` (never
    ``1 / meanfrac`` -- a mean-based number is unrealizable by the one-matmul
    build). We report, per (method, provenance) on the worst mid layer:

      - the worst-anchor nucleus fraction at each threshold (``maxfrac``), and the
        derived build-axis shrink ceiling ``1 / maxfrac`` (a FLOP ratio, NOT a
        measured speedup -- the top-p path also pays a sort + two gathers), and
      - ``meanfrac`` beside it purely to expose how much ``1 / meanfrac`` would
        overstate the achievable shrink.

    Native ``n=512`` (``is_real``) is the only trustworthy peakedness signal; the
    ``tiled_proxy`` rows flatten the attention landscape with a sign that differs
    by anchor family, so they are reported separately and never treated as a
    bound. ``m=1`` is diagnostic-only (its dense forward ignores ``build_topp``).

    Returns a dict keyed by ``(method, provenance)`` for the JSON, and is also the
    source for the printed nucleus block.
    """
    out = {}
    provs = sorted({r["n_provenance"] for r in rows})
    for method in PIECEWISE_METHODS:
        for prov in provs:
            cands = [r for r in rows
                     if r["method"] == method and r["n_provenance"] == prov
                     and not r["skipped"]
                     and not math.isnan(r.get("nucleus_p99_maxfrac", float("nan")))]
            if not cands:
                continue
            # Worst (largest) nucleus layer per length, then the worst length --
            # the pessimistic cell the batched cap would actually size to.
            worst = max(cands, key=lambda r: r["nucleus_p99_maxfrac"])
            entry = {
                "seq_len": worst["seq_len"], "layer": worst["layer"],
                "is_real": worst["is_real"], "n_used": worst["nucleus_n_used"],
                "n_anchors": worst["nucleus_n_anchors"],
                "applies_to_build": worst["nucleus_applies_to_build"],
            }
            for tag in NUCLEUS_TAGS.values():
                entry[f"{tag}_maxfrac"] = worst[f"nucleus_{tag}_maxfrac"]
                entry[f"{tag}_meanfrac"] = worst[f"nucleus_{tag}_meanfrac"]
            out[f"{method}|{prov}"] = entry
    return out


def _print_nucleus(nuc):
    """Print the nucleus block with the mandated honesty caveats."""
    if not nuc:
        return
    print("\n" + "-" * 72)
    print("NUCLEUS (per-anchor softmax mass concentration; top-p build shrink ceiling)")
    print("  shrink ceiling = 1 / maxfrac @p=0.99 -- a FLOP ratio, NOT a timed speedup.")
    print("  cap is batched (t = max anchor), so mean-based shrink is unachievable.")
    for key in sorted(nuc):
        e = nuc[key]
        method, prov = key.split("|", 1)
        mx = e["p99_maxfrac"]
        mn = e["p99_meanfrac"]
        real = "native" if e["is_real"] else "PROXY(flattened; not a bound)"
        applies = "" if e["applies_to_build"] == 1.0 else "  [m=1 diagnostic: build is dense]"
        ceil = (1.0 / mx) if (isinstance(mx, float) and mx > 0) else float("nan")
        overstate = (1.0 / mn) if (isinstance(mn, float) and mn > 0) else float("nan")
        gate = "WIN-eligible" if (isinstance(mx, float) and mx <= 0.10) else "build reverts to ~dense"
        print(f"  {method:22s} n={e['seq_len']:5d} L{e['layer']} [{real}]{applies}")
        print(f"      maxfrac@0.99={mx:.3f} -> shrink<= {ceil:5.1f}x  ({gate}); "
              f"mean-based {overstate:5.1f}x would overstate")
    print("-" * 72)


def _print_banner(metrics):
    print("\n" + "=" * 72)
    v = metrics["verdict"]
    print(f"VERDICT: {v}")
    if v == "SKIP_NO_SOURCE":
        print("Pretrained weights / optional extras unavailable offline; no run.")
        print("=" * 72)
        return
    print(f"  stress length n={metrics['stress_len']}  (is_real={metrics['stress_is_real']})")
    print(f"  piecewise-mean  mean={metrics['pw_mean_at_stress']:.3f} "
          f"p95={metrics['pw_p95_at_stress']:.3f} p99={metrics['pw_p99_at_stress']:.3f}")
    print(f"  best baseline   mean={metrics['best_baseline_mean_at_stress']:.3f}  "
          f"(beats={metrics['beats_baselines_at_stress']}, "
          f"exceeds_ceil={metrics['exceeds_ceil_at_stress']})")
    if v == "PASS_PROXY_ONLY":
        print("  NOTE: long-n rows are a cross-document tiled proxy, not a coherent")
        print("  long document. Read as 'survives the mixture regime', not native.")
    if v == "KILL":
        print("  Long-context framing UNSUPPORTED on real activations at this length.")
    if v.startswith("SCOPE"):
        print("  Long-context claim requires a documented scope reduction.")
    print("=" * 72)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["sentence-transformers/all-MiniLM-L6-v2", "prajjwal1/bert-tiny"],
        help="Pretrained HF encoder ids to source activations from.",
    )
    parser.add_argument("--corpus", type=str, default="opus_books",
                        choices=["opus_books", "sst2"])
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--batch-heads-cap", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    device = get_device(args.device)
    set_seed(args.seed)
    print(f"device: {device}")

    if args.smoke:
        # Exercise the full pure pipeline on synthetic tensors: no model
        # download, no optional deps, no disk, no figures.
        dim = 32
        blocks = {}
        smoke_lengths = (512, 1024)
        for n in smoke_lengths:
            g = torch.Generator().manual_seed(n)
            H = 2
            Q = torch.randn(H, n, dim, generator=g)
            K = torch.randn(H, n, dim, generator=g)
            V = torch.randn(H, n, dim, generator=g)
            _, dmean = block_dispersion(Q)
            blocks[(n, 2)] = {
                "Q": Q, "K": K, "V": V, "provenance": "synthetic_smoke",
                "is_real": False, "n_effective": n, "source_indices": [0],
                "dispersion": dmean, "per_span_dispersion_mean": dmean,
                "kurtosis": excess_kurtosis(Q),
            }
        rows = run_grid(blocks, "synthetic", dim, [2], list(smoke_lengths), device, batch_heads_cap=2)
        assert rows, "smoke produced no rows"
        assert all(r["skipped"] or r["rel_err_mean"] >= 0.0 for r in rows)
        metrics = compute_verdict(rows, list(smoke_lengths))
        _print_banner(metrics)
        _print_nucleus(_nucleus_summary(rows))
        print("smoke OK")
        return 0

    all_rows = []
    environments = []
    skipped_all = True
    for model_id in args.models:
        print(f"\n=== activation source: {model_id} ===")
        blocks = load_real_activations(
            model_id, args.corpus, args.lengths, args.layers, device
        )
        if blocks is None:
            print(f"  unavailable offline; skipping {model_id}")
            continue
        meta = blocks.pop("_meta")
        dim = meta["head_dim"]
        eff_layers = meta["layers"]  # may differ from --layers for small models

        # Highest-risk correctness guard: a wrong per-head permute silently
        # scrambles every rel-err. Verify on the native block before trusting.
        native_key = (NATIVE_LEN, eff_layers[0])
        if native_key in blocks:
            assert _reshape_equivalence_ok(blocks[native_key], dim, device), (
                f"{model_id}: captured Q/K/V fail the reshape-equivalence check; "
                "the per-head fold is likely wrong."
            )

        # Disclose whether tiling actually raised dispersion above the per-span
        # mean (the 'adversarially harder' assumption); flag if it did not.
        tiling_inflated = True
        for n in args.lengths:
            if n <= NATIVE_LEN:
                continue
            for L in eff_layers:
                b = blocks.get((n, L))
                if b and b["dispersion"] <= b["per_span_dispersion_mean"]:
                    tiling_inflated = False

        rows = run_grid(blocks, model_id, dim, eff_layers, args.lengths, device,
                        args.batch_heads_cap)
        all_rows.extend(rows)
        environments.append({
            "model_id": model_id, "attn_dim": dim, "layers": eff_layers,
            "tiling_dispersion_inflated": tiling_inflated,
        })
        skipped_all = False

    if skipped_all:
        metrics = {"verdict": "SKIP_NO_SOURCE"}
        _print_banner(metrics)
        result = ExperimentResult(
            config={"benchmark": "real_activation_fidelity", "models": args.models,
                    "corpus": args.corpus, "lengths": args.lengths, "layers": args.layers,
                    "methods": METHOD_NAMES},
            metrics=metrics,
            environment={"device": str(device), "torch": torch.__version__,
                         "platform": platform.platform(),
                         "note": "optional extras or pretrained weights unavailable offline"},
            history={"grid": []},
        )
        if args.out:
            print(f"wrote {save_result(result, args.out)}")
        return 1

    metrics = compute_verdict(all_rows, args.lengths)
    _print_banner(metrics)
    nucleus = _nucleus_summary(all_rows)
    _print_nucleus(nucleus)
    metrics = {**metrics, "nucleus": nucleus}
    figures = make_figures(all_rows, args.out or "real_activation_fidelity.json")

    result = ExperimentResult(
        config={
            "benchmark": "real_activation_fidelity",
            "models": args.models,
            "model_provenance": "pretrained_hf",
            "activation_construction": "tiled_real_spans_cross_document_mixture",
            "corpus": args.corpus,
            "lengths": args.lengths, "layers": args.layers,
            "batch_heads_cap": args.batch_heads_cap,
            "methods": METHOD_NAMES, "performer_seed": 0,
            "ceil": CEIL, "tail_ceil": TAIL_CEIL, "margin": MARGIN, "eps": EPS,
        },
        metrics=metrics,
        environment={
            "device": str(device), "torch": torch.__version__,
            "platform": platform.platform(),
            "figures": figures, "sources": environments,
            "n_provenance_by_len": {
                str(n): ("native_concat" if n <= NATIVE_LEN else "tiled_proxy")
                for n in args.lengths
            },
        },
        history={"grid": all_rows},
    )
    if args.out:
        print(f"wrote {save_result(result, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
