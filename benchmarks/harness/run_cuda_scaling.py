#!/usr/bin/env python
"""CUDA scaling + memory probe at realistic per-head ``d`` (issue #18).

The efficiency headline for anchor-Taylor (piecewise-linear) attention rests on a
speed claim that has only ever been measured at a tiny per-head dimension. The
method's cost is ``O(batch·m·n·d²)`` and the linear-attention baselines cost
``O(batch·n·d·M)``; the extra ``d²`` factor is benign at small ``d`` but may erase
the speed lead at the realistic per-head ``d = 64-128`` a real transformer uses.
This probe measures the raw speed/memory Pareto on one GPU so that a
Pareto-domination result at ``d = 128`` can be learned from a short run rather than
after a full training pipeline.

The sweep covers per-head ``d ∈ {64, 128}``, sequence length
``n ∈ {1024, 2048, 4096, 8192, 16384}`` at a fixed ``batch·heads = 256``, both
``fp32`` and ``bf16``, both forward and forward+backward, and both non-causal and
causal. It compares piecewise-mean (``m=1``), piecewise-kmeans (``m=4`` and the
accuracy-competitive ``m=64``), Performer, Nyströmformer, Linformer, Luna, a manual
``O(n²)`` softmax reference, and a fair softmax routed through
``scaled_dot_product_attention`` with the flash and memory-efficient backends
explicitly disabled (no custom kernels for anyone).

Per grid cell it records median latency (>= 30 CUDA-event-timed runs after warmup),
peak allocated memory, and an analytic FLOP count, plus a per-method ``status`` of
``ok`` / ``oom`` / ``skipped`` (some baselines have no causal implementation and are
skipped in causal mode). The causal piecewise-mean path materialises a
``(batch, n, d, d)`` cumulative-sum tensor; that footprint is reported explicitly on
those cells because it is the dominant long-context memory risk.

The anchor count is a free hyperparameter of the piecewise method, so ``--anchors``
sweeps an arbitrary set of ``m`` values (each run as ``piecewise_kmeans_m{k}``, with
``m=1`` the mean-anchor fast path) against the fixed no-``m`` baselines on identical
inputs — this maps where the accuracy↔latency↔memory dial lands and where the dense
``(batch, m, d, d)`` build runs out of memory. With ``--accuracy`` the harness also
measures each method's output relative error against an exact dense-softmax ground
truth on one shared input, at a small ``--acc-batch`` where the ``O(n²)`` reference
still fits at long ``n`` (probed empirically); where softmax cannot fit even at the
smallest batch, accuracy is recorded as unavailable and only speed/memory is kept.

Examples
--------
Smoke (CPU, seconds, self-checking)::

    python benchmarks/harness/run_cuda_scaling.py --smoke

Full run (one GPU)::

    python benchmarks/harness/run_cuda_scaling.py \
        --device cuda --dims 64 128 --lengths 1024 2048 4096 8192 16384 \
        --dtypes fp32 bf16 --batch 256 --warmup 5 --repeats 30 \
        --modes noncausal causal --backward --out results/cuda_scaling.json

Anchor sweep with accuracy at long context (one GPU)::

    python benchmarks/harness/run_cuda_scaling.py \
        --device cuda --dims 128 --lengths 8192 16384 \
        --anchors 1 8 16 24 32 40 48 56 60 --dtypes fp32 bf16 \
        --modes noncausal causal --backward --accuracy --acc-batch 8 \
        --out results/cuda_scaling_anchors.json
"""

import argparse
import contextlib
import math
import platform
import statistics
import sys
import time
from pathlib import Path

# Allow running directly from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from piecewise_linear_attention.core.registry import build_attention
from piecewise_linear_attention.harness import get_device, set_seed
from piecewise_linear_attention.harness.results import ExperimentResult, save_result

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

# Fixed method hyperparameters (issue #18): Performer feature count, Nyström
# landmarks, Linformer projection width, Luna pack size, k-means Lloyd iterations.
_PERFORMER_FEATURES = 256
_NYSTROM_LANDMARKS = 64
_LINFORMER_K = 256
_LUNA_PACK = 64
_KMEANS_ITERS = 3

# The full method set. ``standard`` is the manual O(n^2) softmax already in the
# repo (kept as a reference and to record where softmax OOMs); ``sdpa_math`` is the
# fair softmax baseline the verdict compares against. ``linear`` is plain (kernel-
# free) linear attention, the other O(n·d²) point of comparison for piecewise-mean.
# ``piecewise_kmeans_m{k}`` methods are generated dynamically from ``--anchors``;
# the two below are the defaults kept for backward-compatible reruns.
_ALL_METHODS = [
    "piecewise_mean",
    "piecewise_kmeans_m4",
    "piecewise_kmeans_m64",
    "performer",
    "nystromformer",
    "linformer",
    "luna",
    "linear",
    "standard",
    "sdpa_math",
]

# Non-anchor baselines (no ``m`` hyperparameter) — the fixed comparison set the
# anchor sweep is measured against on identical inputs.
_BASELINE_METHODS = [
    "performer", "nystromformer", "linformer", "luna", "linear",
    "standard", "sdpa_math",
]


def _kmeans_anchor_count(method):
    """Anchor count ``m`` for a ``piecewise_kmeans_m{k}`` method name, else None."""
    prefix = "piecewise_kmeans_m"
    if method.startswith(prefix):
        return int(method[len(prefix):])
    return None

# Methods with a working causal implementation. The others raise at construction
# (Nyström/Linformer/Luna) or in the multi-anchor forward (k-means, m>1).
_CAUSAL_OK = {"piecewise_mean", "performer", "linear", "standard", "sdpa_math"}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _sdp_math_ctx():
    """Context manager forcing the pure-math SDPA backend (flash + mem-efficient
    disabled). Uses the torch 2.1 ``torch.backends.cuda.sdp_kernel`` form; the
    ``torch.nn.attention.sdpa_kernel`` API only exists on torch >= 2.2, so this is
    the portable choice for the target runtime. Falls back to a no-op only if the
    2.1 API is entirely absent (in which case SDPA picks its own backend)."""
    sdp = getattr(torch.backends.cuda, "sdp_kernel", None)
    if sdp is None:
        return contextlib.nullcontext()
    return sdp(enable_flash=False, enable_mem_efficient=False, enable_math=True)


class _SdpaMathAttention:
    """Fair softmax baseline: ``scaled_dot_product_attention`` with the fused
    kernels disabled. Matches the ``forward(Q, K, V) -> (out, aux)`` contract of the
    registered attention modules so it drops into the same measurement path."""

    def __init__(self, causal):
        self.causal = causal

    def __call__(self, Q, K, V):
        with _sdp_math_ctx():
            out = F.scaled_dot_product_attention(Q, K, V, is_causal=self.causal)
        return out, None

    def to(self, *args, **kwargs):  # parity with nn.Module.to(); nothing to move
        return self


def _supported(method, causal):
    """Whether ``method`` can run in the requested causal mode."""
    if not causal:
        return True
    return method in _CAUSAL_OK


def _build_method(method, dim, causal, seq_len):
    """Construct the module for ``method`` at per-head ``dim`` in the given mode.

    ``linformer`` is sized to ``seq_len`` (its projection is fixed at construction
    and its forward rejects ``n > max_seq_len``); every other module is independent
    of ``n`` and is reused across the length sweep by the caller.
    """
    common = dict(dim=dim, dropout=0.0, causal=causal)
    if method == "sdpa_math":
        return _SdpaMathAttention(causal)
    if method == "standard":
        return build_attention("standard", **common)
    if method == "linear":
        return build_attention("linear", **common)
    if method == "piecewise_mean":
        return build_attention("piecewise", **common)
    m = _kmeans_anchor_count(method)
    if m is not None:
        return build_attention("piecewise_kmeans", num_anchors=m,
                               kmeans_iters=_KMEANS_ITERS, **common)
    if method == "performer":
        return build_attention("performer", num_features=_PERFORMER_FEATURES, **common)
    if method == "nystromformer":
        return build_attention("nystromformer", num_landmarks=_NYSTROM_LANDMARKS, **common)
    if method == "linformer":
        return build_attention("linformer", k=_LINFORMER_K, max_seq_len=seq_len, **common)
    if method == "luna":
        return build_attention("luna", num_pack=_LUNA_PACK, **common)
    raise ValueError(f"unknown method: {method}")


def _make_qkv(batch, n, dim, dtype, device, seed, requires_grad):
    """Seeded ``(batch, n, dim)`` Q/K/V on ``device`` in ``dtype``. Leaf tensors
    when ``requires_grad`` (for the forward+backward measurement)."""
    gen = torch.Generator().manual_seed(seed + n + dim)
    Q = torch.randn(batch, n, dim, generator=gen)
    K = torch.randn(batch, n, dim, generator=gen)
    V = torch.randn(batch, n, dim, generator=gen)
    out = []
    for t in (Q, K, V):
        t = t.to(device=device, dtype=dtype)
        if requires_grad:
            t = t.requires_grad_(True)
        out.append(t)
    return out


def _time(fn, device, warmup, repeats):
    """Median seconds over ``repeats`` calls after ``warmup``, device-synced.
    Forward-only: runs under ``no_grad``."""
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
                t0 = time.perf_counter()
                fn()
                _sync(device)
                samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _time_fwdbwd(fn, device, warmup, repeats):
    """Median seconds for a forward+backward ``fn`` (no ``no_grad`` wrapper). ``fn``
    is responsible for zeroing grads, running the forward, and calling backward."""
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
            samples.append(start.elapsed_time(end) / 1e3)
        else:
            t0 = time.perf_counter()
            fn()
            _sync(device)
            samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _peak_mem_bytes(fn, device):
    """Peak allocated bytes during a forward ``fn`` (CUDA only; 0 elsewhere)."""
    if device.type != "cuda":
        return 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def _peak_mem_fwdbwd_bytes(fn, device):
    """Peak allocated bytes during a forward+backward ``fn`` (CUDA only)."""
    if device.type != "cuda":
        return 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def _analytic_flops(method, batch, n, dim, causal):
    """Analytic forward FLOPs for one full batch (2 FLOPs per multiply-add). These
    are a theoretical column, not a measurement; the dominant term per method."""
    B, d = batch, dim
    if method in ("standard", "sdpa_math"):
        return 4.0 * B * n * n * d + 3.0 * B * n * n  # scores + softmax + A@V
    if method == "linear":
        return 4.0 * B * n * d * d
    if method == "performer":
        return 4.0 * B * n * _PERFORMER_FEATURES * d
    if method == "nystromformer":
        L = _NYSTROM_LANDMARKS
        return 6.0 * B * n * L * d + 36.0 * B * L * L * L  # kernels + pinv iterations
    if method == "linformer":
        return 8.0 * B * n * _LINFORMER_K * d
    if method == "luna":
        return 8.0 * B * n * _LUNA_PACK * d
    if method == "piecewise_mean":
        return (5.0 if causal else 4.0) * B * n * d * d
    m = _kmeans_anchor_count(method)
    if m is not None:
        return 2.0 * B * m * n * d * d + 2.0 * (_KMEANS_ITERS + 1) * B * n * m * d
    return float("nan")


def _is_oom(exc):
    """True if ``exc`` is a CUDA out-of-memory error (class or message)."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _causal_state_fields(batch, n, dim, dtype):
    """Explicit ``(batch, n, d, d)`` cumsum-state footprint for causal
    piecewise-mean. The instantaneous mathematical state is ``(batch, d, d)`` but
    the implementation materialises the full per-position tensor (and its cumsum),
    which is the dominant long-context memory cost — reported so it can be
    cross-checked against the measured peak."""
    itemsize = torch.empty(0, dtype=dtype).element_size()
    return {
        "state_shape_instantaneous": [batch, dim, dim],
        "state_bytes_instantaneous": batch * dim * dim * itemsize,
        "state_shape_materialized": [batch, n, dim, dim],
        "state_bytes_materialized": batch * n * dim * dim * itemsize,
        "implementation_materializes": True,
    }


def _run_cell(method, dim, dtype_name, causal, n, args, device):
    """Measure one grid cell. Returns a row dict; never raises on OOM."""
    dtype = _DTYPES[dtype_name]
    row = {
        "method": method, "d": dim, "dtype": dtype_name, "causal": causal, "n": n,
        "latency_fwd_ms": None, "latency_fwdbwd_ms": None,
        "peak_mem_fwd_bytes": None, "peak_mem_fwdbwd_bytes": None,
        "analytic_flops": _analytic_flops(method, args.batch, n, dim, causal),
        "status": "ok",
    }
    if method == "piecewise_mean" and causal:
        row.update(_causal_state_fields(args.batch, n, dim, dtype))

    if not _supported(method, causal):
        row["status"] = "skipped"
        return row

    module = None
    try:
        module = _build_method(method, dim, causal, n)
        module = module.to(device=device, dtype=dtype)

        # Forward-only latency + peak memory (fresh no-grad inputs).
        Qf, Kf, Vf = _make_qkv(args.batch, n, dim, dtype, device, args.seed, False)
        fwd = lambda: module(Qf, Kf, Vf)
        row["peak_mem_fwd_bytes"] = _peak_mem_bytes(fwd, device)
        row["latency_fwd_ms"] = _time(fwd, device, args.warmup, args.repeats) * 1e3
        del Qf, Kf, Vf
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Forward+backward latency + peak memory (grad-enabled leaf inputs).
        if args.backward:
            Qb, Kb, Vb = _make_qkv(args.batch, n, dim, dtype, device, args.seed, True)

            def fwdbwd():
                for t in (Qb, Kb, Vb):
                    t.grad = None
                out, _ = module(Qb, Kb, Vb)
                out.sum().backward()

            row["peak_mem_fwdbwd_bytes"] = _peak_mem_fwdbwd_bytes(fwdbwd, device)
            row["latency_fwdbwd_ms"] = _time_fwdbwd(fwdbwd, device, args.warmup,
                                                    args.repeats) * 1e3
            del Qb, Kb, Vb
    except Exception as exc:  # noqa: BLE001 - classify then re-raise non-OOM
        if _is_oom(exc):
            row["status"] = "oom"
            for k in ("latency_fwd_ms", "latency_fwdbwd_ms",
                      "peak_mem_fwd_bytes", "peak_mem_fwdbwd_bytes"):
                row[k] = None
        else:
            raise
    finally:
        del module
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def _lat(rows, d, dtype, causal, n, method):
    """Forward latency for a cell, or ``None`` if missing/OOM/skipped."""
    for r in rows:
        if (r["d"] == d and r["dtype"] == dtype and r["causal"] == causal
                and r["n"] == n and r["method"] == method):
            return r["latency_fwd_ms"] if r["status"] == "ok" else None
    return None


def _status(rows, d, dtype, causal, n, method):
    for r in rows:
        if (r["d"] == d and r["dtype"] == dtype and r["causal"] == causal
                and r["n"] == n and r["method"] == method):
            return r["status"]
    return None


def _compute_verdict(rows, args):
    """Encode issue #18's pass criterion as booleans + a human string, using the
    fair ``sdpa_math`` softmax as the reference. The latency-frontier half of the
    non-domination test is checked here; full accuracy-latency non-domination also
    needs the real-activation fidelity numbers (separate harness)."""
    dtype = "bf16" if "bf16" in args.dtypes else args.dtypes[0]
    ns = sorted(args.lengths)
    linear_baselines = ["linear", "performer", "nystromformer", "linformer", "luna"]

    # (1) piecewise-mean strictly below Performer at d=64, n>=2048 (non-causal).
    below_perf = None
    if 64 in args.dims:
        checks = []
        for n in [x for x in ns if x >= 2048]:
            pm = _lat(rows, 64, dtype, False, n, "piecewise_mean")
            pf = _lat(rows, 64, dtype, False, n, "performer")
            if pm is not None and pf is not None:
                checks.append(pm < pf)
        below_perf = bool(checks) and all(checks)

    # (2) piecewise-mean within 1.5x of the fastest linear baseline at d=128, n>=4096.
    within_15x = None
    nondominated = None
    if 128 in args.dims:
        band_by_n = {}
        for n in [x for x in ns if x >= 4096]:
            pm = _lat(rows, 128, dtype, False, n, "piecewise_mean")
            fastest = [v for v in
                       (_lat(rows, 128, dtype, False, n, b) for b in linear_baselines)
                       if v is not None]
            if pm is not None and fastest:
                band_by_n[n] = pm <= 1.5 * min(fastest)
        within_15x = bool(band_by_n) and all(band_by_n.values())
        # (3) non-dominated for SOME n>=4096 at d=128 (latency-frontier half).
        nondominated = any(band_by_n.values())

    # (4) causal piecewise-mean does not OOM at n=8192, d=64 (checked in bf16).
    causal_ok = None
    if 64 in args.dims and 8192 in args.lengths and "causal" in args.modes:
        causal_ok = _status(rows, 64, "bf16", True, 8192, "piecewise_mean") == "ok"

    conds = {
        "cond_strictly_below_performer_d64": below_perf,
        "cond_within_1_5x_fastest_linear_d128": within_15x,
        "cond_nondominated_d128_some_n_ge_4096": nondominated,
        "cond_causal_no_oom_n8192_d64_bf16": causal_ok,
    }
    present = [v for v in conds.values() if v is not None]
    verdict_pass = bool(present) and all(present)

    parts = [f"{k}={v}" for k, v in conds.items()]
    verdict_string = (
        f"issue #18 gate (softmax ref = sdpa_math, dtype = {dtype}): "
        f"PASS={verdict_pass}. " + "; ".join(parts)
    )
    return conds, verdict_pass, verdict_string, dtype


def _ratio_metrics(rows, args):
    """Reported latency ratios: piecewise-mean vs Performer/Nyström and the
    crossover-n against the fair softmax, per ``d`` (non-causal, bf16 if present)."""
    dtype = "bf16" if "bf16" in args.dtypes else args.dtypes[0]
    metrics = {}
    for d in args.dims:
        for n in sorted(args.lengths):
            pm = _lat(rows, d, dtype, False, n, "piecewise_mean")
            pf = _lat(rows, d, dtype, False, n, "performer")
            ny = _lat(rows, d, dtype, False, n, "nystromformer")
            if pm and pf:
                metrics[f"piecewise_vs_performer_d{d}_n{n}"] = pm / pf
            if pm and ny:
                metrics[f"piecewise_vs_nystrom_d{d}_n{n}"] = pm / ny
        # crossover: smallest n where piecewise-mean beats the fair softmax.
        crossover = None
        for n in sorted(args.lengths):
            pm = _lat(rows, d, dtype, False, n, "piecewise_mean")
            sm = _lat(rows, d, dtype, False, n, "sdpa_math")
            if pm is not None and sm is not None and pm < sm:
                crossover = n
                break
            if pm is not None and _status(rows, d, dtype, False, n, "sdpa_math") == "oom":
                # Softmax OOMs here: piecewise wins at or before this n.
                crossover = n
                break
        metrics[f"crossover_n_vs_sdpa_d{d}"] = crossover
    return metrics


def _relerr_to_reference(approx, exact, eps=1e-9):
    """Mean per-vector relative L2 error over the (batch, n, dim) output, matching
    the real-activation fidelity harness convention (``‖approx−exact‖/‖exact‖``
    per query, averaged). Both tensors are upcast to fp32 first so the metric is
    not itself limited by bf16 rounding."""
    a = approx.detach().float()
    e = exact.detach().float()
    num = torch.linalg.norm(a - e, dim=-1)
    den = torch.linalg.norm(e, dim=-1).clamp_min(eps)
    return float((num / den).mean())


def _largest_reference_n(lengths, batch, dtype, device):
    """The largest ``n`` in ``lengths`` at which a dense ``(batch, n, n)`` softmax
    reference actually fits, probed empirically (a real allocation, freed
    immediately). Returns the ``n`` or ``None`` if even the smallest OOMs. The
    accuracy pass uses this so the ground truth is exact softmax, never an
    approximation of an approximation."""
    if device.type != "cuda":
        return max(lengths)
    fit = None
    for n in sorted(lengths):
        try:
            torch.cuda.empty_cache()
            probe = torch.empty(batch, n, n, dtype=dtype, device=device)
            del probe
            torch.cuda.empty_cache()
            fit = n
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                torch.cuda.empty_cache()
                break
            raise
    return fit


def _accuracy_pass(methods, dim, dtype_name, ref_n, batch, seed, device):
    """Measure output accuracy of every method against exact softmax on ONE shared
    input, at ``(batch, ref_n, dim)``. Non-causal only (the ground truth is dense
    softmax). Every method — anchor configs and the fixed baselines alike — sees
    the SAME seeded Q/K/V, so the rel-err numbers are directly comparable across
    the whole frontier, unlike the separate real-activation harness. Returns a list
    of ``{method, rel_err, status}`` rows. Softmax is the reference, so its own
    rel_err is 0 by construction; if softmax OOMs at this ``ref_n`` the caller
    should not have selected it."""
    dtype = _DTYPES[dtype_name]
    Q, K, V = _make_qkv(batch, ref_n, dim, dtype, device, seed, False)

    # Exact softmax ground truth via the fair math-backend SDPA (fp32 internally
    # upcast in _relerr_to_reference regardless of dtype).
    ref_mod = _SdpaMathAttention(causal=False)
    try:
        with torch.no_grad():
            ref_out, _ = ref_mod(Q, K, V)
        ref_out = ref_out.detach()
    except Exception as exc:  # noqa: BLE001
        del Q, K, V
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if _is_oom(exc):
            return [{"method": "sdpa_math_reference", "rel_err": None,
                     "status": "oom"}]
        raise

    acc_rows = []
    for method in methods:
        if method in ("standard", "sdpa_math"):
            # Softmax methods are the reference; rel_err is 0 by definition.
            acc_rows.append({"method": method, "rel_err": 0.0, "status": "ok"})
            continue
        module = None
        try:
            module = _build_method(method, dim, False, ref_n).to(
                device=device, dtype=dtype)
            with torch.no_grad():
                out = module(Q, K, V)
                out = out[0] if isinstance(out, tuple) else out
            acc_rows.append({"method": method,
                             "rel_err": _relerr_to_reference(out, ref_out),
                             "status": "ok"})
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                acc_rows.append({"method": method, "rel_err": None, "status": "oom"})
            else:
                raise
        finally:
            del module
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del Q, K, V, ref_out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return acc_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--dtypes", type=str, nargs="+", default=["fp32", "bf16"],
                        choices=list(_DTYPES))
    parser.add_argument("--batch", type=int, default=256, help="batch*heads, collapsed")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--modes", type=str, nargs="+", default=["noncausal", "causal"],
                        choices=["noncausal", "causal"])
    parser.add_argument("--methods", type=str, nargs="+", default=list(_ALL_METHODS),
                        help="method names; piecewise_kmeans_m{k} accepted for any k")
    parser.add_argument("--anchors", type=int, nargs="+", default=None,
                        help="anchor counts to sweep as piecewise_kmeans_m{k}; when "
                             "set, these plus the baseline methods form the method "
                             "list (m=1 maps to piecewise_mean).")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--accuracy", action="store_true",
                        help="also measure output rel-err vs exact softmax on one "
                             "shared input at a small batch where softmax fits.")
    parser.add_argument("--acc-batch", type=int, default=8,
                        help="batch size for the accuracy reference pass (small so "
                             "the dense softmax ground truth fits at long n).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    # ``--anchors`` resolves to a method list: one piecewise config per anchor count
    # (m=1 is the mean-anchor fast path, not k-means) plus the fixed baselines, so
    # the anchor dial is measured against the same comparison set on every cell.
    if args.anchors is not None:
        anchor_methods = []
        for m in args.anchors:
            anchor_methods.append("piecewise_mean" if m == 1
                                  else f"piecewise_kmeans_m{m}")
        args.methods = anchor_methods + _BASELINE_METHODS

    if args.smoke:
        args.device = "cpu"
        args.dims = [16]
        args.lengths = [64]
        args.dtypes = ["fp32"]
        args.batch = 2
        args.warmup = 1
        args.repeats = 2
        # Include one causal-unsupported baseline (luna) so the skip path is
        # exercised, one that materialises the causal cumsum state (piecewise_mean),
        # and a dynamic-anchor config (m=8) so the --anchors parsing path is covered.
        args.methods = ["piecewise_mean", "piecewise_kmeans_m8", "performer",
                        "sdpa_math", "luna"]
        args.modes = ["noncausal", "causal"]
        args.backward = True
        args.accuracy = True
        args.acc_batch = 2

    set_seed(args.seed)
    device = get_device(args.device) if args.device == "auto" else torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    print(f"device: {device}  torch: {torch.__version__}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    rows = []
    for dim in args.dims:
        for dtype_name in args.dtypes:
            for mode in args.modes:
                causal = mode == "causal"
                for n in args.lengths:
                    for method in args.methods:
                        row = _run_cell(method, dim, dtype_name, causal, n, args, device)
                        rows.append(row)
                        lat = row["latency_fwd_ms"]
                        lat_s = f"{lat:.3f}ms" if lat is not None else row["status"]
                        print(f"  d={dim} {dtype_name} {mode:9s} n={n:5d} "
                              f"{method:20s} fwd={lat_s}")

    conds, verdict_pass, verdict_string, ref_dtype = _compute_verdict(rows, args)
    ratios = _ratio_metrics(rows, args)
    print("\n" + verdict_string)

    # Accuracy pass: exact-softmax rel-err for every method on ONE shared input, at
    # the largest length where the dense softmax ground truth fits (probed at the
    # small accuracy batch). Speed uses the full batch; accuracy uses --acc-batch so
    # the O(n²) reference can reach long n. Reported per (dim, dtype); non-causal.
    accuracy = {}
    if args.accuracy:
        acc_dtype = "bf16" if "bf16" in args.dtypes else args.dtypes[0]
        for dim in args.dims:
            ref_n = _largest_reference_n(args.lengths, args.acc_batch,
                                         _DTYPES[acc_dtype], device)
            if ref_n is None:
                print(f"\naccuracy(d={dim}): softmax OOMs even at smallest n; "
                      f"accuracy skipped")
                accuracy[str(dim)] = {"reference_n": None, "acc_batch": args.acc_batch,
                                      "dtype": acc_dtype, "rows": []}
                continue
            print(f"\naccuracy(d={dim}, dtype={acc_dtype}, batch={args.acc_batch}, "
                  f"softmax-ground-truth n={ref_n}):")
            acc_rows = _accuracy_pass(args.methods, dim, acc_dtype, ref_n,
                                      args.acc_batch, args.seed, device)
            for ar in acc_rows:
                re = ar["rel_err"]
                re_s = f"{re:.4f}" if re is not None else ar["status"]
                print(f"    {ar['method']:22s} rel_err={re_s}")
            accuracy[str(dim)] = {"reference_n": ref_n, "acc_batch": args.acc_batch,
                                  "dtype": acc_dtype, "rows": acc_rows}

    metrics = {"verdict_pass": bool(verdict_pass)}
    for k, v in conds.items():
        metrics[k] = (bool(v) if v is not None else False)
    metrics.update({k: v for k, v in ratios.items() if v is not None})

    result = ExperimentResult(
        config={
            "benchmark": "cuda_scaling", "dims": args.dims, "lengths": args.lengths,
            "dtypes": args.dtypes, "batch": args.batch, "modes": args.modes,
            "methods": args.methods, "warmup": args.warmup, "repeats": args.repeats,
            "backward": args.backward, "seed": args.seed,
            "anchors": args.anchors, "accuracy": args.accuracy,
            "acc_batch": args.acc_batch,
            "performer_features": _PERFORMER_FEATURES,
            "nystrom_landmarks": _NYSTROM_LANDMARKS, "linformer_k": _LINFORMER_K,
            "luna_pack": _LUNA_PACK,
            "kmeans_anchors": (args.anchors if args.anchors is not None else [4, 64]),
            "verdict_ref_dtype": ref_dtype,
        },
        metrics=metrics,
        environment={
            "device": str(device), "torch": torch.__version__,
            "cuda": torch.version.cuda, "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "sdpa_backend": "math (flash+mem_efficient disabled)",
        },
        history={"rows": rows, "verdict_string": verdict_string,
                 "accuracy": accuracy},
    )

    if args.out:
        print(f"wrote {save_result(result, args.out)}")

    if args.smoke:
        assert rows, "no measurements produced"
        assert any(r["status"] == "ok" for r in rows), "no cell ran to completion"
        assert any(r["status"] == "skipped" for r in rows), \
            "expected causal-unsupported cells to be skipped"
        assert any(_kmeans_anchor_count(r["method"]) is not None for r in rows), \
            "expected a dynamic piecewise_kmeans_m{k} anchor method in the rows"
        # Accuracy path produced comparable rel-err for the anchor config and a
        # baseline against the softmax reference.
        acc_all = [ar for blk in accuracy.values() for ar in blk["rows"]]
        assert acc_all, "accuracy pass produced no rows"
        assert any(ar["method"].startswith("piecewise") and ar["rel_err"] is not None
                   for ar in acc_all), "no piecewise accuracy measured"
        print("smoke OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
