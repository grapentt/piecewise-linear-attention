#!/usr/bin/env python
"""Diagnose the first-few-epochs slowdown of piecewise attention on LRA ListOps.

A ListOps run with four k-means anchors was observed to take ~43 min in epoch 1
and settle to ~13 min by epoch 4. This harness answers three questions the
end-to-end epoch clock cannot:

1. **What is slow in the early epochs, and where does the time go?**  It splits
   each epoch into the data pipeline, the forward's *anchor clustering* and
   *second-moment einsum* (timed natively by the model when a probe is installed —
   see :mod:`piecewise_linear_attention.core.profiling`), and the remaining
   compute (attention apply + FFN + backward + optimizer).

2. **Why does it speed up epoch-over-epoch?**  Four non-exclusive causes are
   separated by their distinct signatures:

     - **Dataset first-touch / page-cache** — shows up as an epoch-1 spike in the
       *data* time that vanishes once the OS has cached the file. Read from the
       trainer's own data-vs-compute split.
     - **CUDA allocator growth** — the caching allocator grows its pool with
       ``cudaMalloc`` calls in the first epoch, then reuses it. Read from
       ``torch.cuda.memory_stats`` (reserved bytes plateau, ``num_alloc_retries``).
     - **cuBLAS / kernel lazy first-touch** — the first handful of *iterations*
       pay a one-time kernel-load / workspace-allocation cost. Invisible at epoch
       granularity, so the harness runs a per-iteration "microscope" over the
       first N steps of epoch 1.
     - **k-means "convergence"** — the intuitive guess. Refuted or confirmed by
       the ``anchor_clustering`` phase total: :class:`KMeansAnchor` is stateless
       per forward (fixed Lloyd iterations, deterministic per-batch init, no
       cross-step memory), and ``cudnn.benchmark`` is off in the trainer, so this
       phase is expected to be **flat** across epochs. A flat curve here is the
       evidence that the speedup is system warmup, not clustering getting cheaper.

3. **How does anchor caching change the picture?**  The same breakdown is run for
   the mean anchor (``m=1``), fresh k-means (``m=4``) and cached k-means
   (``m=4`` with ``anchor_refresh_every`` > 1), so the clustering the cache
   amortizes is visible as a drop in the ``anchor_clustering`` phase, and its
   orthogonality to the (untouched) einsum floor is explicit.

The harness **reuses the real trainer's** ``train_epoch`` / ``evaluate`` (it does
not reimplement the loop), so the numbers are the training path's own, and it only
adds measurement around them. On CUDA all stamps are synchronized.

Making the approach competitive
--------------------------------
The breakdown is the input to that question, not the answer. If the steady-state
forward is einsum-bound, caching cannot help and the lever is the low-rank
Jacobian build (project the second moment onto the few query-residual directions
each cluster excites); if it is clustering-bound, caching / fewer Lloyd iterations
recover it directly. The per-phase steady-state fractions this emits say which
regime the ListOps shape is in.

Examples
--------
Smoke (CPU, seconds, tiny shapes)::

    python benchmarks/harness/run_epoch_warmup_diag.py --smoke

Full diagnostic on an A100 (real ListOps, the run this explains)::

    python benchmarks/harness/run_epoch_warmup_diag.py \
        --device cuda --epochs 4 --batch-size 32 --max-length 2000 \
        --methods piecewise piecewise_kmeans_m4 piecewise_kmeans_m4_cached8 \
        --micro-iters 60 --precision bf16 \
        --out results/lra/listops/epoch_warmup_diag.json
"""

import argparse
import importlib
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Project root: benchmarks/harness/<this> -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from piecewise_linear_attention.core.profiling import PhaseProbe, profile
from piecewise_linear_attention.harness import set_seed
from piecewise_linear_attention.harness.results import ExperimentResult, save_result


# Method-name convention, matching run_cuda_scaling.py:
#   "piecewise"                    -> mean anchor, m=1
#   "piecewise_kmeans_m{k}"        -> k-means, m=k, fresh clustering every forward
#   "piecewise_kmeans_m{k}_cached{N}" -> k-means, m=k, centroids reused, refresh every N
def _parse_method(method: str) -> Dict:
    """Resolve a method name to (attention_type, attn_kwargs) for the encoder."""
    if method == "piecewise":
        return {"attention_type": "piecewise", "attn_kwargs": {}}
    prefix = "piecewise_kmeans_m"
    if not method.startswith(prefix):
        raise ValueError(
            f"unknown method '{method}'; expected 'piecewise' or "
            f"'piecewise_kmeans_m{{k}}[_cached{{N}}]'"
        )
    rest = method[len(prefix):]
    refresh = 1
    if "_cached" in rest:
        rest, _, cached = rest.partition("_cached")
        refresh = int(cached)
    m = int(rest)
    return {
        "attention_type": "piecewise_kmeans",
        "attn_kwargs": {"num_anchors": m, "anchor_refresh_every": refresh},
    }


def _lra_modules():
    """Import the LRA ListOps trainer modules, or return None with a reason.

    The trainer lives under ``experiments/lra`` (present on the LRA branch and any
    branch it has merged into). Returning None instead of raising lets this file be
    imported for its ``--smoke`` self-test even where the trainer is absent.
    """
    try:
        cfg = importlib.import_module("experiments.lra.configs.base_config")
        enc = importlib.import_module("experiments.lra.models.encoder")
        data = importlib.import_module("experiments.lra.data.listops")
        train = importlib.import_module("experiments.lra.utils.training")
        return {
            "ListOpsConfig": cfg.ListOpsConfig,
            "NUM_LANDMARKS": getattr(cfg, "NUM_LANDMARKS", 32),
            "attention_hparams": cfg.attention_hparams,
            "LRAEncoder": enc.LRAEncoder,
            "get_listops_dataloaders": data.get_listops_dataloaders,
            "ListOpsDataset": data.ListOpsDataset,
            "train_epoch": train.train_epoch,
            "evaluate": train.evaluate,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"error": f"{type(exc).__name__}: {exc}"}


def _mem_snapshot(device: torch.device) -> Optional[Dict]:
    """CUDA allocator state, or None off CUDA. Captures the warmup-relevant
    counters: reserved pool size (grows in epoch 1 then plateaus) and the number of
    allocator retries / cudaMalloc calls (the actual OS allocations)."""
    if device.type != "cuda":
        return None
    s = torch.cuda.memory_stats(device)
    return {
        "reserved_bytes": s.get("reserved_bytes.all.current", 0),
        "reserved_peak_bytes": s.get("reserved_bytes.all.peak", 0),
        "allocated_bytes": s.get("allocated_bytes.all.current", 0),
        "num_alloc_retries": s.get("num_alloc_retries", 0),
        "num_device_alloc": s.get("num_device_alloc", 0),
        "num_device_free": s.get("num_device_free", 0),
    }


def _build_model(method: str, mods: Dict, config, vocab_size, device):
    """Construct an LRAEncoder for ``method`` at the trainer's ListOps config."""
    spec = _parse_method(method)
    base_kwargs = mods["attention_hparams"](
        spec["attention_type"], config.max_length, config.pooling_mode
    )
    # attention_hparams returns the matched-budget baseline knobs; overlay the
    # anchor knobs (num_anchors / anchor_refresh_every) this diagnostic sweeps.
    attn_kwargs = {**(base_kwargs or {}), **spec["attn_kwargs"]}
    model = mods["LRAEncoder"](
        vocab_size=vocab_size,
        max_seq_len=config.max_length,
        num_classes=config.num_classes,
        emb_dim=config.emb_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_dim=config.mlp_dim,
        attention_type=spec["attention_type"],
        dropout=config.dropout,
        pooling_mode=config.pooling_mode,
        attn_kwargs=attn_kwargs,
        pad_to_multiple_of=mods["NUM_LANDMARKS"],
        device=device,
    ).to(device)
    return model, spec, attn_kwargs


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _synthetic_loaders(vocab_size, seq_len, num_classes, batch_size, device, seed):
    """Tiny in-memory train/val/test loaders matching the ListOps batch schema.

    Used only by ``--smoke`` so the harness self-test validates the measurement
    machinery (phase split, microscope, per-epoch instrumentation) in seconds,
    without loading the ~96k-row real ListOps dataset. The batch dict keys and
    dtypes mirror :class:`ListOpsDataset` exactly (``input_ids`` long,
    ``padding_mask`` bool, ``labels`` long), so the same ``train_epoch`` /
    ``evaluate`` path exercised in a full run is exercised here.
    """
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(seed)
    n = batch_size * 6  # a handful of batches, enough for the microscope
    # Token 0 is padding; draw valid tokens in [1, vocab_size).
    input_ids = torch.randint(1, vocab_size, (n, seq_len), generator=g)
    # A short valid prefix per sample, padded tail (matches encoder suffix-padding).
    valid_len = torch.randint(seq_len // 2, seq_len + 1, (n,), generator=g)
    padding_mask = torch.arange(seq_len).unsqueeze(0) < valid_len.unsqueeze(1)
    input_ids = input_ids * padding_mask.long()  # zero out padded positions
    labels = torch.randint(0, num_classes, (n,), generator=g)

    def _collate(rows):
        ids, mask, lab = zip(*rows)
        return {
            "input_ids": torch.stack(ids),
            "padding_mask": torch.stack(mask),
            "labels": torch.stack(lab),
        }

    ds = TensorDataset(input_ids, padding_mask, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    return loader, loader, loader


class _SmokeConfig:
    """Minimal stand-in for ListOpsConfig used by the ``--smoke`` self-test."""

    def __init__(self, max_length, batch_size, seed):
        self.max_length = max_length
        self.batch_size = batch_size
        self.random_seed = seed
        self.num_classes = 10
        self.emb_dim = 32
        self.num_heads = 4
        self.num_layers = 2
        self.mlp_dim = 64
        self.dropout = 0.0
        self.pooling_mode = "CLS"
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.grad_clip = 1.0


def _microscope(model, loader, optimizer, criterion, device, amp_dtype, n_iters):
    """Per-iteration timing over the first ``n_iters`` steps of a fresh epoch.

    Isolates the cuBLAS / kernel lazy-first-touch signature (H3): the first few
    iterations carry a one-time kernel-load and workspace-allocation cost that is
    smeared away at epoch granularity. Each step is synchronized so the per-step
    stamp is truthful, and the forward runs under a phase probe so the clustering
    and einsum split is available per step too.
    """
    from contextlib import nullcontext

    def _autocast():
        if amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    model.train()
    per_iter = []
    it = iter(loader)
    for i in range(n_iters):
        try:
            batch = next(it)
        except StopIteration:
            break
        input_ids = batch["input_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = batch["labels"].to(device)
        _sync(device)

        probe = PhaseProbe(device)
        t0 = time.perf_counter()
        optimizer.zero_grad()
        with profile(probe), _autocast():
            logits = model(input_ids, padding_mask)
            loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _sync(device)
        step_ms = (time.perf_counter() - t0) * 1e3
        probe.finalize()
        phases = probe.totals_ms()
        per_iter.append({
            "iter": i,
            "step_ms": round(step_ms, 3),
            "clustering_ms": round(phases.get("anchor_clustering", 0.0), 3),
            "einsum_ms": round(phases.get("second_moment_einsum", 0.0), 3),
            "mem": _mem_snapshot(device),
        })
    return per_iter


def _run_method(method: str, mods: Dict, config, loaders, device, args) -> Dict:
    """Full per-epoch diagnostic for one method: microscope + instrumented epochs."""
    train_loader, val_loader, _ = loaders
    vocab_size = mods["ListOpsDataset"].get_vocab_size()

    set_seed(config.random_seed)
    model, spec, attn_kwargs = _build_model(method, mods, config, vocab_size, device)
    num_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Epoch-1 microscope BEFORE the first full epoch, on a fresh model/optimizer, so
    # the very first optimizer steps (with their cuBLAS first-touch) are the ones
    # measured per-iteration. Its steps are real training steps — the model is not
    # reset afterward — so the subsequent full epochs continue from here.
    mem_before = _mem_snapshot(device)
    micro = _microscope(
        model, train_loader, optimizer, criterion, device, amp_dtype, args.micro_iters
    )
    mem_after_micro = _mem_snapshot(device)

    epochs = []
    for epoch in range(1, args.epochs + 1):
        probe = PhaseProbe(device)
        mem_pre = _mem_snapshot(device)
        with profile(probe):
            metrics = mods["train_epoch"](
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                grad_clip=config.grad_clip,
                amp_dtype=amp_dtype,
            )
        probe.finalize()
        mem_post = _mem_snapshot(device)
        phases = probe.totals_ms()
        clustering_ms = phases.get("anchor_clustering", 0.0)
        einsum_ms = phases.get("second_moment_einsum", 0.0)
        compute_s = metrics["compute_time"]
        data_s = metrics["data_time"]
        # "Other compute" = everything in the synced compute window that is not the
        # two instrumented phases: attention apply + FFN + backward + optimizer.
        other_compute_s = max(compute_s - (clustering_ms + einsum_ms) / 1e3, 0.0)
        epochs.append({
            "epoch": epoch,
            "wall_s": round(metrics["time"], 3),
            "data_s": round(data_s, 3),
            "compute_s": round(compute_s, 3),
            "clustering_s": round(clustering_ms / 1e3, 4),
            "einsum_s": round(einsum_ms / 1e3, 4),
            "other_compute_s": round(other_compute_s, 3),
            "train_acc": round(metrics["accuracy"], 4),
            "train_loss": round(metrics["loss"], 4),
            "mem_pre": mem_pre,
            "mem_post": mem_post,
        })

    return {
        "method": method,
        "attention_type": spec["attention_type"],
        "attn_kwargs": attn_kwargs,
        "num_parameters": num_params,
        "mem_before": mem_before,
        "mem_after_micro": mem_after_micro,
        "microscope": micro,
        "epochs": epochs,
    }


def _summarize(method_result: Dict) -> Dict:
    """Distill the warmup signatures from one method's per-epoch trace.

    Reports the epoch-1 vs steady-state ratio for each phase so the dominant cause
    is legible without reading the full trace: a large data ratio implicates
    page-cache; a flat clustering ratio refutes k-means convergence; a large
    first-iteration step ratio in the microscope implicates cuBLAS first-touch.
    """
    ep = method_result["epochs"]
    if not ep:
        return {"note": "no epochs run"}
    first, last = ep[0], ep[-1]

    def _ratio(key):
        lo = last[key]
        return round(first[key] / lo, 2) if lo > 0 else None

    micro = method_result["microscope"]
    micro_ratio = None
    if len(micro) >= 4:
        head = micro[0]["step_ms"]
        tail = sum(m["step_ms"] for m in micro[-3:]) / 3.0
        micro_ratio = round(head / tail, 2) if tail > 0 else None

    return {
        "epoch1_wall_s": first["wall_s"],
        "steady_wall_s": last["wall_s"],
        "wall_ratio_e1_over_steady": _ratio("wall_s"),
        "data_ratio_e1_over_steady": _ratio("data_s"),
        "compute_ratio_e1_over_steady": _ratio("compute_s"),
        "clustering_ratio_e1_over_steady": _ratio("clustering_s"),
        "einsum_ratio_e1_over_steady": _ratio("einsum_s"),
        "steady_clustering_frac_of_compute": (
            round(last["clustering_s"] / last["compute_s"], 3)
            if last["compute_s"] > 0 else None
        ),
        "steady_einsum_frac_of_compute": (
            round(last["einsum_s"] / last["compute_s"], 3)
            if last["compute_s"] > 0 else None
        ),
        "microscope_firstiter_over_tail": micro_ratio,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="ListOps sequence length (default: config value of 2000).",
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["piecewise", "piecewise_kmeans_m4", "piecewise_kmeans_m4_cached8"],
        help="method names; see the module docstring for the naming convention.",
    )
    parser.add_argument(
        "--micro-iters", type=int, default=60,
        help="per-iteration microscope length over the first epoch's steps.",
    )
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--data-dir", default="data/lra/listops")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--smoke", action="store_true",
        help="tiny CPU self-test: forces cpu, 2 epochs, batch 4, max-length 64, "
        "8 microscope iters, and asserts the phase split is populated.",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        args.device = "cpu"
        args.epochs = 2
        args.batch_size = 4
        args.max_length = 64
        args.micro_iters = 8
        args.precision = "fp32"
        args.methods = ["piecewise", "piecewise_kmeans_m4", "piecewise_kmeans_m4_cached4"]

    device = torch.device(
        "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available())
        else "cpu"
    )

    mods = _lra_modules()
    if "error" in mods:
        msg = (
            "LRA ListOps trainer not importable in this checkout "
            f"({mods['error']}). This diagnostic runs where experiments/lra is "
            "present (the LRA branch / after it merges). Nothing measured."
        )
        if args.smoke:
            # In smoke mode the trainer must be present to exercise the model/loop;
            # make the absence an explicit skip rather than a silent pass.
            print(f"SKIP: {msg}")
            return None
        raise SystemExit(msg)

    vocab_size = mods["ListOpsDataset"].get_vocab_size()

    if args.smoke:
        # Synthetic tiny loaders + config: validate the harness machinery in
        # seconds without the full ListOps dataset load. The real model and the
        # real train_epoch/evaluate path are still exercised.
        config = _SmokeConfig(args.max_length, args.batch_size, args.seed)
        loaders = _synthetic_loaders(
            vocab_size, config.max_length, config.num_classes,
            config.batch_size, device, args.seed,
        )
    else:
        config = mods["ListOpsConfig"]()
        config.batch_size = args.batch_size
        config.random_seed = args.seed
        if args.max_length is not None:
            config.max_length = args.max_length
        loaders = mods["get_listops_dataloaders"](
            data_dir=args.data_dir,
            batch_size=config.batch_size,
            max_length=config.max_length,
            download=True,
            num_workers=args.num_workers,
        )

    print(f"device={device} epochs={args.epochs} batch={config.batch_size} "
          f"max_length={config.max_length} precision={args.precision}")
    print(f"methods={args.methods}")

    results = []
    summaries = {}
    for method in args.methods:
        print(f"\n=== {method} ===")
        r = _run_method(method, mods, config, loaders, device, args)
        results.append(r)
        summaries[method] = _summarize(r)
        s = summaries[method]
        print(f"  epoch1 wall={s['epoch1_wall_s']}s  steady wall={s['steady_wall_s']}s  "
              f"(e1/steady={s['wall_ratio_e1_over_steady']}x)")
        print(f"  data e1/steady={s['data_ratio_e1_over_steady']}x  "
              f"compute e1/steady={s['compute_ratio_e1_over_steady']}x  "
              f"clustering e1/steady={s['clustering_ratio_e1_over_steady']}x  "
              f"einsum e1/steady={s['einsum_ratio_e1_over_steady']}x")
        print(f"  steady compute split: clustering={s['steady_clustering_frac_of_compute']}  "
              f"einsum={s['steady_einsum_frac_of_compute']}")
        print(f"  microscope first-iter/tail={s['microscope_firstiter_over_tail']}x")

    result = ExperimentResult(
        config={
            "benchmark": "epoch_warmup_diag",
            "purpose": (
                "isolate the first-few-epochs slowdown of piecewise attention on "
                "LRA ListOps into data page-cache, CUDA allocator growth, cuBLAS "
                "first-touch, and (refuted) k-means convergence; compare fresh vs "
                "cached anchors epoch-by-epoch"
            ),
            "methods": args.methods,
            "epochs": args.epochs,
            "batch_size": config.batch_size,
            "max_length": config.max_length,
            "precision": args.precision,
            "micro_iters": args.micro_iters,
            "seed": args.seed,
        },
        metrics={"summaries": summaries},
        environment={
            "device": str(device),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "num_threads": torch.get_num_threads(),
        },
        history={"per_method": results},
    )

    if args.out:
        path = save_result(result, args.out)
        print(f"\nwrote {path}")

    if args.smoke:
        assert results, "no methods measured"
        for r in results:
            assert r["epochs"], f"no epochs for {r['method']}"
            e = r["epochs"][-1]
            # The phase split must be populated for the k-means methods (the mean
            # anchor has no clustering/einsum phase, so only assert for m>1).
            if r["attn_kwargs"].get("num_anchors", 1) > 1:
                assert e["clustering_s"] > 0.0, f"clustering not timed for {r['method']}"
                assert e["einsum_s"] > 0.0, f"einsum not timed for {r['method']}"
            assert r["microscope"], f"microscope empty for {r['method']}"
        print("\nsmoke OK: per-epoch phase split and microscope populated")

    return result


if __name__ == "__main__":
    main(sys.argv[1:])
