#!/usr/bin/env python
"""Reproducible associative-recall benchmark.

Trains a small bidirectional encoder with each attention mechanism at a matched
budget on the synthetic associative-recall task and reports test accuracy. This
regenerates the core evidence: softmax and (multi-)anchor piecewise attention
solve recall, while linear and Performer attention do not; and single-anchor
piecewise collapses at longer sequence lengths where multi-anchor recovers.

Examples
--------
Smoke test (seconds, for CI)::

    python benchmarks/harness/run_recall.py --smoke

Full reproduction::

    python benchmarks/harness/run_recall.py --seeds 0 1 2 --lengths 32 64 \
        --out results/recall.json
"""

import argparse
import json
import platform
import sys
from pathlib import Path

# Allow running directly from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from piecewise_linear_attention.harness import (
    ExperimentConfig,
    get_device,
    train_and_evaluate,
)
from piecewise_linear_attention.harness.results import ExperimentResult, save_result
from piecewise_linear_attention.harness.tasks.synthetic_recall import (
    RecallModel,
    make_recall_batch,
)

# Methods under comparison. Multi-anchor presets carry their anchor count.
METHODS = [
    ("standard", {}),
    ("linear", {"kernel_type": "elu"}),
    ("performer", {}),
    ("piecewise", {}),  # single-anchor (mean)
    ("piecewise_kmeans", {"num_anchors": 4}),
    ("piecewise_stride", {"num_anchors": 4}),
]


def _run_one(config: ExperimentConfig, device, eval_every: int = 0) -> dict:
    model = RecallModel(
        attention_type=config.attention_type,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        attention_kwargs=config.attention_kwargs,
    )

    def batch_fn(generator):
        return make_recall_batch(config.batch_size, config.seq_len, config.vocab_size, generator)

    metrics = train_and_evaluate(
        model,
        batch_fn,
        steps=config.steps,
        lr=config.lr,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        seed=config.seed,
        device=device,
        eval_batches=config.eval_batches,
        eval_every=eval_every,
    )
    params = sum(p.numel() for p in model.parameters())
    return {**metrics, "params": params}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-anchors", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluate every N steps and record the accuracy-vs-step curve "
        "(surfaces late/grokking transitions). 0 disables.",
    )
    parser.add_argument("--out", type=str, default=None, help="Path to write JSON results")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny, fast configuration for CI (asserts the pipeline runs)",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        args.seeds = [0]
        args.lengths = [16]
        args.hidden_dim = 16
        args.num_layers = 1
        args.num_heads = 2
        args.batch_size = 8
        args.steps = 2
        args.vocab_size = 12
        args.num_anchors = 2

    device = get_device(args.device)
    print(f"device: {device}")

    rows = []
    for seq_len in args.lengths:
        for attention_type, extra in METHODS:
            kwargs = dict(extra)
            if "piecewise_" in attention_type:
                kwargs["num_anchors"] = args.num_anchors
            for seed in args.seeds:
                config = ExperimentConfig(
                    attention_type=attention_type,
                    task="synthetic_recall",
                    hidden_dim=args.hidden_dim,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers,
                    seq_len=seq_len,
                    vocab_size=args.vocab_size,
                    batch_size=args.batch_size,
                    steps=args.steps,
                    lr=args.lr,
                    seed=seed,
                    device=args.device,
                    attention_kwargs=kwargs,
                    name=f"{attention_type}-n{seq_len}-s{seed}",
                )
                metrics = _run_one(config, device, eval_every=args.eval_every)
                rows.append({"config": config.to_dict(), "metrics": metrics})
                print(
                    f"  n={seq_len:4d} {attention_type:18s} seed={seed} "
                    f"acc={metrics['accuracy']:.4f} loss={metrics['loss']:.3f} "
                    f"time={metrics['train_time_s']:.1f}s"
                )

    result = ExperimentResult(
        config={"benchmark": "recall", "seeds": args.seeds, "lengths": args.lengths},
        metrics={"random_baseline": 1.0 / args.vocab_size},
        environment={
            "device": str(device),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        history={"runs": rows},
    )

    if args.out:
        path = save_result(result, args.out)
        print(f"wrote {path}")

    if args.smoke:
        # CI assertion: the pipeline produced a well-formed result for every run.
        assert rows, "no runs produced"
        for row in rows:
            assert "accuracy" in row["metrics"]
            assert 0.0 <= row["metrics"]["accuracy"] <= 1.0
        print("smoke OK")

    return result


if __name__ == "__main__":
    main(sys.argv[1:])
