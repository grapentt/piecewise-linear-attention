"""Train transformer with different attention types on translation task.

This script compares StandardAttention, LinearAttention, and PiecewiseAttention
in a full encoder-decoder transformer on machine translation.

Usage:
    python experiments/translation/train.py
    python experiments/translation/train.py --dataset opus_books --src en --tgt fr
    python experiments/translation/train.py --epochs 10 --save results/translation/results.json
"""

import argparse
import json
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.data import prepare_dataset, collate_fn
from experiments.utils.training import train_epoch, evaluate
from piecewise_linear_attention.models.translation_transformer import ConfigurableTransformer


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility.

    This ensures all attention variants get identical:
    - Model initializations
    - Data shuffling order
    - Data augmentation samples

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic (may impact performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_transformer_benchmark(
    attention_type: str,
    dataset_name: str,
    subset_name: str,
    src_lang: str,
    tgt_lang: str,
    max_train_samples: int,
    max_eval_samples: int,
    max_seq_len: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> Dict:
    """Run benchmark for one attention type.

    Args:
        attention_type: "standard", "linear", or "piecewise"
        dataset_name: HuggingFace dataset name
        subset_name: Dataset subset
        src_lang: Source language code
        tgt_lang: Target language code
        max_train_samples: Max training samples
        max_eval_samples: Max eval samples
        max_seq_len: Max sequence length
        hidden_dim: Model hidden dimension
        num_layers: Number of encoder/decoder layers
        num_heads: Number of attention heads
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use

    Returns:
        Dictionary with training results
    """
    print("=" * 80)
    print(f"Benchmarking {attention_type.upper()} Attention")
    print("=" * 80)

    # Prepare dataset
    train_dataset, eval_dataset, src_dict, tgt_dict = prepare_dataset(
        dataset_name=dataset_name,
        subset_name=subset_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        max_seq_len=max_seq_len,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, src_dict=src_dict, tgt_dict=tgt_dict, device=device),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, src_dict=src_dict, tgt_dict=tgt_dict, device=device),
    )

    # Create model
    print(f"\nCreating {attention_type} transformer...")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {num_layers} encoder + {num_layers} decoder")
    print(f"  Heads: {num_heads}")
    print(f"  Vocab: {len(src_dict)} src, {len(tgt_dict)} tgt")

    model = ConfigurableTransformer(
        source_dictionary=src_dict,
        target_dictionary=tgt_dict,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=num_heads,
        attention_type=attention_type,
        device=device,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initial evaluation
    print("\nInitial evaluation...")
    initial_metrics = evaluate(model, eval_loader, device=device)
    print(f"  Loss: {initial_metrics['loss']:.4f}")
    print(f"  Perplexity: {initial_metrics['perplexity']:.2f}")

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    train_history = []
    eval_history = [initial_metrics]

    total_train_time = 0.0

    for epoch in range(epochs):
        print(f"\n{'─' * 80}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'─' * 80}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device=device)
        total_train_time += train_metrics["time_seconds"]

        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Perplexity: {train_metrics['perplexity']:.2f}, "
              f"Time: {train_metrics['time_seconds']:.1f}s, "
              f"Speed: {train_metrics['samples_per_second']:.1f} samples/s")

        # Evaluate
        eval_metrics = evaluate(model, eval_loader, device=device)
        print(f"Eval  - Loss: {eval_metrics['loss']:.4f}, "
              f"Perplexity: {eval_metrics['perplexity']:.2f}")

        train_history.append(train_metrics)
        eval_history.append(eval_metrics)

    # Final summary
    final_train = train_history[-1]
    final_eval = eval_history[-1]

    print("\n" + "=" * 80)
    print(f"FINAL RESULTS - {attention_type.upper()}")
    print("=" * 80)
    print(f"Training:")
    print(f"  Loss: {initial_metrics['loss']:.4f} → {final_train['loss']:.4f}")
    print(f"  Perplexity: {initial_metrics['perplexity']:.2f} → {final_train['perplexity']:.2f}")
    print(f"  Total time: {total_train_time:.1f}s")
    print(f"  Avg speed: {sum(m['samples_per_second'] for m in train_history) / len(train_history):.1f} samples/s")
    print(f"\nEvaluation:")
    print(f"  Loss: {initial_metrics['loss']:.4f} → {final_eval['loss']:.4f}")
    print(f"  Perplexity: {initial_metrics['perplexity']:.2f} → {final_eval['perplexity']:.2f}")
    print("=" * 80)

    return {
        "attention_type": attention_type,
        "dataset": f"{dataset_name}/{subset_name}",
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "model_config": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_params": num_params,
        },
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_train_samples": max_train_samples,
            "max_eval_samples": max_eval_samples,
        },
        "initial_metrics": initial_metrics,
        "final_train_metrics": {
            "loss": final_train["loss"],
            "perplexity": final_train["perplexity"],
        },
        "final_eval_metrics": {
            "loss": final_eval["loss"],
            "perplexity": final_eval["perplexity"],
        },
        "total_train_time": total_train_time,
        "avg_samples_per_second": sum(m["samples_per_second"] for m in train_history) / len(train_history),
        "train_history": train_history,
        "eval_history": eval_history,
    }


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark transformer with different attention types on translation"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="bentrevett/multi30k",
        help="HuggingFace dataset name (default: bentrevett/multi30k)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="de-en",
        help="Dataset subset (default: de-en)",
    )
    parser.add_argument("--src", type=str, default="de", help="Source language (default: de)")
    parser.add_argument("--tgt", type=str, default="en", help="Target language (default: en)")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=10000,
        help="Max training samples (default: 10000)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=1000,
        help="Max eval samples (default: 1000)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Max sequence length (default: 128)",
    )

    # Model arguments
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension (default: 256)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of encoder/decoder layers (default: 3)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )

    # Comparison arguments
    parser.add_argument(
        "--attention-types",
        type=str,
        nargs="+",
        default=["standard", "linear", "piecewise"],
        choices=["standard", "linear", "piecewise"],
        help="Attention types to benchmark (default: all three)",
    )

    # Output arguments
    parser.add_argument(
        "--save",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TRANSFORMER TRANSLATION BENCHMARK")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}/{args.subset} ({args.src}→{args.tgt})")
    print(f"Train samples: {args.max_train_samples}, Eval samples: {args.max_eval_samples}")
    print(f"Model: {args.hidden_dim}d, {args.num_layers}L, {args.num_heads}H")
    print(f"Training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print(f"Comparing: {', '.join(args.attention_types)}")
    print(f"Device: {args.device}")
    print(f"Random Seed: {args.seed}")
    print()

    # Run benchmarks for each attention type
    results = []

    for attention_type in args.attention_types:
        # Set the same seed before each experiment for fair comparison
        print(f"\n🎲 Setting random seed to {args.seed} for {attention_type} attention...")
        set_seed(args.seed)

        try:
            result = run_transformer_benchmark(
                attention_type=attention_type,
                dataset_name=args.dataset,
                subset_name=args.subset,
                src_lang=args.src,
                tgt_lang=args.tgt,
                max_train_samples=args.max_train_samples,
                max_eval_samples=args.max_eval_samples,
                max_seq_len=args.max_seq_len,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=args.device,
            )
            results.append(result)

        except Exception as e:
            print(f"\n❌ Error benchmarking {attention_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Comparison summary
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print()

        # Table header
        print(f"{'Attention Type':<20} {'Train Loss':<12} {'Eval Loss':<12} {'Eval Perplexity':<18} {'Train Time (s)':<15} {'Speed (samples/s)':<20}")
        print("-" * 120)

        # Table rows
        for result in results:
            attn = result["attention_type"].capitalize()
            train_loss = result["final_train_metrics"]["loss"]
            eval_loss = result["final_eval_metrics"]["loss"]
            eval_ppl = result["final_eval_metrics"]["perplexity"]
            train_time = result["total_train_time"]
            speed = result["avg_samples_per_second"]

            print(f"{attn:<20} {train_loss:<12.4f} {eval_loss:<12.4f} {eval_ppl:<18.2f} {train_time:<15.1f} {speed:<20.1f}")

        print("-" * 120)
        print()

        # Relative comparison (baseline = standard)
        if any(r["attention_type"] == "standard" for r in results):
            standard = next(r for r in results if r["attention_type"] == "standard")
            baseline_time = standard["total_train_time"]
            baseline_speed = standard["avg_samples_per_second"]

            print("Speedup vs Standard Attention:")
            for result in results:
                if result["attention_type"] != "standard":
                    speedup = baseline_time / result["total_train_time"]
                    throughput_gain = result["avg_samples_per_second"] / baseline_speed
                    print(f"  {result['attention_type'].capitalize()}: {speedup:.2f}× faster, {throughput_gain:.2f}× throughput")
            print()

    # Save results
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {save_path}")

    print("=" * 80)
    print("✅ Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
