"""Training script for ListOps task.

Usage:
    # Quick test (5 epochs, single attention type)
    python experiments/lra/tasks/listops/train.py --attention-types piecewise --epochs 5 --device mps

    # Full benchmark (all attention types)
    python experiments/lra/tasks/listops/train.py --attention-types standard linear piecewise --epochs 20
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from experiments.lra.configs.base_config import (
    ListOpsConfig,
    LISTOPS_METHODS,
    NUM_LANDMARKS,
    attention_hparams,
)
from experiments.lra.models.encoder import LRAEncoder
from experiments.lra.data.listops import get_listops_dataloaders, ListOpsDataset
from experiments.lra.utils.training import train_epoch, evaluate, set_seed, get_device


def print_banner(text: str, char: str = "="):
    """Print a formatted banner."""
    width = 70
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    parser = argparse.ArgumentParser(
        description="Train ListOps task with different attention mechanisms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--attention-types",
        nargs="+",
        default=LISTOPS_METHODS,
        choices=LISTOPS_METHODS,
        help="Attention mechanisms to compare. All are trained end-to-end, so the "
        "learned-projection baselines (Linformer E/F, Luna pack queries) are "
        "actually optimised — the fair comparison the untrained accuracy sweep "
        "could not give.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/lra/listops",
        help="Directory for ListOps data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Sequence length (default: use config value of 2000). Lower for "
        "quick smoke runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (auto will pick best available)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/lra/listops",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers (0=main thread)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: use config value of 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay (default: use config value of 1e-4)",
    )
    args = parser.parse_args()

    # Print configuration
    print_banner("LRA ListOps Benchmark")
    print("Configuration:")
    print(f"  Attention types: {', '.join(args.attention_types)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config = ListOpsConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.random_seed = args.seed
    if args.max_length is not None:
        config.max_length = args.max_length
        print(f"  Sequence length (custom): {config.max_length}")

    # Override learning rate and weight decay if provided
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
        print(f"  Learning rate (custom): {config.learning_rate}")
    else:
        print(f"  Learning rate: {config.learning_rate}")

    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
        print(f"  Weight decay (custom): {config.weight_decay}")
    else:
        print(f"  Weight decay: {config.weight_decay}")

    # Get device
    device = get_device(args.device)

    # Adjust data directory for quick test to avoid conflicts
    data_dir = args.data_dir
    if os.environ.get('LISTOPS_QUICK_TEST', '').lower() in ('1', 'true', 'yes'):
        data_dir = f"{args.data_dir}_quick"
        print(f"\n⚡ Quick Test Mode: Using separate data directory: {data_dir}")

    # Load data
    print("\nLoading ListOps dataset...")
    train_loader, val_loader, test_loader = get_listops_dataloaders(
        data_dir=data_dir,
        batch_size=config.batch_size,
        max_length=config.max_length,
        download=True,
        num_workers=args.num_workers,
    )

    vocab_size = ListOpsDataset.get_vocab_size()
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Train each attention type
    all_results = {}

    for attention_type in args.attention_types:
        print_banner(f"{attention_type.upper()} Attention", char="-")

        # Set seed for reproducibility
        set_seed(config.random_seed)

        # Matched-budget per-method hyperparameters (fair rank/feature/pack sizes).
        attn_kwargs = attention_hparams(
            attention_type, config.max_length, config.pooling_mode
        )
        if attn_kwargs:
            print(f"Attention hyperparameters: {attn_kwargs}")

        # Pad the (CLS-extended) sequence to a multiple of the Nyström landmark
        # count so its segment-mean landmarks divide evenly. Applied uniformly to
        # EVERY method (not just Nyström) so all methods train on the identical
        # input; the padded positions are masked and cannot affect any output.
        model = LRAEncoder(
            vocab_size=vocab_size,
            max_seq_len=config.max_length,
            num_classes=config.num_classes,
            emb_dim=config.emb_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            attention_type=attention_type,
            dropout=config.dropout,
            pooling_mode=config.pooling_mode,
            attn_kwargs=attn_kwargs,
            pad_to_multiple_of=NUM_LANDMARKS,
            device=device,
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

        # Optimizer and criterion
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_acc = 0.0
        train_history = []
        val_history = []

        start_time = time.time()

        for epoch in range(1, config.num_epochs + 1):
            # Train
            train_metrics = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                grad_clip=config.grad_clip,
            )
            train_history.append(train_metrics)

            # Evaluate
            val_metrics = evaluate(
                model=model,
                eval_loader=val_loader,
                criterion=criterion,
                device=device,
            )
            val_history.append(val_metrics)

            # Print progress
            print(f"\nEpoch {epoch}/{config.num_epochs}")
            print(f"  Train: loss={train_metrics['loss']:.4f}, "
                  f"acc={train_metrics['accuracy']*100:.2f}%, "
                  f"time={format_time(train_metrics['time'])}")
            print(f"  Val:   loss={val_metrics['loss']:.4f}, "
                  f"acc={val_metrics['accuracy']*100:.2f}%")

            # Track best
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                print(f"  ✓ New best validation accuracy: {best_val_acc*100:.2f}%")

        total_time = time.time() - start_time

        # Final test evaluation
        test_metrics = evaluate(
            model=model,
            eval_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        # Store results
        results = {
            'attention_type': attention_type,
            'attn_kwargs': attn_kwargs,
            'padded_seq_len': model.position_embedding.num_embeddings,
            'config': config.to_dict(),
            'num_parameters': num_params,
            'train_history': train_history,
            'val_history': val_history,
            'test_metrics': test_metrics,
            'best_val_accuracy': best_val_acc,
            'total_time': total_time,
        }
        all_results[attention_type] = results

        # Print summary
        print(f"\n{attention_type.upper()} Results:")
        print(f"  Best validation accuracy: {best_val_acc*100:.2f}%")
        print(f"  Test accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"  Total training time: {format_time(total_time)}")
        print(f"  Average epoch time: {format_time(total_time / config.num_epochs)}")

    # Save all results
    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {output_file}")

    # Print comparison table
    print_banner("COMPARISON SUMMARY")

    # Header
    print(f"{'Method':<20} {'Val Acc':<12} {'Test Acc':<12} {'Time':<12} {'Speedup':<10}")
    print("-" * 70)

    # Get baseline time (standard attention)
    baseline_time = all_results.get('standard', {}).get('total_time', None)

    # Print each method
    for attention_type in args.attention_types:
        results = all_results[attention_type]
        val_acc = results['best_val_accuracy'] * 100
        test_acc = results['test_metrics']['accuracy'] * 100
        total_time = results['total_time']

        # Calculate speedup
        if baseline_time and attention_type != 'standard':
            speedup = baseline_time / total_time
            speedup_str = f"{speedup:.2f}×"
        else:
            speedup_str = "baseline" if attention_type == 'standard' else "-"

        print(f"{attention_type.capitalize():<20} "
              f"{val_acc:<12.2f} "
              f"{test_acc:<12.2f} "
              f"{format_time(total_time):<12} "
              f"{speedup_str:<10}")

    print("=" * 70)

    # LRA Baseline comparison
    lra_baseline = 36.37  # From LRA paper
    print(f"\nLRA Transformer Baseline: {lra_baseline:.2f}%")

    for attention_type in args.attention_types:
        test_acc = all_results[attention_type]['test_metrics']['accuracy'] * 100
        diff = test_acc - lra_baseline
        sign = "+" if diff >= 0 else ""
        print(f"  {attention_type.capitalize()}: {test_acc:.2f}% ({sign}{diff:.2f}%)")


if __name__ == "__main__":
    main()
