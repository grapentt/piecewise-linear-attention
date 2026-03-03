"""Train BERT with different attention types on language understanding tasks.

This script compares StandardAttention, LinearAttention, and PiecewiseAttention
in a BERT model on text classification tasks (GLUE benchmark).

Usage:
    python experiments/bert/train.py
    python experiments/bert/train.py --task glue/sst2 --epochs 3
    python experiments/bert/train.py --attention-types piecewise --save results/bert/sst2.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from piecewise_linear_attention.models.bert import bert_tiny, bert_base


def get_glue_sst2_loaders(batch_size: int = 32, max_length: int = 128):
    """Create SST-2 (sentiment classification) data loaders."""
    # Load dataset
    dataset = load_dataset("glue", "sst2")

    # Use BERT tokenizer (or any compatible tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, tokenizer


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    num_samples = len(train_loader.dataset)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "time_seconds": epoch_time,
        "samples_per_second": num_samples / epoch_time,
    }


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return {"loss": avg_loss, "accuracy": accuracy}


def run_bert_experiment(
    attention_type: str,
    task: str = "glue/sst2",
    model_size: str = "tiny",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    device: str = "cpu",
) -> Dict:
    """Run BERT experiment for one attention type.

    Args:
        attention_type: "standard", "linear", or "piecewise"
        task: Task name (e.g., "glue/sst2")
        model_size: "tiny" or "base"
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use

    Returns:
        Dictionary with training results
    """
    print("=" * 80)
    print(f"Training BERT-{model_size.upper()} with {attention_type.upper()} Attention")
    print(f"Task: {task.upper()}")
    print("=" * 80)

    # Load dataset
    if task == "glue/sst2":
        train_loader, val_loader, tokenizer = get_glue_sst2_loaders(batch_size)
        num_labels = 2  # Binary classification
        vocab_size = tokenizer.vocab_size
    else:
        raise NotImplementedError(f"Task {task} not yet implemented")

    # Create model
    if model_size == "tiny":
        model = bert_tiny(
            vocab_size=vocab_size,
            num_labels=num_labels,
            attention_type=attention_type,
            device=device,
        )
    elif model_size == "base":
        model = bert_base(
            vocab_size=vocab_size,
            num_labels=num_labels,
            attention_type=attention_type,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    train_history = []
    val_history = []
    total_train_time = 0.0

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        total_train_time += train_metrics["time_seconds"]
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, "
              f"Time: {train_metrics['time_seconds']:.1f}s, "
              f"Speed: {train_metrics['samples_per_second']:.1f} samples/s")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")

    avg_samples_per_second = sum(m["samples_per_second"] for m in train_history) / len(train_history)

    print("\n" + "=" * 80)
    print(f"FINAL RESULTS - {attention_type.upper()}")
    print("=" * 80)
    print(f"Val Accuracy: {val_metrics['accuracy']:.2f}%")
    print(f"Val Loss: {val_metrics['loss']:.4f}")
    print(f"Total Training Time: {total_train_time:.1f}s")
    print(f"Avg Training Speed: {avg_samples_per_second:.1f} samples/s")
    print("=" * 80)

    return {
        "attention_type": attention_type,
        "model_size": model_size,
        "task": task,
        "num_params": num_params,
        "final_train_accuracy": train_metrics["accuracy"],
        "final_val_accuracy": val_metrics["accuracy"],
        "final_train_loss": train_metrics["loss"],
        "final_val_loss": val_metrics["loss"],
        "total_train_time": total_train_time,
        "avg_samples_per_second": avg_samples_per_second,
        "train_history": train_history,
        "val_history": val_history,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train BERT with different attention types"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="glue/sst2",
        help="Task name (default: glue/sst2)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="tiny",
        choices=["tiny", "base"],
        help="Model size (default: tiny)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs (default: 3)",
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
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--attention-types",
        type=str,
        nargs="+",
        default=["standard", "linear", "piecewise"],
        choices=["standard", "linear", "piecewise"],
        help="Attention types to compare (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (default: cpu)",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("BERT LANGUAGE UNDERSTANDING BENCHMARK")
    print("=" * 80)
    print(f"Task: {args.task.upper()}")
    print(f"Model: BERT-{args.model_size.upper()}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Comparing: {', '.join(args.attention_types)}")
    print(f"Device: {args.device}")
    print()

    # Run experiments
    results = []
    for attention_type in args.attention_types:
        try:
            result = run_bert_experiment(
                attention_type=attention_type,
                task=args.task,
                model_size=args.model_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=args.device,
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error with {attention_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Comparison summary
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Attention':<15} {'Val Acc (%)':<15} {'Val Loss':<15} {'Train Time (s)':<18} {'Speed (samples/s)':<20}")
        print("-" * 90)
        for result in results:
            print(f"{result['attention_type'].capitalize():<15} "
                  f"{result['final_val_accuracy']:<15.2f} "
                  f"{result['final_val_loss']:<15.4f} "
                  f"{result['total_train_time']:<18.1f} "
                  f"{result['avg_samples_per_second']:<20.1f}")
        print("-" * 90)

        # Speedup comparison (if standard exists)
        if any(r["attention_type"] == "standard" for r in results):
            standard = next(r for r in results if r["attention_type"] == "standard")
            baseline_time = standard["total_train_time"]
            print("\nSpeedup vs Standard Attention:")
            for result in results:
                if result["attention_type"] != "standard":
                    speedup = baseline_time / result["total_train_time"]
                    print(f"  {result['attention_type'].capitalize()}: {speedup:.2f}× faster")
            print()

    # Save results
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {save_path}")

    print("\n" + "=" * 80)
    print("✅ Experiment complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
