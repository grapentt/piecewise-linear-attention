"""Train Vision Transformer with different attention types on image classification.

This script compares StandardAttention, LinearAttention, and PiecewiseAttention
in a Vision Transformer on image classification tasks (CIFAR-10, CIFAR-100, ImageNet).

Usage:
    python experiments/vision/train.py
    python experiments/vision/train.py --dataset cifar100 --epochs 100
    python experiments/vision/train.py --attention-types piecewise --save results/vision/cifar10.json
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
from torchvision import datasets, transforms

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from piecewise_linear_attention.models.vit import vit_tiny, vit_small, vit_base


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 4):
    """Create CIFAR-10 data loaders with data augmentation."""
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "time_seconds": epoch_time,
        "samples_per_second": len(train_loader.dataset) / epoch_time,
    }


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    return {"loss": avg_loss, "accuracy": accuracy}


def run_vit_experiment(
    attention_type: str,
    model_size: str = "tiny",
    dataset: str = "cifar10",
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> Dict:
    """Run ViT experiment for one attention type.

    Args:
        attention_type: "standard", "linear", or "piecewise"
        model_size: "tiny", "small", or "base"
        dataset: "cifar10" or "cifar100"
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use

    Returns:
        Dictionary with training results
    """
    print("=" * 80)
    print(f"Training ViT-{model_size.upper()} with {attention_type.upper()} Attention")
    print(f"Dataset: {dataset.upper()}")
    print("=" * 80)

    # Load dataset
    if dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(batch_size)
        num_classes = 10
    elif dataset == "cifar100":
        # TODO: Implement CIFAR-100 loader
        raise NotImplementedError("CIFAR-100 not yet implemented")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Create model
    if model_size == "tiny":
        model = vit_tiny(
            num_classes=num_classes,
            attention_type=attention_type,
            img_size=32,  # CIFAR image size
            patch_size=4,  # Smaller patches for 32×32 images
            device=device,
        )
    elif model_size == "small":
        model = vit_small(
            num_classes=num_classes,
            attention_type=attention_type,
            img_size=32,
            patch_size=4,
            device=device,
        )
    elif model_size == "base":
        model = vit_base(
            num_classes=num_classes,
            attention_type=attention_type,
            img_size=32,
            patch_size=4,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    train_history = []
    test_history = []
    total_train_time = 0.0

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        total_train_time += train_metrics["time_seconds"]
        train_history.append(train_metrics)
        test_history.append(test_metrics)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, "
                  f"Time: {train_metrics['time_seconds']:.1f}s, "
                  f"Speed: {train_metrics['samples_per_second']:.1f} samples/s")
            print(f"  Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%")

    avg_samples_per_second = sum(m["samples_per_second"] for m in train_history) / len(train_history)

    print("\n" + "=" * 80)
    print(f"FINAL RESULTS - {attention_type.upper()}")
    print("=" * 80)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Total Training Time: {total_train_time:.1f}s")
    print(f"Avg Training Speed: {avg_samples_per_second:.1f} samples/s")
    print("=" * 80)

    return {
        "attention_type": attention_type,
        "model_size": model_size,
        "dataset": dataset,
        "num_params": num_params,
        "final_train_accuracy": train_metrics["accuracy"],
        "final_test_accuracy": test_metrics["accuracy"],
        "final_train_loss": train_metrics["loss"],
        "final_test_loss": test_metrics["loss"],
        "total_train_time": total_train_time,
        "avg_samples_per_second": avg_samples_per_second,
        "train_history": train_history,
        "test_history": test_history,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Vision Transformer with different attention types"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset (default: cifar10)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base"],
        help="Model size (default: tiny)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
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
    print("VISION TRANSFORMER IMAGE CLASSIFICATION BENCHMARK")
    print("=" * 80)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: ViT-{args.model_size.upper()}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Comparing: {', '.join(args.attention_types)}")
    print(f"Device: {args.device}")
    print()

    # Run experiments
    results = []
    for attention_type in args.attention_types:
        try:
            result = run_vit_experiment(
                attention_type=attention_type,
                model_size=args.model_size,
                dataset=args.dataset,
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
        print(f"{'Attention':<15} {'Test Acc (%)':<15} {'Test Loss':<15} {'Train Time (s)':<18} {'Speed (samples/s)':<20}")
        print("-" * 90)
        for result in results:
            print(f"{result['attention_type'].capitalize():<15} "
                  f"{result['final_test_accuracy']:<15.2f} "
                  f"{result['final_test_loss']:<15.4f} "
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
