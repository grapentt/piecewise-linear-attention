"""Training utilities for LRA experiments.

Extends patterns from experiments/utils/training.py adapted for classification tasks.
"""
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        epoch: Current epoch number
        grad_clip: Gradient clipping value

    Returns:
        Dictionary of metrics (loss, accuracy, time, throughput)
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, padding_mask)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Compute metrics
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()

        total_loss += loss.item() * input_ids.size(0)
        total_correct += correct
        total_samples += input_ids.size(0)

        # Update progress bar
        current_acc = correct / input_ids.size(0)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.4f}',
        })

    epoch_time = time.time() - start_time
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'time': epoch_time,
        'samples_per_sec': total_samples / epoch_time,
    }


def evaluate(
    model: nn.Module,
    eval_loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model.

    Args:
        model: The model to evaluate
        eval_loader: Evaluation data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Dictionary of metrics (loss, accuracy)
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids, padding_mask)
            loss = criterion(logits, labels)

            # Compute metrics
            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels).sum().item()

            total_loss += loss.item() * input_ids.size(0)
            total_correct += correct
            total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
    }


def set_seed(seed: int):
    """Set random seed for reproducibility across all frameworks.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_name: str = "auto") -> torch.device:
    """Get the appropriate device.

    Args:
        device_name: "auto", "cuda", "mps", or "cpu"

    Returns:
        torch.device object
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        if device_name == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        elif device_name == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device(device_name)

    return device
