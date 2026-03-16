"""Training utilities for translation benchmarks."""

import time
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: The transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to use

    Returns:
        Dictionary with loss, perplexity, time, and throughput metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <pad>

    for source, target in tqdm(dataloader, desc="Training", leave=False):
        # Forward pass (use all but last target token as input)
        logits = model(source, target[:, :-1])

        # Compute loss (predict next token)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target[:, 1:].reshape(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches

    return {
        "loss": avg_loss,
        "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        "time_seconds": epoch_time,
        "samples_per_second": len(dataloader.dataset) / epoch_time,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model.

    Args:
        model: The transformer model
        dataloader: Evaluation data loader
        device: Device to use

    Returns:
        Dictionary with loss and perplexity
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for source, target in tqdm(dataloader, desc="Evaluating", leave=False):
            # Forward pass
            logits = model(source, target[:, :-1])

            # Compute loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target[:, 1:].reshape(-1)
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
    }
