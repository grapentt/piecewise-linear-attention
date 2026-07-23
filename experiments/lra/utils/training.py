"""Training utilities for LRA experiments.

Extends patterns from experiments/utils/training.py adapted for classification tasks.
"""
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Optional
from contextlib import nullcontext


def _autocast_ctx(device: torch.device, amp_dtype: Optional[torch.dtype]):
    """Autocast context for mixed precision, or a no-op when ``amp_dtype`` is None.

    bf16 needs no ``GradScaler`` (it keeps fp32's exponent range), so the forward is
    simply wrapped in ``torch.autocast`` while master weights stay fp32.
    """
    if amp_dtype is None:
        return nullcontext()
    device_type = device.type if isinstance(device, torch.device) else str(device)
    return torch.autocast(device_type=device_type, dtype=amp_dtype)


def _sync(device: torch.device) -> None:
    """Block until queued device work is done, so wall-clock stamps are truthful.

    CUDA kernels launch asynchronously: without a synchronize, a ``time.time()``
    taken right after ``optimizer.step()`` records only *launch* time, and the real
    compute is billed to whenever the next ``.item()`` forces a sync — smearing
    compute into the following iteration's data-load window. Syncing at each stamp
    attributes data-load vs compute time correctly. No-op on CPU/MPS.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    amp_dtype: Optional[torch.dtype] = None,
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
        amp_dtype: If set (e.g. ``torch.bfloat16``), run the forward under
            ``torch.autocast`` in that dtype; master weights stay fp32. ``None``
            = full fp32.

    Returns:
        Dictionary of metrics (loss, accuracy, time, throughput)
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    data_time = 0.0      # wall time waiting on the dataloader (collation + h2d)
    compute_time = 0.0   # wall time in forward + backward + optimizer step
    start_time = time.time()
    end = time.time()  # timestamp of the previous iteration's close

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        # Data-load window: from the previous iteration's close until the batch is
        # on-device and ready to compute on. Includes DataLoader collation and the
        # host->device copies (synced so the copies are actually finished).
        input_ids = batch['input_ids'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        labels = batch['labels'].to(device)
        _sync(device)
        data_time += time.time() - end

        # Compute window: forward + backward + step, synced at the close so async
        # CUDA kernels are billed here rather than smeared into the next data window.
        compute_start = time.time()
        optimizer.zero_grad()
        with _autocast_ctx(device, amp_dtype):
            logits = model(input_ids, padding_mask)
            loss = criterion(logits, labels)

        # Backward pass (bf16 autocast needs no GradScaler)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        _sync(device)
        compute_time += time.time() - compute_start

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
        end = time.time()

    epoch_time = time.time() - start_time
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'time': epoch_time,
        'data_time': data_time,
        'compute_time': compute_time,
        'samples_per_sec': total_samples / epoch_time,
        'compute_samples_per_sec': total_samples / compute_time if compute_time else 0.0,
    }


def evaluate(
    model: nn.Module,
    eval_loader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """Evaluate model.

    Args:
        model: The model to evaluate
        eval_loader: Evaluation data loader
        criterion: Loss function
        device: Device to use
        amp_dtype: If set, run the forward under ``torch.autocast`` in that dtype
            (matches the training precision). ``None`` = full fp32.

    Returns:
        Dictionary of metrics (loss, accuracy)
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    compute_time = 0.0
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass (synced so compute time reflects the real kernel cost)
            _sync(device)
            compute_start = time.time()
            with _autocast_ctx(device, amp_dtype):
                logits = model(input_ids, padding_mask)
                loss = criterion(logits, labels)
            _sync(device)
            compute_time += time.time() - compute_start

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
        'time': time.time() - start_time,
        'compute_time': compute_time,
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
