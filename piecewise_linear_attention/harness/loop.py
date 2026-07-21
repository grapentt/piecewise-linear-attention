"""Task-agnostic training and evaluation loop.

The loop is decoupled from any specific task through three pluggable callables:

- ``batch_fn(generator) -> (inputs, targets)`` produces a batch.
- ``forward_fn(model, inputs) -> outputs`` runs the model (the "forward adapter",
  so classification vs seq2seq vs regression all fit the same loop).
- ``loss_fn(outputs, targets) -> scalar`` and ``metric_fn(outputs, targets) ->
  float`` score the outputs.

This keeps the optimization machinery (AdamW, grad clipping, seeding, device
placement, tracker calls) in one place while tasks vary only the four callables.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .device import get_device, set_seed
from .tracker import NoOpTracker, Tracker

BatchFn = Callable[[torch.Generator], Tuple[torch.Tensor, torch.Tensor]]
ForwardFn = Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


def default_forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Forward adapter for single-input models: ``model(inputs)``."""
    return model(inputs)


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 classification accuracy for logits ``(batch, classes)``."""
    preds = outputs.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def _evaluate(
    model: torch.nn.Module,
    batch_fn: "BatchFn",
    *,
    forward_fn: "ForwardFn",
    metric_fn: "MetricFn",
    eval_batches: int,
    eval_seed: int,
    device: torch.device,
) -> float:
    """Mean metric over ``eval_batches`` drawn from a fixed-seed generator.

    Uses its own generator each call so the evaluation set is identical across
    calls (and across runs), making periodic-eval curves comparable step to step.
    Restores the model's train/eval mode so it can be called mid-training.
    """
    was_training = model.training
    model.eval()
    eval_gen = torch.Generator().manual_seed(eval_seed)
    total = 0.0
    with torch.no_grad():
        for _ in range(eval_batches):
            inputs, targets = batch_fn(eval_gen)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = forward_fn(model, inputs)
            total += metric_fn(outputs, targets)
    if was_training:
        model.train()
    return total / max(eval_batches, 1)


def train_and_evaluate(
    model: torch.nn.Module,
    batch_fn: BatchFn,
    *,
    steps: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: Optional[torch.device] = None,
    forward_fn: ForwardFn = default_forward,
    loss_fn: LossFn = torch.nn.functional.cross_entropy,
    metric_fn: MetricFn = accuracy,
    eval_batches: int = 20,
    eval_seed: int = 999,
    eval_every: int = 0,
    tracker: Optional[Tracker] = None,
    log_every: int = 0,
) -> Dict[str, float]:
    """Train ``model`` for ``steps`` then evaluate it.

    Parameters
    ----------
    model:
        The module to train (moved to ``device``).
    batch_fn:
        Produces ``(inputs, targets)`` given a torch generator.
    steps:
        Number of optimization steps.
    lr, weight_decay, grad_clip:
        Optimizer/regularization settings.
    seed:
        Seed for training RNG (weights + train batches).
    device:
        Compute device; auto-selected if ``None``.
    forward_fn, loss_fn, metric_fn:
        Pluggable forward adapter, loss, and metric.
    eval_batches, eval_seed:
        Evaluation batch count and its (separate) RNG seed.
    eval_every:
        If > 0, evaluate every ``eval_every`` steps during training and record
        the ``(step, accuracy)`` pairs under the ``"curve"`` key of the return
        value. This surfaces late/grokking-style transitions that a single
        end-of-training snapshot would hide. Default ``0`` disables it, leaving
        behavior identical to a plain train-then-eval run.
    tracker:
        Optional experiment tracker; :class:`NoOpTracker` if ``None``.
    log_every:
        If > 0, log train loss to the tracker every ``log_every`` steps.

    Returns
    -------
    dict
        ``{"accuracy": ..., "loss": ..., "train_time_s": ...}`` (or the metric's
        semantics for a custom ``metric_fn``). When ``eval_every > 0`` a
        ``"curve"`` key holds a list of ``{"step", "accuracy"}`` dicts.
    """
    device = device or get_device()
    tracker = tracker or NoOpTracker()
    set_seed(seed)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_gen = torch.Generator().manual_seed(seed + 100)

    curve: List[Dict[str, float]] = []
    model.train()
    last_loss = float("nan")
    start = time.perf_counter()
    for step in range(steps):
        inputs, targets = batch_fn(train_gen)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = forward_fn(model, inputs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        last_loss = loss.item()
        if log_every and step % log_every == 0:
            tracker.log_metrics({"train_loss": last_loss}, step=step)
        if eval_every and step > 0 and step % eval_every == 0:
            acc_at_step = _evaluate(
                model,
                batch_fn,
                forward_fn=forward_fn,
                metric_fn=metric_fn,
                eval_batches=eval_batches,
                eval_seed=eval_seed,
                device=device,
            )
            curve.append({"step": step, "accuracy": acc_at_step})
            tracker.log_metrics({"eval_accuracy": acc_at_step}, step=step)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    train_time = time.perf_counter() - start

    # Final evaluation on the same fixed-seed set as any periodic evals.
    metric_value = _evaluate(
        model,
        batch_fn,
        forward_fn=forward_fn,
        metric_fn=metric_fn,
        eval_batches=eval_batches,
        eval_seed=eval_seed,
        device=device,
    )

    metrics = {
        "accuracy": metric_value,
        "loss": last_loss,
        "train_time_s": train_time,
    }
    if eval_every:
        curve.append({"step": steps, "accuracy": metric_value})
        metrics["curve"] = curve
    tracker.log_metrics({k: v for k, v in metrics.items() if k != "curve"})
    return metrics
