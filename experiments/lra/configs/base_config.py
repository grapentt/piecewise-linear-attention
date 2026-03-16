"""Base configuration for LRA benchmarks.

Matches the official LRA configuration for fair comparison.
Based on: lra_benchmarks/listops/configs/base_listops_config.py
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class LRAConfig:
    """Base LRA configuration matching official benchmarks."""

    # Model architecture (LRA standard)
    emb_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    qkv_dim: int = 512  # Same as emb_dim for LRA
    mlp_dim: int = 1024
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3  # 0.001 - much more reasonable for transformers
    num_epochs: int = 20  # For quick iteration; LRA uses 5000 steps
    weight_decay: float = 1e-4  # Reduced from 1e-1 which was too aggressive
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # Task-specific (override in task configs)
    max_length: int = 2000
    num_classes: int = 10
    vocab_size: Optional[int] = None  # Set by dataset

    # Experiment
    random_seed: int = 42

    # Pooling strategy
    pooling_mode: str = "CLS"  # or "MEAN"

    # Device
    device: str = "cuda"  # or "mps" or "cpu"

    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class ListOpsConfig(LRAConfig):
    """ListOps-specific configuration."""
    max_length: int = 2000
    num_classes: int = 10
    # ListOps uses character-level tokenization, vocab set by dataset
