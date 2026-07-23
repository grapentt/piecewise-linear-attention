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


def attention_hparams(method: str, max_length: int, pooling_mode: str = "CLS") -> dict:
    """Matched-budget per-method attention hyperparameters for a fair comparison.

    Every efficient baseline compresses the sequence to a fixed rank; sizing those
    ranks comparably (all in the same ~128–256 ballpark) keeps the comparison fair
    — no method is starved or over-provisioned relative to the others. The values
    follow the widely used LRA configurations (Nyströmformer / Skyformer report
    landmark/feature counts in this range for ListOps).

    Returns a dict suitable for ``LRAEncoder(attn_kwargs=...)`` / ``build_attention``.
    Keys not consumed by a given mechanism are ignored by its builder, so the same
    call site works for every method.

    Parameters
    ----------
    method:
        Registered attention name (``standard``/``linear``/``performer``/
        ``nystromformer``/``linformer``/``luna``/``piecewise``).
    max_length:
        Task sequence length (before the CLS token).
    pooling_mode:
        ``"CLS"`` adds one position, so the effective key length — and hence the
        Linformer projection size — is ``max_length + 1``.

    Notes
    -----
    - **Linformer** requires ``max_seq_len >= key_len`` (its sequence-axis
      projection is defined only up to that length); it is set to the CLS-extended
      length so the baseline builds instead of raising.
    - **Nyström** ``num_landmarks`` must divide the sequence length for an exact
      segment-mean; the forward falls back to the largest divisor otherwise, so a
      round value is used here.
    """
    key_len = max_length + 1 if pooling_mode == "CLS" else max_length
    budget = 256  # shared rank budget: features / rank k / pack size
    tables = {
        "standard": {},
        "linear": {},
        "performer": {"num_features": budget},
        "nystromformer": {"num_landmarks": 128},
        "linformer": {"k": budget, "max_seq_len": key_len},
        "luna": {"num_pack": budget},
        # Multi-anchor k-means piecewise; anchor count is the method's own dial.
        # Real-activation sweeps show m>1 is a genuine accuracy lever, so the
        # comparison uses a multi-anchor config rather than the single-anchor mean.
        "piecewise_kmeans": {"num_anchors": 16, "kmeans_iters": 3},
    }
    return dict(tables.get(method, {}))


#: Methods trained end-to-end in the fair ListOps comparison. Every mechanism's
#: defining component (Linformer E/F, Luna pack queries, piecewise anchors) is
#: optimised inside the encoder, which is the structural fix for scoring the
#: learned-projection baselines at random init. ``piecewise_kmeans`` is the
#: multi-anchor build (its ``num_anchors`` actually takes effect, unlike the
#: single-anchor ``piecewise`` mean).
LISTOPS_METHODS = [
    "standard",
    "linear",
    "performer",
    "nystromformer",
    "linformer",
    "luna",
    "piecewise_kmeans",
]
