"""Experiment configuration.

:class:`ExperimentConfig` is the single object that fully specifies a run —
model shape, attention mechanism, optimization, and task. It is logged verbatim
by the tracker and embedded in results, so a run can always be reproduced from
its recorded config. A best-effort git SHA and a schema version are captured so
old result files remain interpretable as the schema evolves.
"""

import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

# Bump when the config field layout changes in a backward-incompatible way.
CONFIG_SCHEMA_VERSION = 1


def _git_sha() -> Optional[str]:
    """Best-effort short git SHA of the working tree, or None if unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return sha.decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


@dataclass
class ExperimentConfig:
    """Full specification of a single benchmark run.

    Parameters
    ----------
    attention_type:
        Registered attention name (see ``available_attention_types``).
    task:
        Task identifier (e.g. ``"synthetic_recall"``).
    hidden_dim, num_heads, num_layers:
        Model shape.
    seq_len, vocab_size:
        Task/sequence sizing.
    batch_size, steps, lr, weight_decay, grad_clip, dropout:
        Optimization.
    seed:
        Global RNG seed.
    device:
        Device preference passed to ``get_device`` (``"auto"`` by default).
    attention_kwargs:
        Mechanism-specific options forwarded to ``build_attention`` (e.g.
        ``{"num_anchors": 4}`` or ``{"kernel_type": "relu"}``).
    eval_batches:
        Number of batches used for evaluation.
    name:
        Optional human-readable run name.
    """

    attention_type: str
    task: str = "synthetic_recall"

    # Model shape
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2

    # Task / sequence
    seq_len: int = 32
    vocab_size: int = 16

    # Optimization
    batch_size: int = 64
    steps: int = 3000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    dropout: float = 0.0

    # Reproducibility / environment
    seed: int = 0
    device: str = "auto"

    # Mechanism-specific options forwarded to build_attention.
    attention_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Evaluation
    eval_batches: int = 20

    name: Optional[str] = None

    # Provenance (populated automatically; overridable for reproduction).
    schema_version: int = CONFIG_SCHEMA_VERSION
    git_sha: Optional[str] = field(default_factory=_git_sha)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-friendly)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Reconstruct from a dict, ignoring unknown keys.

        Unknown keys are dropped rather than raising so that configs written by
        a newer schema can still be loaded (best-effort forward compatibility).
        """
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
