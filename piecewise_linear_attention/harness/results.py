"""Versioned result schema and JSON I/O.

An :class:`ExperimentResult` bundles the config that produced a run with its
metrics and environment, under an explicit schema version so result files stay
interpretable as the schema evolves. Results are written as human-readable JSON.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Bump when the result field layout changes in a backward-incompatible way.
RESULT_SCHEMA_VERSION = 1


@dataclass
class ExperimentResult:
    """The recorded outcome of a single run.

    Parameters
    ----------
    config:
        The ``ExperimentConfig.to_dict()`` that produced this run.
    metrics:
        Final metric values (e.g. ``{"accuracy": 0.95, "loss": 0.04}``).
    environment:
        Environment details (device, torch version, timestamps, ...).
    history:
        Optional per-step metric history.
    """

    config: Dict[str, Any]
    metrics: Dict[str, float]
    environment: Dict[str, Any] = field(default_factory=dict)
    history: Optional[Dict[str, list]] = None
    schema_version: int = RESULT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-friendly)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Reconstruct from a dict, ignoring unknown keys for forward-compat."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def save_result(result: ExperimentResult, path: Union[str, Path]) -> Path:
    """Write a result to ``path`` as indented JSON, creating parent dirs.

    Returns the resolved path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)
    return path


def load_result(path: Union[str, Path]) -> ExperimentResult:
    """Read a result written by :func:`save_result`."""
    with Path(path).open() as f:
        return ExperimentResult.from_dict(json.load(f))
