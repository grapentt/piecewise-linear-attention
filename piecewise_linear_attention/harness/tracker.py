"""Experiment-tracking abstraction.

A :class:`Tracker` receives run lifecycle events, parameters, metrics, and
artifacts. The harness calls it at well-defined points so that swapping in a
real backend (e.g. MLflow) is a one-class change with no edits to experiment
code. Two backends ship here:

- :class:`NoOpTracker` — the default; does nothing (zero-overhead, no deps).
- :class:`ConsoleTracker` — prints events for local runs.

A future ``MLflowTracker`` would subclass :class:`Tracker` and forward each call
to the ``mlflow`` API. No tracking dependency is added now.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Tracker(ABC):
    """Base class for experiment trackers.

    Subclasses implement the lifecycle hooks. All methods are optional to act on
    (a subclass may implement only those it cares about), but must exist.
    """

    @abstractmethod
    def start_run(self, name: Optional[str] = None) -> None:
        """Begin a run, optionally named."""

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Record run parameters (typically the config)."""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metric values, optionally at a training step."""

    @abstractmethod
    def log_artifact(self, path: str) -> None:
        """Record an output artifact by filesystem path."""

    @abstractmethod
    def end_run(self) -> None:
        """Finish the current run."""

    def __enter__(self) -> "Tracker":
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_run()


class NoOpTracker(Tracker):
    """Tracker that ignores every event. The zero-overhead default."""

    def start_run(self, name: Optional[str] = None) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def end_run(self) -> None:
        pass


class ConsoleTracker(Tracker):
    """Tracker that prints events to stdout for local runs."""

    def __init__(self):
        self._name: Optional[str] = None

    def start_run(self, name: Optional[str] = None) -> None:
        self._name = name or "run"
        print(f"[tracker] start run: {self._name}")

    def log_params(self, params: Dict[str, Any]) -> None:
        print(f"[tracker] params: {params}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        prefix = f"[tracker] step {step}" if step is not None else "[tracker] metrics"
        formatted = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"{prefix}: {formatted}")

    def log_artifact(self, path: str) -> None:
        print(f"[tracker] artifact: {path}")

    def end_run(self) -> None:
        print(f"[tracker] end run: {self._name}")
