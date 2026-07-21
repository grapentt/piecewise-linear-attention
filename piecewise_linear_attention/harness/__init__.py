"""Unified benchmark harness for attention experiments.

This subpackage provides config-driven, tracker-instrumented infrastructure for
running comparative attention experiments in one place:

- :mod:`~piecewise_linear_attention.harness.config` — the ``ExperimentConfig``
  dataclass that fully specifies a run.
- :mod:`~piecewise_linear_attention.harness.device` — device selection and
  seeding (single source of truth).
- :mod:`~piecewise_linear_attention.harness.tracker` — experiment-tracking
  abstraction with no-op and console backends (MLflow-ready).
- :mod:`~piecewise_linear_attention.harness.results` — versioned result schema
  and JSON I/O.
- :mod:`~piecewise_linear_attention.harness.loop` — task-agnostic train/eval.
- :mod:`~piecewise_linear_attention.harness.tasks` — task datasets.
"""

from .config import ExperimentConfig
from .device import get_device, set_seed
from .loop import accuracy, train_and_evaluate
from .results import ExperimentResult, load_result, save_result
from .tracker import ConsoleTracker, NoOpTracker, Tracker

__all__ = [
    "ExperimentConfig",
    "get_device",
    "set_seed",
    "train_and_evaluate",
    "accuracy",
    "ExperimentResult",
    "load_result",
    "save_result",
    "Tracker",
    "NoOpTracker",
    "ConsoleTracker",
]
