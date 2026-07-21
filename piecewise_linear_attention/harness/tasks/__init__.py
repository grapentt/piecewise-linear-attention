"""Task datasets and models for the benchmark harness."""

from .synthetic_recall import RecallModel, make_recall_batch

__all__ = ["RecallModel", "make_recall_batch"]
