"""Optional MLflow experiment tracking for LRA runs.

MLflow is an *optional* dependency: the trainer works identically without it. All
tracking goes through :class:`MLflowTracker`, which is a no-op unless tracking is
explicitly enabled (``--mlflow``) *and* the ``mlflow`` package imports. That keeps
GPU/headless environments that don't have it from breaking, while giving a single
place to record the fair-comparison run (one MLflow run per attention method,
grouped under a parent run so the methods' curves stay comparable but separable).

What gets logged per method run:
- **params**: attention method, its matched ``attn_kwargs``, precision, the padded
  sequence length actually fed to attention, and the flat training config — i.e.
  everything needed to reproduce the number and to see that the budget was matched.
- **metrics** (per epoch, stepped): train/val loss and accuracy, epoch time.
- **final metrics**: test loss/accuracy, best val accuracy, total train time.
- **artifact**: the aggregate ``results.json`` (attached once, to the parent run).

Nothing here touches the attention math or the fairness protocol; it only observes.
"""
from __future__ import annotations

import numbers
from contextlib import contextmanager
from typing import Any, Dict, Optional


def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    """Flatten nested dicts into ``dotted.key -> value`` params.

    Recurses through nested dicts (``config``/``attn_kwargs`` are shallow in
    practice, but arbitrary depth is handled) and keeps keys readable
    (``attn.num_landmarks``, ``config.learning_rate``). Non-dict values — scalars,
    ``None``, lists, tuples — are stored as-is and stringified by the caller.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = obj


class MLflowTracker:
    """Thin, always-safe wrapper around MLflow.

    When ``enabled`` is False (the default) or ``mlflow`` cannot be imported, every
    method is a no-op and the trainer behaves exactly as before. This is the single
    gate so the calling code never needs ``if mlflow:`` guards.
    """

    def __init__(
        self,
        enabled: bool = False,
        experiment: str = "lra-listops",
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        self.enabled = False
        self._mlflow = None
        self._parent_run = None
        if not enabled:
            return
        try:
            import mlflow  # noqa: F401
        except ImportError:
            print(
                "⚠️  --mlflow requested but the 'mlflow' package is not installed; "
                "continuing without tracking (pip install mlflow to enable)."
            )
            return
        self._mlflow = mlflow
        self.enabled = True
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        # A parent run groups the per-method child runs of one comparison so they
        # share an experiment view but each keep their own curves.
        self._parent_run = mlflow.start_run(run_name=run_name)
        print(
            f"📊 MLflow tracking on: experiment='{experiment}', "
            f"uri='{mlflow.get_tracking_uri()}', parent_run={self._parent_run.info.run_id}"
        )

    # -- lifecycle ---------------------------------------------------------

    def log_parent_params(self, params: Dict[str, Any]) -> None:
        """Log run-wide params (seed, epochs, device...) on the parent run."""
        if not self.enabled:
            return
        flat: Dict[str, Any] = {}
        _flatten("", params, flat)
        self._mlflow.log_params({k: str(v) for k, v in flat.items()})

    @contextmanager
    def method_run(self, method: str):
        """Context manager for a single method's nested run.

        Yields ``self`` so the caller uses the same ``log_*`` API; outside tracking
        it yields a live-but-inert tracker so the ``with`` body is unconditional.
        """
        if not self.enabled:
            yield self
            return
        with self._mlflow.start_run(run_name=method, nested=True):
            self._mlflow.set_tag("attention_method", method)
            yield self

    def start_method(self, method: str) -> None:
        """Begin a nested per-method run (flat-code alternative to ``method_run``).

        Pair with :meth:`end_method`. A no-op when tracking is disabled, so the
        trainer can call it unconditionally without an ``if``/``with`` wrapper.

        Exception-safety: if a *previous* method's run was left open (its
        :meth:`end_method` skipped because that iteration raised), this first ends
        it so the new run nests under the parent rather than under the leaked
        sibling. Without this, one failed method would silently re-parent every
        method after it, corrupting the comparison grouping.
        """
        if not self.enabled:
            return
        active = self._mlflow.active_run()
        if active is not None and (
            self._parent_run is None
            or active.info.run_id != self._parent_run.info.run_id
        ):
            # A leaked nested (method) run is on top of the stack — close it so we
            # start clean under the parent.
            self._mlflow.end_run()
        self._mlflow.start_run(run_name=method, nested=True)
        self._mlflow.set_tag("attention_method", method)

    def end_method(self) -> None:
        """End the nested per-method run started by :meth:`start_method`.

        Only ends a run if the active run is actually a nested (non-parent) run,
        so a double call or a call after the run already closed cannot accidentally
        end the parent.
        """
        if not self.enabled:
            return
        active = self._mlflow.active_run()
        if active is None:
            return
        if self._parent_run is not None and active.info.run_id == self._parent_run.info.run_id:
            return  # never end the parent here — that's close()'s job
        self._mlflow.end_run()

    def log_method_params(self, params: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        flat: Dict[str, Any] = {}
        _flatten("", params, flat)
        self._mlflow.log_params({k: str(v) for k, v in flat.items()})

    def log_epoch(self, epoch: int, train: Dict[str, float], val: Dict[str, float]) -> None:
        """Log stepped per-epoch train/val metrics (skips non-numeric entries)."""
        if not self.enabled:
            return
        metrics: Dict[str, float] = {}
        for split, d in (("train", train), ("val", val)):
            for k, v in d.items():
                if isinstance(v, numbers.Number):
                    metrics[f"{split}_{k}"] = float(v)
        self._mlflow.log_metrics(metrics, step=epoch)

    def log_final(self, test: Dict[str, float], best_val_acc: float, total_time: float) -> None:
        if not self.enabled:
            return
        metrics = {f"test_{k}": float(v) for k, v in test.items() if isinstance(v, numbers.Number)}
        metrics["best_val_accuracy"] = float(best_val_acc)
        metrics["total_train_time_sec"] = float(total_time)
        self._mlflow.log_metrics(metrics)

    def log_artifact(self, path: str) -> None:
        """Attach a file (e.g. the aggregate results.json) to the parent run."""
        if not self.enabled:
            return
        self._mlflow.log_artifact(path)

    def close(self) -> None:
        """End all open runs, draining any leaked nested run before the parent.

        A bare ``end_run()`` only pops the top of MLflow's run stack, so if a
        method run was left open (its iteration raised) it would end *that* and
        leave the parent stuck in RUNNING. Draining the whole stack guarantees the
        parent reaches a terminal state.
        """
        if not self.enabled:
            return
        while self._mlflow.active_run() is not None:
            self._mlflow.end_run()
