"""Opt-in, near-zero-cost phase timing for the piecewise-attention forward.

The multi-anchor forward has two terms whose relative cost shifts with the anchor
count ``m`` and the device: the k-means *clustering* that selects the anchors, and
the ``O(batch·m·n·d²)`` second-moment *einsum* that builds the per-anchor Jacobian
operator. Wall-clock at the training-step level cannot tell them apart, and neither
can the coarse data-vs-compute split in the training loop. This module lets those
two sections time *themselves* when — and only when — a probe is installed, so a
diagnostic can attribute a real training step's forward cost without forking the
model or patching it from outside.

Design
------
* A single module-global points at the active :class:`PhaseProbe` (``None`` when
  profiling is off). Every instrumented section guards on ``_active is not None``
  before doing any timing work, so the disabled path is one attribute load and a
  branch — the model runs exactly as it did before when nothing is profiling.
* Timing is CUDA-aware: on CUDA the section is bracketed by ``cuda.Event`` records
  (the only truthful way to time async kernels), elsewhere by ``perf_counter``.
  The probe decides whether to synchronize when it reads the elapsed time.
* Nothing here imports the training stack; it depends only on ``torch``. The
  instrumented call sites (``KMeansAnchor.select`` / ``PiecewiseAttention.
  _compute_multi_anchor``) call :func:`phase_timer` unconditionally — it is the
  probe's absence, not a call-site flag, that makes it free.

Usage
-----
::

    from piecewise_linear_attention.core.profiling import PhaseProbe, profile

    probe = PhaseProbe(device)
    with profile(probe):
        logits = model(input_ids, padding_mask)   # forward records its phases
        loss = criterion(logits, labels)
    loss.backward()
    probe.finalize()          # sync + convert events to milliseconds
    print(probe.totals_ms())  # {"anchor_clustering": ..., "second_moment_einsum": ...}
"""

import contextlib
import time
from typing import Dict, List, Optional

import torch

# The active probe, or None when profiling is disabled. Read on every instrumented
# forward; kept module-global (not thread-local) because the training step this is
# built for is single-threaded on one device — a deliberate simplicity choice.
_active: Optional["PhaseProbe"] = None


class PhaseProbe:
    """Accumulates per-phase elapsed time over one or more forward passes.

    On CUDA, each timed section pushes a pair of ``cuda.Event``s; the real elapsed
    time is only known after a synchronize, so it is deferred to :meth:`finalize`.
    On CPU/MPS, ``perf_counter`` deltas are summed directly (no deferral needed).

    Parameters
    ----------
    device:
        The device the model runs on. Determines whether CUDA events or
        ``perf_counter`` is used, and whether :meth:`finalize` synchronizes.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        # label -> accumulated milliseconds (filled directly on CPU, in finalize on CUDA)
        self._ms: Dict[str, float] = {}
        # label -> number of timed sections (so a per-call mean can be reported)
        self._count: Dict[str, int] = {}
        # CUDA only: label -> list of (start_event, end_event) awaiting elapsed_time
        self._events: Dict[str, List] = {}
        self._finalized = False

    def _add_ms(self, label: str, ms: float) -> None:
        self._ms[label] = self._ms.get(label, 0.0) + ms

    def _bump_count(self, label: str) -> None:
        self._count[label] = self._count.get(label, 0) + 1

    @contextlib.contextmanager
    def time(self, label: str):
        """Time the wrapped block under ``label``. Prefer the module-level
        :func:`phase_timer`, which is a no-op when no probe is installed; this
        method assumes the block should always be timed."""
        self._bump_count(label)
        if self.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._events.setdefault(label, []).append((start, end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._add_ms(label, (time.perf_counter() - t0) * 1e3)

    def finalize(self) -> "PhaseProbe":
        """Resolve any pending CUDA events into milliseconds. Synchronizes once on
        CUDA so the event pairs have completed. Idempotent."""
        if self._finalized:
            return self
        if self.is_cuda and self._events:
            torch.cuda.synchronize(self.device)
            for label, pairs in self._events.items():
                total = 0.0
                for start, end in pairs:
                    total += start.elapsed_time(end)  # milliseconds
                self._add_ms(label, total)
            self._events.clear()
        self._finalized = True
        return self

    def totals_ms(self) -> Dict[str, float]:
        """Accumulated milliseconds per phase. Call :meth:`finalize` first on CUDA."""
        return dict(self._ms)

    def counts(self) -> Dict[str, int]:
        """Number of timed sections per phase (for per-call means)."""
        return dict(self._count)


@contextlib.contextmanager
def profile(probe: Optional["PhaseProbe"]):
    """Install ``probe`` as the active probe for the duration of the block.

    Nesting restores the previous probe on exit, so a diagnostic can profile an
    inner region without clobbering an outer one. Passing ``None`` explicitly
    disables profiling for the block (useful to exclude a warmup forward).
    """
    global _active
    previous = _active
    _active = probe
    try:
        yield probe
    finally:
        _active = previous


@contextlib.contextmanager
def phase_timer(label: str):
    """Time the wrapped block under ``label`` **iff** a probe is installed.

    This is the call-site entry point used inside the model. When no probe is
    active (the default in training/inference) it yields immediately with no timing
    work — a single global load and a branch — so instrumented code paths carry no
    cost in normal use.
    """
    probe = _active
    if probe is None:
        yield
        return
    with probe.time(label):
        yield


def active_probe() -> Optional["PhaseProbe"]:
    """The currently installed probe, or ``None``. Exposed for tests/introspection."""
    return _active
