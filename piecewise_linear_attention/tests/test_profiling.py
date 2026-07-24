"""Tests for the opt-in phase-timing facility used by the epoch-warmup diagnostic.

The profiler is on the attention hot path (``phase_timer`` is called every
multi-anchor forward), so the invariant that matters most is that it is a true
no-op when nothing is profiling. These also lock that an installed probe records
the two phases the diagnostic attributes cost to, and that ``profile`` nesting
restores the previous probe.
"""

import torch

from piecewise_linear_attention.core.profiling import (
    PhaseProbe,
    active_probe,
    phase_timer,
    profile,
)
from piecewise_linear_attention.core.registry import build_attention


class TestPhaseProfiling:
    def test_disabled_is_noop(self):
        """With no probe installed, ``phase_timer`` records nothing and leaves the
        active probe untouched.

        Locks: the disabled path carries no timing state — the reason the
        instrumented forward is free in normal training/inference. Catches a
        regression that installs a default probe or leaks state when profiling is
        off.
        """
        assert active_probe() is None
        with phase_timer("anything"):
            pass
        assert active_probe() is None

    def test_records_forward_phases(self):
        """An installed probe records both instrumented phases of a multi-anchor
        forward.

        Locks: the native call sites (KMeansAnchor clustering, second-moment
        einsum) push into the active probe under the labels the diagnostic reads.
        Catches a rename of either phase or a call site that stops timing.
        """
        attn = build_attention(
            "piecewise_kmeans", dim=16, dropout=0.0, causal=False, num_anchors=4
        )
        q = torch.randn(2, 32, 16)
        probe = PhaseProbe(torch.device("cpu"))
        with profile(probe):
            out = attn(q, q, q)
            (out[0] if isinstance(out, tuple) else out).sum()
        probe.finalize()
        totals = probe.totals_ms()
        assert "anchor_clustering" in totals and totals["anchor_clustering"] > 0.0
        assert "second_moment_einsum" in totals and totals["second_moment_einsum"] > 0.0
        assert probe.counts()["anchor_clustering"] == 1

    def test_profile_nesting_restores_previous(self):
        """Exiting a ``profile`` block restores the probe that was active on entry.

        Locks: a diagnostic can profile an inner region without clobbering an
        outer probe (and the disabled state is restored after the outermost
        block). Catches a context manager that unconditionally clears the active
        probe on exit.
        """
        outer = PhaseProbe(torch.device("cpu"))
        inner = PhaseProbe(torch.device("cpu"))
        assert active_probe() is None
        with profile(outer):
            assert active_probe() is outer
            with profile(inner):
                assert active_probe() is inner
            assert active_probe() is outer
        assert active_probe() is None
