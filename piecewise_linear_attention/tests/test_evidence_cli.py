"""Smoke tests for the evidence CLIs.

These invoke the benchmark entry points in their tiny ``--smoke`` configuration
so the end-to-end pipelines (config -> model -> loop -> result schema) cannot rot
unnoticed. They assert the pipeline runs and produces a well-formed result; they
do NOT assert the scientific outcome (that requires the full, expensive run).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# The CLIs live under benchmarks/harness (glue, outside the importable package),
# so we load them by path.
_BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "harness"


def _load(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, _BENCH_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not _BENCH_DIR.exists(), reason="benchmark CLIs not present")
class TestEvidenceCLISmoke:
    def test_recall_smoke(self, tmp_path):
        """run_recall --smoke completes and writes a well-formed result.

        Locks: the trained-task pipeline wiring across all methods. Catches a
        break in config/model/loop/result integration.
        """
        module = _load("_run_recall_cli", "run_recall.py")
        out = tmp_path / "recall.json"
        result = module.main(["--smoke", "--out", str(out)])
        assert out.exists()
        assert result.history["runs"], "no runs recorded"

    def test_microbench_smoke(self, tmp_path):
        """run_microbench --smoke completes and writes a well-formed result.

        Locks: the approximation-quality pipeline. Catches a break in the
        no-training measurement path or result serialization.
        """
        module = _load("_run_microbench_cli", "run_microbench.py")
        out = tmp_path / "microbench.json"
        result = module.main(["--smoke", "--out", str(out)])
        assert out.exists()
        assert result.history["grid"], "no measurements recorded"

    def test_scaling_smoke(self, tmp_path):
        """run_scaling --smoke completes and writes a well-formed result.

        Locks: the wall-clock scaling pipeline (build_attention -> timed forward
        -> result schema), including the ``speedup_vs_performer`` head-to-head
        metric. Catches a break in the timing loop, the device-sync helper, the
        speedup computation, or result serialization.
        """
        module = _load("_run_scaling_cli", "run_scaling.py")
        out = tmp_path / "scaling.json"
        result = module.main(["--smoke", "--out", str(out)])
        assert out.exists()
        assert result.history["grid"], "no measurements recorded"
        assert result.metrics["speedup_vs_performer"], "no performer speedup recorded"

    def test_multianchor_profile_smoke(self, tmp_path):
        """run_multianchor_profile --smoke completes and writes a well-formed result.

        Locks: the op-level attribution pipeline (isolated-op timing + profiler
        self-time -> attribution metric -> result schema). Catches a break in the
        per-op decomposition or a rename of the ops the verdict cites.
        """
        module = _load("_run_multianchor_profile_cli", "run_multianchor_profile.py")
        out = tmp_path / "multianchor_profile.json"
        result = module.main(["--smoke", "--out", str(out)])
        assert out.exists()
        assert result.history["grid"], "no measurements recorded"
        assert result.metrics["attribution"], "no attribution recorded"

    def test_anchor_reuse_smoke(self, tmp_path):
        """run_anchor_reuse --smoke completes and writes a well-formed result.

        Locks: the reuse speed/accuracy pipeline (drifting-input loop -> cached
        vs fresh comparison -> result schema). Catches a break in the
        CachedAnchor wiring or the speedup/penalty summary.
        """
        module = _load("_run_anchor_reuse_cli", "run_anchor_reuse.py")
        out = tmp_path / "anchor_reuse.json"
        result = module.main(["--smoke", "--out", str(out)])
        assert out.exists()
        assert result.history["configs"], "no measurements recorded"
        assert result.metrics["summary_vs_fresh"], "no reuse summary recorded"

    def test_epoch_warmup_diag_smoke(self, tmp_path):
        """run_epoch_warmup_diag --smoke completes and writes a well-formed result.

        Locks: the epoch-warmup diagnostic pipeline (synthetic loader -> real
        train_epoch under a phase probe -> per-epoch clustering/einsum split +
        per-iteration microscope -> summary -> result schema). Catches a break in
        the native phase instrumentation wiring or a rename of the phases the
        summary cites. Skips cleanly where the LRA trainer is absent (the
        diagnostic returns None), since the smoke path drives the real model/loop.
        """
        module = _load("_run_epoch_warmup_diag_cli", "run_epoch_warmup_diag.py")
        out = tmp_path / "epoch_warmup_diag.json"
        result = module.main(["--smoke", "--out", str(out)])
        if result is None:
            pytest.skip("LRA ListOps trainer not present in this checkout")
        assert out.exists()
        assert result.metrics["summaries"], "no method summaries recorded"
        assert result.history["per_method"], "no per-method traces recorded"
