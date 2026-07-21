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
        -> result schema). Catches a break in the timing loop, the device-sync
        helper, or result serialization.
        """
        module = _load("_run_scaling_cli", "run_scaling.py")
        out = tmp_path / "scaling.json"
        result = module.main(["--smoke", "--out", str(out)])
        assert out.exists()
        assert result.history["grid"], "no measurements recorded"
