"""Tests for the benchmark harness core (config, device, tracker, results)."""

import torch

from piecewise_linear_attention.harness import (
    ConsoleTracker,
    ExperimentConfig,
    ExperimentResult,
    NoOpTracker,
    get_device,
    load_result,
    save_result,
    set_seed,
)
from piecewise_linear_attention.harness.config import CONFIG_SCHEMA_VERSION
from piecewise_linear_attention.harness.results import RESULT_SCHEMA_VERSION


class TestConfig:
    """ExperimentConfig serialization and provenance."""

    def test_roundtrip(self):
        """to_dict/from_dict is lossless for set fields.

        Locks: config serialization. Catches a field that fails to round-trip,
        which would silently corrupt logged/reloaded experiment specs.
        """
        cfg = ExperimentConfig(
            attention_type="piecewise_kmeans",
            seq_len=64,
            attention_kwargs={"num_anchors": 4},
        )
        restored = ExperimentConfig.from_dict(cfg.to_dict())
        assert restored == cfg

    def test_from_dict_ignores_unknown_keys(self):
        """Unknown keys are dropped, not fatal.

        Locks: forward compatibility when a newer schema adds fields. Catches a
        strict constructor that would break loading newer configs.
        """
        data = ExperimentConfig(attention_type="standard").to_dict()
        data["a_future_field"] = 123
        cfg = ExperimentConfig.from_dict(data)
        assert cfg.attention_type == "standard"

    def test_carries_schema_version(self):
        """Every config records the schema version.

        Locks: versioning is always present. Catches a config produced without
        provenance, which would be uninterpretable if the schema changes.
        """
        cfg = ExperimentConfig(attention_type="standard")
        assert cfg.schema_version == CONFIG_SCHEMA_VERSION


class TestDevice:
    """Device selection and seeding."""

    def test_get_device_returns_torch_device(self):
        """Auto selection returns a torch.device.

        Locks: the return type contract. Catches a string/None leak that would
        break ``.to(device)`` calls.
        """
        assert isinstance(get_device("auto"), torch.device)

    def test_explicit_cpu(self):
        """An explicit preference is honored.

        Locks: the override path. Catches auto-selection stomping an explicit
        device request.
        """
        assert get_device("cpu").type == "cpu"

    def test_set_seed_is_reproducible(self):
        """Same seed yields the same random draw.

        Locks: seeding actually seeds torch. Catches a no-op set_seed that would
        make "reproducible" runs vary.
        """
        set_seed(42)
        a = torch.randn(8)
        set_seed(42)
        b = torch.randn(8)
        assert torch.equal(a, b)


class TestTracker:
    """Tracker backends."""

    def test_noop_accepts_all_events(self):
        """NoOpTracker swallows every lifecycle call without error.

        Locks: the default tracker never interferes with a run. Catches a
        missing method that would raise when the harness logs.
        """
        tracker = NoOpTracker()
        with tracker:
            tracker.log_params({"lr": 1e-3})
            tracker.log_metrics({"loss": 0.5}, step=1)
            tracker.log_artifact("nonexistent.json")

    def test_console_context_manager(self, capsys):
        """ConsoleTracker prints start/end around a run.

        Locks: the context-manager lifecycle. Catches a broken __enter__/__exit__
        that would skip run boundaries.
        """
        with ConsoleTracker():
            pass
        out = capsys.readouterr().out
        assert "start run" in out
        assert "end run" in out


class TestResults:
    """Result schema and JSON I/O."""

    def test_save_load_roundtrip(self, tmp_path):
        """A saved result reloads identically.

        Locks: result persistence. Catches a serialization change that would
        make recorded evidence unreadable.
        """
        result = ExperimentResult(
            config={"attention_type": "standard"},
            metrics={"accuracy": 0.95, "loss": 0.04},
            environment={"device": "cpu"},
        )
        path = save_result(result, tmp_path / "sub" / "result.json")
        assert path.exists()
        assert load_result(path) == result

    def test_carries_schema_version(self):
        """Results record their schema version.

        Locks: result versioning. Catches an unversioned result that could not
        be migrated later.
        """
        result = ExperimentResult(config={}, metrics={})
        assert result.schema_version == RESULT_SCHEMA_VERSION
