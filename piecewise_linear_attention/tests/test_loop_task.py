"""Tests for the training loop and the synthetic recall task."""

import torch

from piecewise_linear_attention.harness.loop import accuracy, train_and_evaluate
from piecewise_linear_attention.harness.tasks.synthetic_recall import (
    FIRST_CONTENT_TOKEN,
    MARKER_TOKEN,
    RecallModel,
    make_recall_batch,
)


def _batch_fn(seq_len=16, vocab_size=12, batch_size=8):
    def make(generator):
        return make_recall_batch(batch_size, seq_len, vocab_size, generator)

    return make


class TestRecallDataset:
    """The associative-recall data generator."""

    def test_determinism(self):
        """Same generator seed yields the same batch.

        Locks: batch reproducibility, which underpins comparable runs. Catches a
        generator that draws from global RNG state instead of the passed one.
        """
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        x1, y1 = make_recall_batch(8, 16, 12, g1)
        x2, y2 = make_recall_batch(8, 16, 12, g2)
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)

    def test_label_is_token_after_marker(self):
        """The label equals the token immediately after the marker.

        Locks: the task definition itself. Catches an off-by-one in marker/label
        placement that would make the task trivial or unsolvable.

        Assessment: this is the single most important task test — the entire
        scientific claim rests on this being a genuine long-range read. Not
        over-testing.
        """
        g = torch.Generator().manual_seed(0)
        x, y = make_recall_batch(16, 20, 12, g)
        for b in range(x.shape[0]):
            marker_positions = (x[b] == MARKER_TOKEN).nonzero(as_tuple=True)[0]
            assert len(marker_positions) == 1, "exactly one marker per sequence"
            pos = marker_positions.item()
            assert x[b, pos + 1] == y[b]

    def test_content_tokens_in_range(self):
        """Content tokens avoid reserved ids.

        Locks: reserved-id discipline (PAD=0, MARKER=1). Catches content
        colliding with the marker, which would corrupt the label signal.
        """
        g = torch.Generator().manual_seed(1)
        x, _ = make_recall_batch(16, 20, 12, g)
        # Every non-marker token is a valid content id.
        non_marker = x[x != MARKER_TOKEN]
        assert (non_marker >= FIRST_CONTENT_TOKEN).all()


class TestLoopSmoke:
    """The training loop runs end-to-end."""

    def test_one_step_finite_loss(self):
        """A single training step produces finite metrics on CPU.

        Locks: the loop wiring (forward adapter, loss, optimizer, eval). Catches
        a shape or device mismatch that would break any real run, without paying
        for a full-length experiment.
        """
        model = RecallModel(
            attention_type="standard",
            vocab_size=12,
            seq_len=16,
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
        )
        metrics = train_and_evaluate(
            model,
            _batch_fn(),
            steps=1,
            device=torch.device("cpu"),
            eval_batches=1,
        )
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["train_time_s"] >= 0.0

    def test_all_attention_types_run(self):
        """Every registered mechanism plugs into the model and trains a step.

        Locks: the build_attention -> RecallModel integration for each family,
        including the multi-anchor presets. Catches a mechanism whose kwargs or
        output shape are incompatible with the encoder block.
        """
        for attention_type in [
            "standard",
            "linear",
            "performer",
            "piecewise",
            "piecewise_kmeans",
            "piecewise_stride",
        ]:
            model = RecallModel(
                attention_type=attention_type,
                vocab_size=12,
                seq_len=16,
                hidden_dim=16,
                num_heads=2,
                num_layers=1,
                attention_kwargs={"num_anchors": 2} if "piecewise_" in attention_type else {},
            )
            metrics = train_and_evaluate(
                model,
                _batch_fn(),
                steps=1,
                device=torch.device("cpu"),
                eval_batches=1,
            )
            assert torch.isfinite(torch.tensor(metrics["loss"])), attention_type

    def test_eval_every_records_curve(self):
        """``eval_every`` records an accuracy-vs-step curve; default omits it.

        Locks: periodic evaluation produces one curve point per interval plus a
        final point at ``steps``, and that leaving ``eval_every`` at its default
        adds no ``"curve"`` key (backward-compatible). Catches an off-by-one in
        the eval schedule or a curve that silently changes the default return
        shape — the capability that surfaces late training transitions.

        Assessment: appropriate, not over-testing. The curve is the mechanism
        that would have caught the multi-anchor late-transition being mistaken
        for failure, so pinning its schedule and its opt-in nature has real
        regression value; a single representative attention type suffices.
        """
        model = RecallModel(
            attention_type="standard",
            vocab_size=12,
            seq_len=16,
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
        )
        metrics = train_and_evaluate(
            model,
            _batch_fn(),
            steps=4,
            device=torch.device("cpu"),
            eval_batches=1,
            eval_every=2,
        )
        curve = metrics["curve"]
        # Interval evals at steps 2 (step>0 and step%2==0) plus the final point.
        steps_seen = [point["step"] for point in curve]
        assert steps_seen == [2, 4]
        for point in curve:
            assert 0.0 <= point["accuracy"] <= 1.0
        # The final curve accuracy is exactly the reported end-of-run accuracy.
        assert curve[-1]["accuracy"] == metrics["accuracy"]

        # Default (eval_every=0) stays curve-free.
        plain = train_and_evaluate(
            model,
            _batch_fn(),
            steps=1,
            device=torch.device("cpu"),
            eval_batches=1,
        )
        assert "curve" not in plain


class TestAccuracyMetric:
    """The accuracy helper."""

    def test_perfect_and_zero(self):
        """Accuracy is 1.0 for correct preds, 0.0 for all-wrong.

        Locks: metric correctness. Catches an argmax-axis mistake.
        """
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        assert accuracy(logits, torch.tensor([1, 0])) == 1.0
        assert accuracy(logits, torch.tensor([0, 1])) == 0.0
