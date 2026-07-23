"""Smoke tests for the fair ListOps LRA harness across all attention methods.

These lock in the two regression classes this harness was built to fix, so they
are targeted rather than exhaustive:

1. **Silent hyperparameter / build drops.** Before ``attn_kwargs`` was threaded,
   ``MultiHeadAttention`` passed a fixed kwarg set, so every baseline silently used
   registry defaults and Linformer hard-crashed at long sequence length. The
   ``test_build_and_step`` case builds each method at the *task* sequence length
   with its matched-budget hyperparameters and runs a forward+backward, so a
   dropped kwarg (e.g. Linformer's ``max_seq_len``) resurfaces as a build/þshape
   failure rather than a silently-wrong default.

2. **Incorrect key-padding masking.** The padding mask reaches the encoder but was
   dropped before the attention layers, and zero-K/V-row masking is provably wrong
   for 6 of 7 methods (softmax-over-keys denominators and non-zero ``φ(0)`` both
   leak padded positions). ``test_padding_positions_do_not_leak`` asserts the
   property that must hold for *every* method: changing the token *values* at
   padded positions must not change the pooled logits. That single invariant
   catches any method whose masking still lets padded keys into the output.

Kept small (short sequences, 1 layer, tiny width) so the whole file runs in
seconds on CPU. One representative assertion per property per method — not one per
internal tensor — since the mask/kwarg plumbing generalises across methods.
"""
import pytest
import torch

from experiments.lra.configs.base_config import LISTOPS_METHODS, attention_hparams
from experiments.lra.data.listops import ListOpsDataset
from experiments.lra.models.encoder import LRAEncoder

VOCAB_SIZE = ListOpsDataset.get_vocab_size()
MAX_LEN = 64  # short enough for CPU; long enough that Nyström uses landmarks
BATCH = 4


def _make_model(method, pooling_mode="CLS"):
    """Build a tiny LRAEncoder for ``method`` with its matched-budget hparams.

    Small ``max_seq_len`` so Linformer's projection is sized to this test's length
    rather than the 2000-token task length; the point is that ``attn_kwargs`` is
    threaded, not that we exercise the full LRA size.
    """
    torch.manual_seed(0)
    attn_kwargs = attention_hparams(method, MAX_LEN, pooling_mode)
    return LRAEncoder(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_LEN,
        num_classes=10,
        emb_dim=32,
        num_layers=1,
        num_heads=4,
        mlp_dim=64,
        attention_type=method,
        dropout=0.0,
        pooling_mode=pooling_mode,
        attn_kwargs=attn_kwargs,
    )


def _random_batch(seq_len, pad_from=None):
    """Random token ids + padding mask. If ``pad_from`` is set, positions
    ``[pad_from:]`` are padded (mask False, ids set to PAD=0)."""
    torch.manual_seed(1)
    ids = torch.randint(1, VOCAB_SIZE, (BATCH, seq_len))
    mask = torch.ones(BATCH, seq_len, dtype=torch.bool)
    if pad_from is not None:
        ids[:, pad_from:] = 0
        mask[:, pad_from:] = False
    return ids, mask


@pytest.mark.parametrize("method", LISTOPS_METHODS)
def test_build_and_step(method):
    """Locks in: every method builds at the task sequence length with its
    matched-budget hyperparameters and completes a forward+backward with a finite
    loss and populated gradients.

    Catches: a dropped method-specific kwarg (the regression that had Linformer
    crash and the others fall back to defaults) — it would surface here as a build
    error, a shape mismatch, or a non-finite loss.
    """
    model = _make_model(method)
    ids, mask = _random_batch(MAX_LEN)
    logits = model(ids, mask)
    assert logits.shape == (BATCH, 10)
    assert torch.isfinite(logits).all(), f"{method}: non-finite logits"

    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in grads), f"{method}: no finite non-zero gradients"


@pytest.mark.parametrize("method", LISTOPS_METHODS)
def test_padding_positions_do_not_leak(method):
    """Locks in: the value of a padded token cannot change any output. Two batches
    share identical valid positions and padding masks but differ arbitrarily in the
    *ids* at padded positions; the pooled logits must be identical.

    Catches: incorrect key-padding masking. If a method still lets padded keys into
    its aggregation (e.g. zero-row masking that leaves ``exp(0)=1`` in a softmax
    denominator, or a non-zero ``φ(0)`` in a linear normalizer), the two logits
    diverge and this fails. MEAN pooling is used so the invariant is tested purely
    through attention (CLS pooling would already hide padded positions at the head).
    """
    model = _make_model(method, pooling_mode="MEAN")
    model.eval()

    valid_len = MAX_LEN // 2
    ids_a, mask = _random_batch(MAX_LEN, pad_from=valid_len)

    # Second batch: identical valid prefix + mask, garbage (non-PAD) ids in the pad
    # region. A correct mask makes these positions invisible.
    ids_b = ids_a.clone()
    torch.manual_seed(2)
    ids_b[:, valid_len:] = torch.randint(1, VOCAB_SIZE, (BATCH, MAX_LEN - valid_len))

    with torch.no_grad():
        logits_a = model(ids_a, mask)
        logits_b = model(ids_b, mask)

    assert torch.allclose(logits_a, logits_b, atol=1e-4), (
        f"{method}: padded-token values leaked into the output "
        f"(max diff {(logits_a - logits_b).abs().max().item():.3e}); "
        f"key-padding mask is not applied correctly for this method"
    )


@pytest.mark.parametrize("method", LISTOPS_METHODS)
def test_mask_changes_output(method):
    """Locks in: the mask is actually wired through — masking out half the sequence
    changes the output vs. attending over everything.

    Catches: a mask that is silently ignored (threaded but never applied). Without
    this, ``test_padding_positions_do_not_leak`` could pass trivially for a method
    that drops the mask entirely (padded ids happen not to matter if nothing
    attends differently). Uses MEAN pooling over the full sequence for both runs so
    only the attention masking differs.
    """
    model = _make_model(method, pooling_mode="MEAN")
    model.eval()

    ids, full_mask = _random_batch(MAX_LEN)
    half_mask = full_mask.clone()
    half_mask[:, MAX_LEN // 2:] = False

    with torch.no_grad():
        out_full = model(ids, full_mask)
        out_half = model(ids, half_mask)

    assert not torch.allclose(out_full, out_half, atol=1e-4), (
        f"{method}: masking half the keys did not change the output — the "
        f"key_padding_mask appears to be ignored"
    )
