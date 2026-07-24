"""Padding-leak regression tests for ``key_padding_mask``.

Locks in one cross-cutting invariant for every registered attention mechanism:
**a padded key/value (or padded query) position's *value* must not change the
output at any valid position, at a fixed sequence length.** A fair benchmark
number must be independent of what happens to sit in the padding.

Failure mode caught: a method masking on the wrong axis (or, historically, the
feature-map methods gating the mask behind ``and not self.causal`` so it was
silently dropped on the causal path) lets padded content leak into valid outputs
through a softmax denominator, a feature-map normalizer, or a causal cumsum. This
is exactly the class of bug that biases a padding-sensitive comparison, so a
registry-wide parametrized check is warranted rather than over-testing — there is
no other core-suite coverage of ``key_padding_mask``.

Not exhaustive by design: one representative leak probe per method (the invariant
generalizes across ops), not a per-shape matrix.
"""

import pytest
import torch

from piecewise_linear_attention.core.registry import build_attention

# One representative config per registered mechanism. Ranks are kept small so the
# probe is fast; the leak property does not depend on rank size.
_METHOD_KWARGS = {
    "standard": {},
    "linear": {},
    "performer": {"num_features": 32},
    "nystromformer": {"num_landmarks": 8},
    "linformer": {"k": 16, "max_seq_len": 32},
    "luna": {"num_pack": 16},
    "piecewise": {},
    "piecewise_kmeans": {"num_anchors": 4, "kmeans_iters": 3},
    "piecewise_stride": {"num_anchors": 4},
}

# Causal masking is only implemented for some mechanisms (others raise on
# ``causal=True``); the causal-path leak check runs on this subset.
_CAUSAL_CAPABLE = {"standard", "linear", "performer", "piecewise"}

_BH, _LVALID, _LPAD, _DIM = 3, 24, 8, 16  # batch*heads, valid keys, padded tail, per-head dim
_LEN = _LVALID + _LPAD


def _max_valid_drift(method, kwargs, causal):
    """Max change at valid positions when padded positions' values are scrambled.

    Builds the mechanism, then runs it twice with the padded tail (positions
    ``>= _LVALID``) filled with two different random draws for Q, K and V. The
    valid prefix of Q/K/V is byte-identical across the two runs, so any change at
    a valid output position can only come from padded content leaking in.
    """
    torch.manual_seed(1)
    attn = build_attention(method, dim=_DIM, dropout=0.0, causal=causal, **kwargs).eval()

    Q = torch.randn(_BH, _LEN, _DIM)
    K = torch.randn(_BH, _LEN, _DIM)
    V = torch.randn(_BH, _LEN, _DIM)
    mask = torch.ones(_BH, _LEN, dtype=torch.bool)
    mask[:, _LVALID:] = False  # padding is a tail suffix (True = valid)

    def run(seed):
        Qx, Kx, Vx = Q.clone(), K.clone(), V.clone()
        torch.manual_seed(seed)
        Qx[:, _LVALID:] = torch.randn(_BH, _LPAD, _DIM)
        Kx[:, _LVALID:] = torch.randn(_BH, _LPAD, _DIM)
        Vx[:, _LVALID:] = torch.randn(_BH, _LPAD, _DIM)
        with torch.no_grad():
            out, _ = attn(Qx, Kx, Vx, key_padding_mask=mask)
        return out

    out_a = run(100)
    out_b = run(999)
    return (out_a[:, :_LVALID] - out_b[:, :_LVALID]).abs().max().item()


@pytest.mark.parametrize("method", sorted(_METHOD_KWARGS))
def test_padding_value_does_not_leak_noncausal(method):
    """Padded content cannot move a valid output on the (encoder) non-causal path."""
    drift = _max_valid_drift(method, _METHOD_KWARGS[method], causal=False)
    assert drift < 1e-5, f"{method}: padded value leaked into valid outputs (drift={drift:.2e})"


@pytest.mark.parametrize("method", sorted(_CAUSAL_CAPABLE))
def test_padding_value_does_not_leak_causal(method):
    """Same invariant on the causal path.

    Regression guard for the feature-map methods (linear/performer), whose mask
    was once gated behind ``and not self.causal`` and silently dropped when
    ``causal=True`` — padded keys then leaked through the prefix-sum into every
    later valid query.
    """
    drift = _max_valid_drift(method, _METHOD_KWARGS[method], causal=True)
    assert drift < 1e-5, f"{method}: padded value leaked on causal path (drift={drift:.2e})"
