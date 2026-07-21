"""Tests for the attention registry."""

import pytest
import torch

from piecewise_linear_attention.core.attention import (
    LinearAttention,
    PiecewiseAttention,
    StandardAttention,
)
from piecewise_linear_attention.core.registry import (
    available_attention_types,
    build_attention,
    register_attention,
)


class TestBuildAttention:
    """Construction dispatch."""

    @pytest.mark.parametrize(
        "name,expected_cls",
        [
            ("standard", StandardAttention),
            ("linear", LinearAttention),
            ("piecewise", PiecewiseAttention),
        ],
    )
    def test_builds_expected_class(self, name, expected_cls):
        """Each built-in name maps to its class.

        Locks: the built-in registrations. Catches a registration that is
        dropped or wired to the wrong class.
        """
        attn = build_attention(name, dim=16, dropout=0.0, causal=False)
        assert isinstance(attn, expected_cls)

    def test_unknown_name_lists_available(self):
        """Unknown names raise ValueError naming the valid options.

        Locks: the discoverability contract. Catches a silent fallthrough that
        would return None or a default mechanism for a typo'd name.
        """
        with pytest.raises(ValueError) as excinfo:
            build_attention("does_not_exist", dim=16)
        msg = str(excinfo.value)
        for name in ("standard", "linear", "piecewise"):
            assert name in msg

    def test_threads_linear_kwargs(self):
        """kernel_type and eps reach LinearAttention.

        Locks: the previously-dropped kwargs are now routed. Catches a builder
        that ignores mechanism-specific options.
        """
        attn = build_attention("linear", dim=16, kernel_type="relu", eps=1e-4, causal=False)
        assert attn.kernel == "relu"
        assert attn.eps == 1e-4

    def test_threads_piecewise_kwargs(self):
        """causal_pseudo_query reaches PiecewiseAttention.

        Locks: routing of the causal anchor strategy. Catches a builder that
        drops the option and silently uses the default.
        """
        attn = build_attention("piecewise", dim=16, causal=True, causal_pseudo_query="first")
        assert attn.causal_pseudo_query == "first"

    def test_extra_kwargs_are_ignored_per_mechanism(self):
        """Irrelevant kwargs do not break construction.

        Locks: the shared call convention where the multi-head wrapper passes
        the union of all mechanisms' options. Catches a builder that fails on an
        option meant for a different mechanism.
        """
        # kernel_type is meaningless for standard attention; must be ignored.
        attn = build_attention("standard", dim=16, kernel_type="relu")
        assert isinstance(attn, StandardAttention)


class TestRegistration:
    """Registering new mechanisms."""

    def test_available_types_includes_builtins(self):
        """The built-ins are discoverable and sorted.

        Locks: available_attention_types() reflects the registry. Catches a
        registry that is not populated at import time.
        """
        types = available_attention_types()
        assert types == sorted(types)
        for name in ("standard", "linear", "piecewise"):
            assert name in types

    def test_duplicate_registration_raises(self):
        """Re-registering an existing name is an error.

        Locks: registry integrity. Catches accidental shadowing of a built-in.
        """
        with pytest.raises(ValueError):

            @register_attention("standard")
            def _dup(**kwargs):  # pragma: no cover - should never run
                raise AssertionError

    def test_custom_registration_roundtrip(self):
        """A freshly registered builder is usable and then cleanable.

        Locks: the register -> build path for third-party mechanisms. Catches a
        decorator that registers under the wrong key or does not return the
        builder. The registry entry is removed afterward to avoid leaking state
        into other tests.
        """
        from piecewise_linear_attention.core.registry import ATTENTION_REGISTRY

        name = "_test_custom_mechanism"

        @register_attention(name)
        def _build(*, dim, dropout, causal, **kwargs):
            return StandardAttention(dim=dim, dropout=dropout, causal=causal)

        try:
            attn = build_attention(name, dim=8)
            assert isinstance(attn, StandardAttention)
            assert name in available_attention_types()
        finally:
            ATTENTION_REGISTRY.pop(name, None)
