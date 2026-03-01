"""Tests for attention mechanisms."""

import math

import pytest
import torch
import torch.nn.functional as F

from piecewise_linear_attention.core.attention import (
    StandardAttention,
    LinearAttention,
)


class TestStandardAttention:
    """Tests for StandardAttention."""

    @pytest.fixture
    def attention(self):
        """Create a standard attention module."""
        return StandardAttention(dim=64, dropout=0.0, scale=True)

    @pytest.fixture
    def sample_inputs(self):
        """Create sample Q, K, V tensors."""
        batch_size = 2
        seq_len = 10
        dim = 64

        Q = torch.randn(batch_size, seq_len, dim)
        K = torch.randn(batch_size, seq_len, dim)
        V = torch.randn(batch_size, seq_len, dim)

        return Q, K, V

    def test_output_shape(self, attention, sample_inputs):
        """Test that output has correct shape."""
        Q, K, V = sample_inputs
        output, _ = attention(Q, K, V)

        assert output.shape == Q.shape, f"Expected {Q.shape}, got {output.shape}"

    def test_output_dtype(self, attention, sample_inputs):
        """Test that output has same dtype as input."""
        Q, K, V = sample_inputs
        output, _ = attention(Q, K, V)

        assert output.dtype == Q.dtype

    def test_return_attention_weights(self, attention, sample_inputs):
        """Test that attention weights are returned."""
        Q, K, V = sample_inputs
        batch_size, seq_len, _ = Q.shape

        output, weights = attention(Q, K, V)

        assert weights is not None, "Attention weights should be returned"
        assert weights.shape == (batch_size, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self, attention, sample_inputs):
        """Test that attention weights sum to 1 across key dimension."""
        Q, K, V = sample_inputs
        _, weights = attention(Q, K, V)

        # Sum over the key dimension (last dimension)
        weight_sums = weights.sum(dim=-1)

        # Should sum to 1 (allowing small numerical error)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)

    def test_attention_weights_positive(self, attention, sample_inputs):
        """Test that attention weights are non-negative."""
        Q, K, V = sample_inputs
        _, weights = attention(Q, K, V)

        assert (weights >= 0).all(), "Attention weights should be non-negative"

    def test_scaling(self, sample_inputs):
        """Test that scaling affects the attention sharpness."""
        Q, K, V = sample_inputs

        attn_scaled = StandardAttention(dim=64, scale=True)
        attn_unscaled = StandardAttention(dim=64, scale=False)

        _, weights_scaled = attn_scaled(Q, K, V)
        _, weights_unscaled = attn_unscaled(Q, K, V)

        # Scaled attention should have different (typically less peaked) distribution
        assert not torch.allclose(weights_scaled, weights_unscaled)

    def test_non_causal_attention(self, attention):
        """Test bidirectional (non-causal) attention can attend to all positions."""
        batch_size = 2
        seq_len = 5
        dim = 64

        Q = torch.randn(batch_size, seq_len, dim)
        K = torch.randn(batch_size, seq_len, dim)
        V = torch.randn(batch_size, seq_len, dim)

        # Non-causal attention (default)
        _, weights = attention(Q, K, V)

        # Check that all positions have non-zero attention (no masking)
        # First query should attend to all keys including future ones
        assert (weights[0, 0, :] > 0).all(), "Non-causal attention should attend to all positions"

    def test_causal_mask(self):
        """Test causal masking (prevents attending to future positions)."""
        batch_size = 1
        seq_len = 5
        dim = 64

        Q = torch.randn(batch_size, seq_len, dim)
        K = torch.randn(batch_size, seq_len, dim)
        V = torch.randn(batch_size, seq_len, dim)

        # Use causal parameter
        attention_causal = StandardAttention(dim=dim, scale=True, causal=True)
        _, weights = attention_causal(Q, K, V)

        # Check that upper triangle (future positions) have zero attention
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.isclose(
                    weights[0, i, j],
                    torch.tensor(0.0),
                    atol=1e-6,
                ), f"Position ({i}, {j}) should be masked"

    def test_self_attention(self, attention):
        """Test self-attention where Q=K=V."""
        batch_size = 2
        seq_len = 8
        dim = 64

        X = torch.randn(batch_size, seq_len, dim)
        output, weights = attention(X, X, X)

        # Weights should be symmetric-ish for self-attention with same Q, K
        # But not exactly due to softmax
        assert output.shape == X.shape
        assert weights.shape == (batch_size, seq_len, seq_len)

    def test_single_token_attention(self, attention):
        """Test attention with single token sequence."""
        batch_size = 2
        seq_len = 1
        dim = 64

        Q = torch.randn(batch_size, seq_len, dim)
        K = torch.randn(batch_size, seq_len, dim)
        V = torch.randn(batch_size, seq_len, dim)

        output, weights = attention(Q, K, V)

        # With single token, attention weight should be 1.0
        assert torch.allclose(weights, torch.ones_like(weights))
        # Output should equal V (since attention weight is 1.0)
        assert torch.allclose(output, V)

    def test_gradient_flow(self, attention, sample_inputs):
        """Test that gradients flow through attention."""
        Q, K, V = sample_inputs
        Q.requires_grad = True
        K.requires_grad = True
        V.requires_grad = True

        output, _ = attention(Q, K, V)
        loss = output.sum()
        loss.backward()

        assert Q.grad is not None, "Gradient should flow to Q"
        assert K.grad is not None, "Gradient should flow to K"
        assert V.grad is not None, "Gradient should flow to V"

    def test_batch_independence(self, attention):
        """Test that different batch elements are computed independently."""
        batch_size = 2
        seq_len = 5
        dim = 64

        Q = torch.randn(batch_size, seq_len, dim)
        K = torch.randn(batch_size, seq_len, dim)
        V = torch.randn(batch_size, seq_len, dim)

        # Compute batched
        output_batched, _ = attention(Q, K, V)

        # Compute individually
        output_0, _ = attention(Q[0:1], K[0:1], V[0:1])
        output_1, _ = attention(Q[1:2], K[1:2], V[1:2])

        assert torch.allclose(output_batched[0], output_0[0], atol=1e-6)
        assert torch.allclose(output_batched[1], output_1[0], atol=1e-6)

    def test_deterministic(self, attention, sample_inputs):
        """Test that attention is deterministic (with dropout=0)."""
        Q, K, V = sample_inputs

        output1, _ = attention(Q, K, V)
        output2, _ = attention(Q, K, V)

        assert torch.allclose(output1, output2)


class TestLinearAttention:
    """Tests for LinearAttention."""

    @pytest.fixture
    def attention(self):
        """Create a linear attention module."""
        return LinearAttention(dim=64, dropout=0.0)

    @pytest.fixture
    def sample_inputs(self):
        """Create sample Q, K, V tensors."""
        batch_size = 2
        seq_len = 10
        dim = 64

        Q = torch.randn(batch_size, seq_len, dim)
        K = torch.randn(batch_size, seq_len, dim)
        V = torch.randn(batch_size, seq_len, dim)

        return Q, K, V

    def test_output_shape(self, attention, sample_inputs):
        """Test that output has correct shape."""
        Q, K, V = sample_inputs
        output, _ = attention(Q, K, V)

        assert output.shape == Q.shape

    def test_linear_complexity(self, attention):
        """Test that linear attention scales linearly with sequence length."""
        # This is more of a performance test, but we can check it runs
        batch_size = 1
        dim = 64

        # Should handle long sequences efficiently
        seq_len = 1000
        Q = torch.randn(batch_size, seq_len, dim)
        K = torch.randn(batch_size, seq_len, dim)
        V = torch.randn(batch_size, seq_len, dim)

        output, _ = attention(Q, K, V)
        assert output.shape == Q.shape

    def test_different_from_standard(self, sample_inputs):
        """Test that linear attention gives different results than standard attention."""
        Q, K, V = sample_inputs

        standard_attn = StandardAttention(dim=64, dropout=0.0)
        linear_attn = LinearAttention(dim=64, dropout=0.0)

        output_standard, _ = standard_attn(Q, K, V)
        output_linear, _ = linear_attn(Q, K, V)

        # Should give different results (linear doesn't use softmax)
        assert not torch.allclose(output_standard, output_linear)

    def test_causal_mode(self):
        """Test that LinearAttention supports causal mode."""
        batch_size = 2
        seq_len = 10
        dim = 64

        Q = torch.randn(batch_size, seq_len, dim)
        K = torch.randn(batch_size, seq_len, dim)
        V = torch.randn(batch_size, seq_len, dim)

        # Test non-causal
        attn_noncausal = LinearAttention(dim=dim, causal=False)
        output_noncausal, _ = attn_noncausal(Q, K, V)
        assert output_noncausal.shape == (batch_size, seq_len, dim)

        # Test causal
        attn_causal = LinearAttention(dim=dim, causal=True)
        output_causal, _ = attn_causal(Q, K, V)
        assert output_causal.shape == (batch_size, seq_len, dim)

        # Outputs should be different
        assert not torch.allclose(output_noncausal, output_causal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
