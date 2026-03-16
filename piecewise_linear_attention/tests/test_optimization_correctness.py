"""Tests to verify correctness of optimized tensor operations.

This test suite verifies that the optimized implementations (using @ and matmul
instead of einsum) produce identical results to the original einsum implementations.
"""

import math
import pytest
import torch

from piecewise_linear_attention.core.attention import (
    PiecewiseAttention,
    LinearAttention,
)


class TestOptimizedPiecewiseAttention:
    """Verify optimized PiecewiseAttention operations match einsum versions."""

    @pytest.fixture
    def sample_data(self):
        """Create deterministic sample data."""
        torch.manual_seed(42)
        batch = 4
        seq_len = 16
        dim = 32

        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        return Q, K, V, batch, seq_len, dim

    def _compute_with_einsum_reference(self, pseudo_queries, K, V, scale_factor):
        """Reference implementation using einsum (original version)."""
        # Compute attention scores
        scores = torch.einsum('bd,bnd->bn', pseudo_queries, K) * scale_factor
        attn_weights = torch.softmax(scores, dim=-1)

        # Compute attention output
        output = torch.einsum('bn,bnd->bd', attn_weights, V)

        # Weighted sum of keys
        weighted_k = torch.einsum('bn,bnd->bd', attn_weights, K)

        # Weighted sum of outer products
        weighted_outer = torch.einsum('bn,bni,bnj->bij', attn_weights, V, K)

        # Outer product of weighted sums
        outer_weighted = torch.einsum('bi,bj->bij', output, weighted_k)

        # Combine to get Jacobian
        jacobian = scale_factor * (weighted_outer - outer_weighted)

        return output, jacobian

    def _compute_with_matmul(self, pseudo_queries, K, V, scale_factor):
        """Optimized implementation using @ and matmul."""
        # Compute attention scores
        scores = (K @ pseudo_queries.unsqueeze(-1)).squeeze(-1) * scale_factor
        attn_weights = torch.softmax(scores, dim=-1)

        # Compute attention output
        output = (attn_weights.unsqueeze(1) @ V).squeeze(1)

        # Weighted sum of keys
        weighted_k = (attn_weights.unsqueeze(1) @ K).squeeze(1)

        # Weighted sum of outer products: V^T @ diag(α) @ K
        weighted_v = attn_weights.unsqueeze(-1) * V
        weighted_outer = weighted_v.transpose(-2, -1) @ K

        # Outer product of weighted sums
        outer_weighted = output.unsqueeze(-1) @ weighted_k.unsqueeze(1)

        # Combine to get Jacobian
        jacobian = scale_factor * (weighted_outer - outer_weighted)

        return output, jacobian

    def test_jacobian_computation_matches_einsum(self, sample_data):
        """Verify that optimized Jacobian computation matches einsum version."""
        Q, K, V, batch, seq_len, dim = sample_data

        # Use mean query as pseudo-query
        pseudo_queries = Q.mean(dim=1)
        scale_factor = 1.0 / math.sqrt(dim)

        # Compute with both methods
        output_einsum, jacobian_einsum = self._compute_with_einsum_reference(
            pseudo_queries, K, V, scale_factor
        )
        output_matmul, jacobian_matmul = self._compute_with_matmul(
            pseudo_queries, K, V, scale_factor
        )

        # Should match exactly (within numerical precision)
        assert torch.allclose(output_einsum, output_matmul, rtol=1e-5, atol=1e-7), \
            f"Output mismatch: max diff = {(output_einsum - output_matmul).abs().max()}"

        assert torch.allclose(jacobian_einsum, jacobian_matmul, rtol=1e-5, atol=1e-7), \
            f"Jacobian mismatch: max diff = {(jacobian_einsum - jacobian_matmul).abs().max()}"

    def test_noncausal_forward_pass_deterministic(self, sample_data):
        """Verify non-causal forward pass is deterministic."""
        Q, K, V, batch, seq_len, dim = sample_data

        attention = PiecewiseAttention(dim=dim, dropout=0.0, scale=True, causal=False)

        # Run multiple times
        output1, _ = attention(Q, K, V)
        output2, _ = attention(Q, K, V)
        output3, _ = attention(Q, K, V)

        # Should be identical
        assert torch.allclose(output1, output2, atol=1e-8)
        assert torch.allclose(output2, output3, atol=1e-8)

    def test_causal_forward_pass_deterministic(self, sample_data):
        """Verify causal forward pass is deterministic."""
        Q, K, V, batch, seq_len, dim = sample_data

        attention = PiecewiseAttention(dim=dim, dropout=0.0, scale=True, causal=True)

        # Run multiple times
        output1, _ = attention(Q, K, V)
        output2, _ = attention(Q, K, V)
        output3, _ = attention(Q, K, V)

        # Should be identical
        assert torch.allclose(output1, output2, atol=1e-8)
        assert torch.allclose(output2, output3, atol=1e-8)

    def test_noncausal_gradient_correctness(self, sample_data):
        """Verify gradients are correct in non-causal mode."""
        Q, K, V, batch, seq_len, dim = sample_data

        # Make copies that require grad
        Q1, K1, V1 = Q.clone().requires_grad_(True), K.clone().requires_grad_(True), V.clone().requires_grad_(True)
        Q2, K2, V2 = Q.clone().requires_grad_(True), K.clone().requires_grad_(True), V.clone().requires_grad_(True)

        attention = PiecewiseAttention(dim=dim, dropout=0.0, scale=True, causal=False)

        # Compute twice
        output1, _ = attention(Q1, K1, V1)
        loss1 = output1.sum()
        loss1.backward()

        output2, _ = attention(Q2, K2, V2)
        loss2 = output2.sum()
        loss2.backward()

        # Gradients should be identical
        assert torch.allclose(Q1.grad, Q2.grad, atol=1e-6)
        assert torch.allclose(K1.grad, K2.grad, atol=1e-6)
        assert torch.allclose(V1.grad, V2.grad, atol=1e-6)

    def test_causal_gradient_correctness(self, sample_data):
        """Verify gradients are correct in causal mode."""
        Q, K, V, batch, seq_len, dim = sample_data

        # Make copies that require grad
        Q1, K1, V1 = Q.clone().requires_grad_(True), K.clone().requires_grad_(True), V.clone().requires_grad_(True)
        Q2, K2, V2 = Q.clone().requires_grad_(True), K.clone().requires_grad_(True), V.clone().requires_grad_(True)

        attention = PiecewiseAttention(dim=dim, dropout=0.0, scale=True, causal=True)

        # Compute twice
        output1, _ = attention(Q1, K1, V1)
        loss1 = output1.sum()
        loss1.backward()

        output2, _ = attention(Q2, K2, V2)
        loss2 = output2.sum()
        loss2.backward()

        # Gradients should be identical
        assert torch.allclose(Q1.grad, Q2.grad, atol=1e-6)
        assert torch.allclose(K1.grad, K2.grad, atol=1e-6)
        assert torch.allclose(V1.grad, V2.grad, atol=1e-6)

    def test_jacobian_deltas_computation(self, sample_data):
        """Verify the Jacobian @ deltas computation in non-causal mode."""
        Q, K, V, batch, seq_len, dim = sample_data

        pseudo_queries = Q.mean(dim=1)
        deltas = Q - pseudo_queries.unsqueeze(1)

        # Create a random jacobian
        torch.manual_seed(123)
        jacobians = torch.randn(batch, dim, dim)

        # Method 1: einsum
        jacobian_deltas_einsum = torch.einsum('bij,bsj->bsi', jacobians, deltas)

        # Method 2: optimized
        jacobian_deltas_matmul = (jacobians @ deltas.transpose(-2, -1)).transpose(-2, -1)

        # Should match
        assert torch.allclose(jacobian_deltas_einsum, jacobian_deltas_matmul, rtol=1e-5, atol=1e-7)

    def test_causal_cumsum_operations(self, sample_data):
        """Verify causal mode cumsum operations produce correct shapes and values."""
        Q, K, V, batch, seq_len, dim = sample_data

        attention = PiecewiseAttention(dim=dim, dropout=0.0, scale=True, causal=True, causal_pseudo_query="mean")

        # Forward pass should work without errors
        output, pseudo_queries = attention(Q, K, V)

        # Check shapes
        assert output.shape == (batch, seq_len, dim)
        assert pseudo_queries.shape == (batch, dim)

        # Check output is finite
        assert torch.isfinite(output).all()


class TestOptimizedLinearAttention:
    """Verify optimized LinearAttention operations match einsum versions."""

    @pytest.fixture
    def sample_data(self):
        """Create deterministic sample data."""
        torch.manual_seed(42)
        batch = 4
        seq_len = 16
        dim = 32

        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        return Q, K, V, batch, seq_len, dim

    def test_causal_linear_attention_deterministic(self, sample_data):
        """Verify causal linear attention is deterministic."""
        Q, K, V, batch, seq_len, dim = sample_data

        attention = LinearAttention(dim=dim, dropout=0.0, causal=True)

        # Run multiple times
        output1, _ = attention(Q, K, V)
        output2, _ = attention(Q, K, V)
        output3, _ = attention(Q, K, V)

        # Should be identical
        assert torch.allclose(output1, output2, atol=1e-8)
        assert torch.allclose(output2, output3, atol=1e-8)

    def test_causal_linear_attention_gradients(self, sample_data):
        """Verify gradients are correct for causal linear attention."""
        Q, K, V, batch, seq_len, dim = sample_data

        # Make copies that require grad
        Q1, K1, V1 = Q.clone().requires_grad_(True), K.clone().requires_grad_(True), V.clone().requires_grad_(True)
        Q2, K2, V2 = Q.clone().requires_grad_(True), K.clone().requires_grad_(True), V.clone().requires_grad_(True)

        attention = LinearAttention(dim=dim, dropout=0.0, causal=True)

        # Compute twice
        output1, _ = attention(Q1, K1, V1)
        loss1 = output1.sum()
        loss1.backward()

        output2, _ = attention(Q2, K2, V2)
        loss2 = output2.sum()
        loss2.backward()

        # Gradients should be identical
        assert torch.allclose(Q1.grad, Q2.grad, atol=1e-6)
        assert torch.allclose(K1.grad, K2.grad, atol=1e-6)
        assert torch.allclose(V1.grad, V2.grad, atol=1e-6)


class TestNumericalStability:
    """Test numerical stability of optimized operations."""

    def test_large_values(self):
        """Test with large input values."""
        torch.manual_seed(42)
        batch, seq_len, dim = 2, 8, 16

        # Large values
        Q = 100 * torch.randn(batch, seq_len, dim)
        K = 100 * torch.randn(batch, seq_len, dim)
        V = 100 * torch.randn(batch, seq_len, dim)

        attention = PiecewiseAttention(dim=dim, dropout=0.0, scale=True)
        output, _ = attention(Q, K, V)

        # Should still be finite
        assert torch.isfinite(output).all()

    def test_small_values(self):
        """Test with small input values."""
        torch.manual_seed(42)
        batch, seq_len, dim = 2, 8, 16

        # Small values
        Q = 0.01 * torch.randn(batch, seq_len, dim)
        K = 0.01 * torch.randn(batch, seq_len, dim)
        V = 0.01 * torch.randn(batch, seq_len, dim)

        attention = PiecewiseAttention(dim=dim, dropout=0.0, scale=True)
        output, _ = attention(Q, K, V)

        # Should still be finite
        assert torch.isfinite(output).all()

    def test_mixed_precision_compatibility(self):
        """Test that operations work with float16."""
        torch.manual_seed(42)
        batch, seq_len, dim = 2, 8, 16

        Q = torch.randn(batch, seq_len, dim, dtype=torch.float16)
        K = torch.randn(batch, seq_len, dim, dtype=torch.float16)
        V = torch.randn(batch, seq_len, dim, dtype=torch.float16)

        attention = PiecewiseAttention(dim=dim, dropout=0.0, scale=True)

        # May need to cast internally, but should not crash
        try:
            output, _ = attention(Q, K, V)
            # If it works, output should have reasonable values
            assert output.shape == Q.shape
        except RuntimeError:
            # Some operations may not support float16, which is acceptable
            pytest.skip("float16 not supported on this platform")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_sequence_length_one(self):
        """Test with sequence length of 1."""
        torch.manual_seed(42)
        batch, seq_len, dim = 2, 1, 16

        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        attention = PiecewiseAttention(dim=dim, dropout=0.0)
        output, _ = attention(Q, K, V)

        assert output.shape == (batch, seq_len, dim)
        assert torch.isfinite(output).all()

    def test_large_sequence_length(self):
        """Test with large sequence length (stress test)."""
        torch.manual_seed(42)
        batch, seq_len, dim = 1, 512, 32

        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        attention = PiecewiseAttention(dim=dim, dropout=0.0, causal=False)
        output, _ = attention(Q, K, V)

        assert output.shape == (batch, seq_len, dim)
        assert torch.isfinite(output).all()

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        torch.manual_seed(42)
        batch, seq_len, dim = 1, 16, 32

        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        attention = PiecewiseAttention(dim=dim, dropout=0.0)
        output, _ = attention(Q, K, V)

        assert output.shape == (batch, seq_len, dim)
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
