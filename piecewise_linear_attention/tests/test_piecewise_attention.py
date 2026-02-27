"""Tests for piecewise linear attention."""

import pytest
import torch

from piecewise_linear_attention.core.attention import StandardAttention, PiecewiseAttention


class TestPiecewiseAttention:
    """Test PiecewiseAttention (piecewise linear attention)."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch = 4
        seq_len = 32
        dim = 64

        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        return Q, K, V

    def test_initialization(self):
        """Test initialization."""
        attention = PiecewiseAttention(dim=64, dropout=0.0)
        assert attention.dim == 64

        # Test custom pseudo-query function
        def first_query(Q):
            return Q[:, 0, :]

        attention_custom = PiecewiseAttention(dim=64, pseudo_query_fn=first_query)
        assert attention_custom.dim == 64

    def test_output_shape(self, sample_data):
        """Test that output has correct shape."""
        Q, K, V = sample_data
        batch, seq_len, dim = Q.shape

        attention = PiecewiseAttention(dim=dim, dropout=0.0)
        output, _ = attention(Q, K, V)

        assert output.shape == (batch, seq_len, dim)

    def test_output_is_finite(self, sample_data):
        """Test that output contains no NaN or inf."""
        Q, K, V = sample_data

        attention = PiecewiseAttention(dim=64, dropout=0.0)
        output, _ = attention(Q, K, V)

        assert torch.isfinite(output).all()

    def test_gradient_flow(self, sample_data):
        """Test that gradients flow correctly."""
        Q, K, V = sample_data
        Q = Q.requires_grad_(True)
        K = K.requires_grad_(True)
        V = V.requires_grad_(True)

        attention = PiecewiseAttention(dim=64, dropout=0.0)
        output, _ = attention(Q, K, V)

        loss = output.sum()
        loss.backward()

        assert Q.grad is not None
        assert K.grad is not None
        assert V.grad is not None
        assert torch.isfinite(Q.grad).all()
        assert torch.isfinite(K.grad).all()
        assert torch.isfinite(V.grad).all()

    def test_pseudo_query_selection_methods(self, sample_data):
        """Test different pseudo-query selection methods."""
        Q, K, V = sample_data
        batch, seq_len, dim = Q.shape

        methods = ["mean", "first", "median", "random"]
        outputs = []

        for method in methods:
            # Create pseudo-query function based on method
            if method == "mean":
                pseudo_query_fn = lambda Q: Q.mean(dim=1)
            elif method == "first":
                pseudo_query_fn = lambda Q: Q[:, 0, :]
            elif method == "median":
                pseudo_query_fn = lambda Q: Q.median(dim=1)[0]
            else:  # random
                pseudo_query_fn = lambda Q: Q[:, torch.randint(Q.size(1), (1,)).item(), :]

            attention = PiecewiseAttention(
                dim=dim, pseudo_query_fn=pseudo_query_fn, dropout=0.0
            )
            output, pseudo_queries = attention(Q, K, V)

            # Check pseudo-query shape
            assert pseudo_queries.shape == (batch, dim)

            # Check output shape
            assert output.shape == (batch, seq_len, dim)

            # Verify method produces expected result
            if method == "mean":
                expected = Q.mean(dim=1)
                assert torch.allclose(pseudo_queries, expected, atol=1e-6)
            elif method == "first":
                expected = Q[:, 0, :]
                assert torch.allclose(pseudo_queries, expected, atol=1e-6)

            outputs.append(output)

        # Different methods should give different results
        assert not torch.allclose(outputs[0], outputs[1])

    def test_approximation_quality_vs_standard(self, sample_data):
        """Test approximation quality compared to standard attention."""
        Q, K, V = sample_data

        # Standard attention (ground truth)
        standard_attn = StandardAttention(dim=64, dropout=0.0, scale=True)
        standard_output, _ = standard_attn(Q, K, V)

        # Per-sample piecewise attention
        piecewise_attn = PiecewiseAttention(
            dim=64, dropout=0.0, scale=True
        )
        piecewise_output, _ = piecewise_attn(Q, K, V)

        # Compute relative error
        relative_error = (
            torch.norm(standard_output - piecewise_output) /
            torch.norm(standard_output)
        ).item()

        # Should have reasonable approximation (< 100% error)
        assert relative_error < 1.0, f"Relative error too large: {relative_error}"

        # Should be finite
        assert torch.isfinite(piecewise_output).all()

    def test_batch_independence(self):
        """Test that each batch is computed independently."""
        batch = 2
        seq_len = 16
        dim = 32

        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        attention = PiecewiseAttention(dim=dim, dropout=0.0)

        # Compute full batch
        output_batched, _ = attention(Q, K, V)

        # Compute each sample individually
        outputs_individual = []
        for b in range(batch):
            output_b, _ = attention(Q[b:b+1], K[b:b+1], V[b:b+1])
            outputs_individual.append(output_b)

        output_individual = torch.cat(outputs_individual, dim=0)

        # Should be identical
        assert torch.allclose(output_batched, output_individual, atol=1e-6)

    def test_single_query_case(self):
        """Test edge case where sample has only one query."""
        batch = 2
        seq_len = 1  # Only one query
        dim = 32

        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        attention = PiecewiseAttention(dim=dim)
        output, _ = attention(Q, K, V)

        # Should still work
        assert output.shape == (batch, seq_len, dim)
        assert torch.isfinite(output).all()

    def test_deterministic_with_mean(self):
        """Test that mean method is deterministic."""
        Q = torch.randn(2, 16, 32)
        K = torch.randn(2, 16, 32)
        V = torch.randn(2, 16, 32)

        attention = PiecewiseAttention(dim=32, dropout=0.0)

        # Run twice
        output1, _ = attention(Q, K, V)
        output2, _ = attention(Q, K, V)

        # Should be identical
        assert torch.allclose(output1, output2, atol=1e-8)


class TestCorrectnessVsSequential:
    """Test that vectorized per-sample computation matches sequential."""

    def test_matches_sequential_computation(self):
        """Test that batched computation matches computing each sample sequentially."""
        batch = 4
        seq_len = 16
        dim = 32

        torch.manual_seed(42)
        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        attention = PerSamplePiecewiseAttention(
            dim=dim, pseudo_query_method="mean", dropout=0.0, scale=True
        )

        # Batched computation
        output_batched, _ = attention(Q, K, V)

        # Sequential computation
        outputs_sequential = []
        for b in range(batch):
            output_b, _ = attention(
                Q[b:b+1],
                K[b:b+1],
                V[b:b+1]
            )
            outputs_sequential.append(output_b)

        output_sequential = torch.cat(outputs_sequential, dim=0)

        # Should match exactly
        assert torch.allclose(output_batched, output_sequential, atol=1e-6)

    def test_matches_manual_computation(self):
        """Test against manual computation of the algorithm."""
        batch = 2
        seq_len = 4
        dim = 8

        torch.manual_seed(123)
        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        attention = PiecewiseAttention(dim=dim, dropout=0.0, scale=True)

        # Get output from module
        output_module, pseudo_queries_module = attention(Q, K, V)

        # Manual computation
        import math

        outputs_manual = []
        for b in range(batch):
            # 1. Select pseudo-query (mean)
            pseudo_q = Q[b].mean(dim=0)  # (dim,)

            # 2. Compute attention at pseudo-query
            scores = (pseudo_q @ K[b].T) / math.sqrt(dim)  # (seq_len,)
            attn_weights = torch.softmax(scores, dim=0)  # (seq_len,)
            pseudo_output = (attn_weights.unsqueeze(1) * V[b]).sum(dim=0)  # (dim,)

            # 3. Compute Jacobian
            weighted_k = (attn_weights.unsqueeze(1) * K[b]).sum(dim=0)  # (dim,)
            weighted_v = pseudo_output  # (dim,)

            # J = scale * [sum_i alpha_i (v_i ⊗ k_i) - weighted_v ⊗ weighted_k]
            scale = 1.0 / math.sqrt(dim)
            J = torch.zeros(dim, dim)
            for i in range(seq_len):
                J += attn_weights[i] * torch.outer(V[b, i], K[b, i])
            J = scale * (J - torch.outer(weighted_v, weighted_k))

            # 4. Apply approximation to all queries
            outputs_b = []
            for i in range(seq_len):
                delta = Q[b, i] - pseudo_q
                output_i = pseudo_output + J @ delta
                outputs_b.append(output_i)

            output_b = torch.stack(outputs_b, dim=0)  # (seq_len, dim)
            outputs_manual.append(output_b)

        output_manual = torch.stack(outputs_manual, dim=0)  # (batch, seq_len, dim)

        # Should match
        assert torch.allclose(output_module, output_manual, atol=1e-5)


class TestApproximationQuality:
    """Test approximation quality in different scenarios."""

    def test_quality_with_similar_queries(self):
        """Test that approximation is good when queries are similar."""
        batch = 2
        seq_len = 16
        dim = 32

        # Create similar queries (mean + small noise)
        torch.manual_seed(42)
        base = torch.randn(batch, 1, dim)
        noise = 0.1 * torch.randn(batch, seq_len, dim)
        Q = base + noise

        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        # Standard attention
        standard_attn = StandardAttention(dim=dim, dropout=0.0, scale=True)
        standard_output, _ = standard_attn(Q, K, V)

        # Per-sample piecewise
        piecewise_attn = PerSamplePiecewiseAttention(
            dim=dim, pseudo_query_method="mean", dropout=0.0, scale=True
        )
        piecewise_output, _ = piecewise_attn(Q, K, V)

        # Error should be small when queries are similar
        relative_error = (
            torch.norm(standard_output - piecewise_output) /
            torch.norm(standard_output)
        ).item()

        # With similar queries, error should be < 20%
        assert relative_error < 0.2, f"Error too large with similar queries: {relative_error}"

    def test_quality_degrades_with_diverse_queries(self):
        """Test that approximation degrades when queries are diverse."""
        batch = 2
        seq_len = 16
        dim = 32

        # Similar queries
        torch.manual_seed(42)
        base = torch.randn(batch, 1, dim)
        Q_similar = base + 0.1 * torch.randn(batch, seq_len, dim)

        # Diverse queries
        Q_diverse = torch.randn(batch, seq_len, dim)

        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        # Standard attention
        standard_attn = StandardAttention(dim=dim, dropout=0.0, scale=True)

        # Per-sample piecewise
        piecewise_attn = PerSamplePiecewiseAttention(
            dim=dim, pseudo_query_method="mean", dropout=0.0, scale=True
        )

        # Error with similar queries
        standard_similar, _ = standard_attn(Q_similar, K, V)
        piecewise_similar, _ = piecewise_attn(Q_similar, K, V)
        error_similar = (
            torch.norm(standard_similar - piecewise_similar) /
            torch.norm(standard_similar)
        ).item()

        # Error with diverse queries
        standard_diverse, _ = standard_attn(Q_diverse, K, V)
        piecewise_diverse, _ = piecewise_attn(Q_diverse, K, V)
        error_diverse = (
            torch.norm(standard_diverse - piecewise_diverse) /
            torch.norm(standard_diverse)
        ).item()

        # Diverse queries should have higher error
        assert error_diverse > error_similar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
