"""Tests for Jacobian computation."""

import pytest
import torch
import torch.nn.functional as F

from piecewise_linear_attention.core.jacobian import (
    compute_attention_jacobian_analytical,
    compute_attention_jacobian_autograd,
    compute_attention_with_jacobian,
    batch_compute_jacobians,
)


class TestJacobianCorrectness:
    """Test that analytical Jacobian matches autograd."""

    def test_jacobian_matches_autograd(self):
        """Test that analytical Jacobian matches autograd computation."""
        dim = 32
        seq_len = 16

        query = torch.randn(dim)
        K = torch.randn(seq_len, dim)
        V = torch.randn(seq_len, dim)

        # Compute using both methods
        jac_analytical = compute_attention_jacobian_analytical(query, K, V, scale=True)
        jac_autograd = compute_attention_jacobian_autograd(query, K, V, scale=True)

        # Should match within numerical tolerance
        assert torch.allclose(jac_analytical, jac_autograd, atol=1e-5, rtol=1e-4), \
            f"Max diff: {(jac_analytical - jac_autograd).abs().max()}"

    def test_jacobian_without_scaling(self):
        """Test Jacobian computation without scaling."""
        dim = 32
        seq_len = 16

        query = torch.randn(dim)
        K = torch.randn(seq_len, dim)
        V = torch.randn(seq_len, dim)

        jac_analytical = compute_attention_jacobian_analytical(query, K, V, scale=False)
        jac_autograd = compute_attention_jacobian_autograd(query, K, V, scale=False)

        assert torch.allclose(jac_analytical, jac_autograd, atol=1e-5, rtol=1e-4)

    def test_jacobian_with_mask(self):
        """Test Jacobian computation with masking."""
        dim = 32
        seq_len = 16

        query = torch.randn(dim)
        K = torch.randn(seq_len, dim)
        V = torch.randn(seq_len, dim)

        # Mask out last 5 positions
        mask = torch.ones(seq_len)
        mask[-5:] = 0

        jac_analytical = compute_attention_jacobian_analytical(
            query, K, V, scale=True, mask=mask
        )
        jac_autograd = compute_attention_jacobian_autograd(
            query, K, V, scale=True, mask=mask
        )

        assert torch.allclose(jac_analytical, jac_autograd, atol=1e-5, rtol=1e-4)

    def test_jacobian_different_dimensions(self):
        """Test Jacobian for various dimensions."""
        test_configs = [
            (16, 8),   # dim=16, seq_len=8
            (64, 32),  # dim=64, seq_len=32
            (128, 64), # dim=128, seq_len=64
        ]

        for dim, seq_len in test_configs:
            query = torch.randn(dim)
            K = torch.randn(seq_len, dim)
            V = torch.randn(seq_len, dim)

            jac_analytical = compute_attention_jacobian_analytical(query, K, V)
            jac_autograd = compute_attention_jacobian_autograd(query, K, V)

            assert torch.allclose(jac_analytical, jac_autograd, atol=1e-5, rtol=1e-4), \
                f"Failed for dim={dim}, seq_len={seq_len}"


class TestJacobianProperties:
    """Test mathematical properties of the Jacobian."""

    def test_jacobian_shape(self):
        """Test that Jacobian has correct shape."""
        dim = 32
        seq_len = 16

        query = torch.randn(dim)
        K = torch.randn(seq_len, dim)
        V = torch.randn(seq_len, dim)

        jacobian = compute_attention_jacobian_analytical(query, K, V)

        assert jacobian.shape == (dim, dim)

    def test_jacobian_finite(self):
        """Test that Jacobian contains no NaNs or infinities."""
        dim = 32
        seq_len = 16

        query = torch.randn(dim)
        K = torch.randn(seq_len, dim)
        V = torch.randn(seq_len, dim)

        jacobian = compute_attention_jacobian_analytical(query, K, V)

        assert torch.isfinite(jacobian).all()

    def test_jacobian_numerical_stability(self):
        """Test Jacobian computation with extreme values."""
        dim = 32
        seq_len = 16

        # Test with large values
        query = torch.randn(dim) * 10
        K = torch.randn(seq_len, dim) * 10
        V = torch.randn(seq_len, dim) * 10

        jacobian = compute_attention_jacobian_analytical(query, K, V, scale=True)
        assert torch.isfinite(jacobian).all()

        # Test with small values
        query = torch.randn(dim) * 0.1
        K = torch.randn(seq_len, dim) * 0.1
        V = torch.randn(seq_len, dim) * 0.1

        jacobian = compute_attention_jacobian_analytical(query, K, V, scale=True)
        assert torch.isfinite(jacobian).all()


class TestLinearApproximationQuality:
    """Test that linear approximation using Jacobian is accurate near query point."""

    def compute_attention_output(self, query, K, V, scale=True, mask=None):
        """Helper to compute attention output."""
        import math
        scores = torch.matmul(K, query)
        if scale:
            scores = scores / math.sqrt(query.shape[0])
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(scores, dim=0)
        return torch.matmul(alpha, V)

    def test_first_order_approximation_accuracy(self):
        """Test that linear approximation is accurate for small perturbations."""
        dim = 32
        seq_len = 16

        # Base query point
        query = torch.randn(dim)
        K = torch.randn(seq_len, dim)
        V = torch.randn(seq_len, dim)

        # Compute Jacobian and output at base point
        output_base, jacobian = compute_attention_with_jacobian(query, K, V)

        # Test at various distances
        epsilons = [0.001, 0.01, 0.1]
        errors = []

        for eps in epsilons:
            # Perturbed query
            direction = torch.randn(dim)
            direction = direction / direction.norm()  # Unit direction
            query_perturbed = query + eps * direction

            # True output at perturbed point
            output_true = self.compute_attention_output(query_perturbed, K, V)

            # Linear approximation: A(q + δq) ≈ A(q) + J·δq
            delta_q = eps * direction
            output_approx = output_base + torch.matmul(jacobian, delta_q)

            # Compute error
            error = torch.norm(output_true - output_approx) / torch.norm(output_true)
            errors.append(error.item())

        # Error should decrease roughly quadratically with epsilon
        # (first-order approximation has O(epsilon^2) error)
        assert errors[0] < errors[1] < errors[2], \
            f"Errors should increase with epsilon: {errors}"

        # Error at eps=0.001 should be very small
        assert errors[0] < 1e-4, f"Error at eps=0.001: {errors[0]}"

    def test_approximation_error_scaling(self):
        """Test that approximation error scales as O(epsilon^2)."""
        dim = 32
        seq_len = 16

        query = torch.randn(dim)
        K = torch.randn(seq_len, dim)
        V = torch.randn(seq_len, dim)

        output_base, jacobian = compute_attention_with_jacobian(query, K, V)

        # Test at multiple scales
        direction = torch.randn(dim)
        direction = direction / direction.norm()

        # Use larger epsilons to avoid numerical underflow
        # At eps < 0.003, errors become smaller than machine precision (~1e-6)
        eps_values = [0.05, 0.025, 0.0125]
        errors = []

        for eps in eps_values:
            query_perturbed = query + eps * direction
            output_true = self.compute_attention_output(query_perturbed, K, V)
            output_approx = output_base + torch.matmul(jacobian, eps * direction)
            error = torch.norm(output_true - output_approx).item()
            errors.append(error)

        # When eps is halved, error should be divided by ~4 (quadratic scaling)
        ratio1 = errors[0] / errors[1]  # Should be ~ 4
        ratio2 = errors[1] / errors[2]  # Should be ~ 4

        # Allow tolerance for numerical effects and random variation
        # Typical ratios observed: 3.5-4.5, but can vary from 2.5-6.0
        # depending on local curvature of the attention function
        assert 2.5 < ratio1 < 6.0, f"Ratio 1: {ratio1}, errors: {errors}"
        assert 2.5 < ratio2 < 6.0, f"Ratio 2: {ratio2}, errors: {errors}"

        # Key check: both ratios should be > 2 (better than linear)
        assert ratio1 > 2.0 and ratio2 > 2.0, \
            "Ratios should be > 2 for quadratic approximation"


class TestBatchComputation:
    """Test batched Jacobian computation."""

    def test_batch_jacobians_shape(self):
        """Test that batch computation returns correct shape."""
        num_queries = 5
        dim = 32
        seq_len = 16

        queries = torch.randn(num_queries, dim)
        K = torch.randn(seq_len, dim)
        V = torch.randn(seq_len, dim)

        jacobians = batch_compute_jacobians(queries, K, V)

        assert jacobians.shape == (num_queries, dim, dim)

    def test_batch_matches_individual(self):
        """Test that batch computation matches individual computations."""
        num_queries = 5
        dim = 32
        seq_len = 16

        queries = torch.randn(num_queries, dim)
        K = torch.randn(seq_len, dim)
        V = torch.randn(seq_len, dim)

        # Batch computation
        jacobians_batch = batch_compute_jacobians(queries, K, V)

        # Individual computations
        for i in range(num_queries):
            jac_individual = compute_attention_jacobian_analytical(queries[i], K, V)
            assert torch.allclose(jacobians_batch[i], jac_individual, atol=1e-6)


class TestAttentionWithJacobian:
    """Test joint computation of attention and Jacobian."""

    def test_output_matches_separate_computation(self):
        """Test that joint computation gives same results as separate."""
        dim = 32
        seq_len = 16

        query = torch.randn(dim)
        K = torch.randn(seq_len, dim)
        V = torch.randn(seq_len, dim)

        # Joint computation
        output_joint, jac_joint = compute_attention_with_jacobian(query, K, V)

        # Separate computations
        scores = torch.matmul(K, query) / (dim ** 0.5)
        alpha = F.softmax(scores, dim=0)
        output_separate = torch.matmul(alpha, V)
        jac_separate = compute_attention_jacobian_analytical(query, K, V)

        assert torch.allclose(output_joint, output_separate, atol=1e-6)
        assert torch.allclose(jac_joint, jac_separate, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
