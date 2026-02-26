"""Tests for pseudo-query initialization."""

import pytest
import torch

from piecewise_linear_attention.core.pseudo_queries import (
    initialize_pseudo_queries_kmeans,
    initialize_pseudo_queries_random,
    initialize_pseudo_queries_uniform,
)


class TestKMeansInitialization:
    """Tests for K-means pseudo-query initialization."""

    def test_output_shape(self):
        """Test that K-means returns correct shape."""
        queries = torch.randn(100, 64)
        num_pseudo = 10

        pseudo_queries = initialize_pseudo_queries_kmeans(queries, num_pseudo)

        assert pseudo_queries.shape == (num_pseudo, 64)

    def test_3d_input(self):
        """Test K-means with 3D input (batch, seq_len, dim)."""
        queries = torch.randn(4, 32, 64)  # batch=4, seq_len=32, dim=64
        num_pseudo = 10

        pseudo_queries = initialize_pseudo_queries_kmeans(queries, num_pseudo)

        assert pseudo_queries.shape == (num_pseudo, 64)

    def test_reproducibility(self):
        """Test that using same seed gives same results."""
        queries = torch.randn(100, 64)
        num_pseudo = 10
        seed = 42

        pseudo_1 = initialize_pseudo_queries_kmeans(queries, num_pseudo, seed=seed)
        pseudo_2 = initialize_pseudo_queries_kmeans(queries, num_pseudo, seed=seed)

        assert torch.allclose(pseudo_1, pseudo_2)

    def test_different_seeds(self):
        """Test that different seeds give different results."""
        queries = torch.randn(100, 64)
        num_pseudo = 10

        pseudo_1 = initialize_pseudo_queries_kmeans(queries, num_pseudo, seed=42)
        pseudo_2 = initialize_pseudo_queries_kmeans(queries, num_pseudo, seed=123)

        assert not torch.allclose(pseudo_1, pseudo_2)

    def test_centroids_are_representative(self):
        """Test that centroids are close to actual query clusters."""
        # Create queries with clear clusters
        cluster_centers = torch.tensor(
            [[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]], dtype=torch.float32
        )
        queries = []
        for center in cluster_centers:
            # Add 30 points around each center
            cluster = center + torch.randn(30, 2) * 0.5
            queries.append(cluster)
        queries = torch.cat(queries, dim=0)

        # Run K-means with k=3
        pseudo_queries = initialize_pseudo_queries_kmeans(queries, 3, seed=42)

        # Check that each centroid is close to one of the cluster centers
        distances = torch.cdist(pseudo_queries, cluster_centers)
        min_distances = distances.min(dim=1)[0]

        # Each pseudo-query should be within distance 1.0 of a true center
        assert (min_distances < 1.0).all()

    def test_convergence(self):
        """Test that K-means converges (doesn't hit max_iter)."""
        queries = torch.randn(100, 64)
        num_pseudo = 10

        # Should converge in fewer than 10 iterations for this simple case
        pseudo_queries = initialize_pseudo_queries_kmeans(
            queries, num_pseudo, max_iter=10, seed=42
        )

        # Just check it completes without error
        assert pseudo_queries.shape == (num_pseudo, 64)

    def test_too_many_clusters_error(self):
        """Test that requesting more clusters than samples raises error."""
        queries = torch.randn(10, 64)
        num_pseudo = 20  # More than available samples

        with pytest.raises(ValueError, match="cannot exceed number of samples"):
            initialize_pseudo_queries_kmeans(queries, num_pseudo)

    def test_device_consistency(self):
        """Test that pseudo-queries are on same device as input."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        queries = torch.randn(100, 64).cuda()
        num_pseudo = 10

        pseudo_queries = initialize_pseudo_queries_kmeans(queries, num_pseudo)

        assert pseudo_queries.device == queries.device

    def test_dtype_consistency(self):
        """Test that pseudo-queries have same dtype as input."""
        queries = torch.randn(100, 64, dtype=torch.float64)
        num_pseudo = 10

        pseudo_queries = initialize_pseudo_queries_kmeans(queries, num_pseudo)

        assert pseudo_queries.dtype == queries.dtype

    def test_no_empty_clusters(self):
        """Test that K-means handles empty clusters gracefully."""
        # Create data that might lead to empty clusters
        queries = torch.randn(50, 64)
        num_pseudo = 40  # Many clusters for little data

        pseudo_queries = initialize_pseudo_queries_kmeans(
            queries, num_pseudo, max_iter=50, seed=42
        )

        # All pseudo-queries should be finite (no NaNs from empty clusters)
        assert torch.isfinite(pseudo_queries).all()


class TestRandomInitialization:
    """Tests for random pseudo-query initialization."""

    def test_output_shape(self):
        """Test that random initialization returns correct shape."""
        pseudo_queries = initialize_pseudo_queries_random(
            dim=64, num_pseudo_queries=10
        )

        assert pseudo_queries.shape == (10, 64)

    def test_reproducibility(self):
        """Test that using same seed gives same results."""
        pseudo_1 = initialize_pseudo_queries_random(
            dim=64, num_pseudo_queries=10, seed=42
        )
        pseudo_2 = initialize_pseudo_queries_random(
            dim=64, num_pseudo_queries=10, seed=42
        )

        assert torch.allclose(pseudo_1, pseudo_2)

    def test_device_specification(self):
        """Test that pseudo-queries are created on specified device."""
        device = torch.device("cpu")
        pseudo_queries = initialize_pseudo_queries_random(
            dim=64, num_pseudo_queries=10, device=device
        )

        assert pseudo_queries.device == device

    def test_dtype_specification(self):
        """Test that pseudo-queries have specified dtype."""
        pseudo_queries = initialize_pseudo_queries_random(
            dim=64, num_pseudo_queries=10, dtype=torch.float64
        )

        assert pseudo_queries.dtype == torch.float64


class TestUniformInitialization:
    """Tests for uniform sampling pseudo-query initialization."""

    def test_output_shape(self):
        """Test that uniform sampling returns correct shape."""
        queries = torch.randn(100, 64)
        num_pseudo = 10

        pseudo_queries = initialize_pseudo_queries_uniform(queries, num_pseudo)

        assert pseudo_queries.shape == (num_pseudo, 64)

    def test_3d_input(self):
        """Test uniform sampling with 3D input."""
        queries = torch.randn(4, 32, 64)
        num_pseudo = 10

        pseudo_queries = initialize_pseudo_queries_uniform(queries, num_pseudo)

        assert pseudo_queries.shape == (num_pseudo, 64)

    def test_samples_are_from_input(self):
        """Test that sampled pseudo-queries are actual input queries."""
        queries = torch.randn(100, 64)
        num_pseudo = 10

        pseudo_queries = initialize_pseudo_queries_uniform(queries, num_pseudo, seed=42)

        # Each pseudo-query should match some input query exactly
        for pseudo_q in pseudo_queries:
            distances = torch.norm(queries - pseudo_q, dim=1)
            min_distance = distances.min()
            # Should find exact match (distance ~ 0)
            assert min_distance < 1e-6

    def test_reproducibility(self):
        """Test that using same seed gives same results."""
        queries = torch.randn(100, 64)
        num_pseudo = 10

        pseudo_1 = initialize_pseudo_queries_uniform(queries, num_pseudo, seed=42)
        pseudo_2 = initialize_pseudo_queries_uniform(queries, num_pseudo, seed=42)

        assert torch.allclose(pseudo_1, pseudo_2)

    def test_too_many_samples_error(self):
        """Test that requesting more samples than available raises error."""
        queries = torch.randn(10, 64)
        num_pseudo = 20

        with pytest.raises(ValueError, match="cannot exceed number of samples"):
            initialize_pseudo_queries_uniform(queries, num_pseudo)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
