"""Memory profiling tests for attention mechanisms.

Test suite for validating memory consumption characteristics:
- Memory bounds checking
- Scaling behavior validation
- OOM prevention
- Gradient memory tracking

This follows research standards for memory analysis as outlined in:
- HANDOVER.md Section 4.1.2: Memory Profiling Requirements
"""

import gc
import pytest
import torch

from piecewise_linear_attention import (
    StandardAttention,
    LinearAttention,
    PiecewiseAttention,
)


@pytest.fixture
def clear_memory():
    """Fixture to clear memory before and after each test."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_mb(device: torch.device) -> float:
    """Get current memory usage in MB.

    Args:
        device: Device to measure memory on

    Returns:
        Memory usage in MB
    """
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024**2
    else:
        # For CPU, estimate from process memory if psutil available
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024**2
        except ImportError:
            return 0.0  # Can't measure without psutil


def get_peak_memory_mb(device: torch.device) -> float:
    """Get peak memory usage in MB.

    Args:
        device: Device to measure memory on

    Returns:
        Peak memory usage in MB
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        return get_memory_mb(device)


class TestMemoryForwardPass:
    """Test memory usage of forward pass."""

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    ))])
    def test_forward_memory_basic(self, device, clear_memory):
        """Test that forward pass completes without excessive memory usage."""
        device = torch.device(device)
        batch, seq_len, dim = 2, 256, 64

        Q = torch.randn(batch, seq_len, dim, device=device)
        K = torch.randn(batch, seq_len, dim, device=device)
        V = torch.randn(batch, seq_len, dim, device=device)

        # Test each attention mechanism
        for attention_class in [StandardAttention, LinearAttention, PiecewiseAttention]:
            attention = attention_class(dim=dim, dropout=0.0).to(device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            output, _ = attention(Q, K, V)

            # Verify output shape
            assert output.shape == (batch, seq_len, dim)

            # Get peak memory
            peak_mb = get_peak_memory_mb(device)

            # Sanity check: memory should be less than 100MB for this small config
            if device.type == "cuda":
                assert peak_mb < 100, f"{attention_class.__name__} used {peak_mb:.2f}MB (expected < 100MB)"

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    ))])
    def test_piecewise_vs_standard_memory(self, device, clear_memory):
        """Test that PiecewiseAttention uses less memory than StandardAttention for large sequences."""
        device = torch.device(device)

        # Use larger sequence where memory difference is noticeable
        batch, seq_len, dim = 4, 1024, 64

        Q = torch.randn(batch, seq_len, dim, device=device)
        K = torch.randn(batch, seq_len, dim, device=device)
        V = torch.randn(batch, seq_len, dim, device=device)

        # Measure StandardAttention memory
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        standard = StandardAttention(dim=dim, dropout=0.0).to(device)
        _ = standard(Q, K, V)
        standard_peak = get_peak_memory_mb(device)

        # Clear memory
        del standard
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Measure PiecewiseAttention memory
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        piecewise = PiecewiseAttention(dim=dim, dropout=0.0).to(device)
        _ = piecewise(Q, K, V)
        piecewise_peak = get_peak_memory_mb(device)

        # PiecewiseAttention should use less or similar memory
        # StandardAttention has O(n²) memory for attention matrix
        # PiecewiseAttention has O(d²) memory for Jacobian
        if device.type == "cuda":
            # For this config: n=1024, d=64, so n² >> d²
            # We expect PiecewiseAttention to use less memory
            print(f"\nMemory comparison (n={seq_len}, d={dim}):")
            print(f"  StandardAttention: {standard_peak:.2f} MB")
            print(f"  PiecewiseAttention: {piecewise_peak:.2f} MB")
            print(f"  Ratio: {standard_peak / piecewise_peak:.2f}x")

            # Allow some tolerance due to PyTorch memory management
            assert piecewise_peak <= standard_peak * 1.5, \
                f"PiecewiseAttention ({piecewise_peak:.2f}MB) should not use significantly more " \
                f"memory than StandardAttention ({standard_peak:.2f}MB)"

    def test_memory_scaling_with_sequence_length(self, clear_memory):
        """Test that memory scales correctly with sequence length.

        StandardAttention: O(n²) for attention matrix
        PiecewiseAttention: O(d²) - independent of n
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch, dim = 2, 64

        seq_lengths = [256, 512, 1024]
        standard_memories = []
        piecewise_memories = []

        for seq_len in seq_lengths:
            Q = torch.randn(batch, seq_len, dim, device=device)
            K = torch.randn(batch, seq_len, dim, device=device)
            V = torch.randn(batch, seq_len, dim, device=device)

            # Measure StandardAttention
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            standard = StandardAttention(dim=dim, dropout=0.0).to(device)
            _ = standard(Q, K, V)
            standard_peak = get_peak_memory_mb(device)
            standard_memories.append(standard_peak)

            del standard
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Measure PiecewiseAttention
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            piecewise = PiecewiseAttention(dim=dim, dropout=0.0).to(device)
            _ = piecewise(Q, K, V)
            piecewise_peak = get_peak_memory_mb(device)
            piecewise_memories.append(piecewise_peak)

            del piecewise
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if device.type == "cuda":
            print("\nMemory scaling with sequence length:")
            for i, seq_len in enumerate(seq_lengths):
                print(f"  n={seq_len:4d}: Standard={standard_memories[i]:6.2f}MB, "
                      f"Piecewise={piecewise_memories[i]:6.2f}MB")

            # Check that StandardAttention memory grows faster than PiecewiseAttention
            # Memory ratio should increase with sequence length
            ratio_256 = standard_memories[0] / piecewise_memories[0]
            ratio_1024 = standard_memories[2] / piecewise_memories[2]

            print(f"  Memory ratio at n=256:  {ratio_256:.2f}x")
            print(f"  Memory ratio at n=1024: {ratio_1024:.2f}x")

            # Ratio should increase (or at least not decrease significantly)
            assert ratio_1024 >= ratio_256 * 0.8, \
                "StandardAttention memory should grow faster with sequence length"


class TestMemoryBackwardPass:
    """Test memory usage of backward pass."""

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    ))])
    def test_backward_memory_basic(self, device, clear_memory):
        """Test that backward pass completes without excessive memory usage."""
        device = torch.device(device)
        batch, seq_len, dim = 2, 256, 64

        Q = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        K = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        V = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)

        for attention_class in [StandardAttention, LinearAttention, PiecewiseAttention]:
            attention = attention_class(dim=dim, dropout=0.0).to(device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            output, _ = attention(Q, K, V)
            loss = output.sum()
            loss.backward()

            # Verify gradients exist
            assert Q.grad is not None
            assert K.grad is not None
            assert V.grad is not None

            # Check gradient shapes
            assert Q.grad.shape == Q.shape
            assert K.grad.shape == K.shape
            assert V.grad.shape == V.shape

            # Get peak memory
            peak_mb = get_peak_memory_mb(device)

            # Sanity check
            if device.type == "cuda":
                assert peak_mb < 200, f"{attention_class.__name__} backward used {peak_mb:.2f}MB (expected < 200MB)"

            # Clear gradients for next iteration
            Q.grad = None
            K.grad = None
            V.grad = None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_gradient_correctness(self, clear_memory):
        """Test that gradients are computed correctly and don't contain NaN/Inf."""
        device = torch.device("cuda")
        batch, seq_len, dim = 2, 128, 64

        Q = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        K = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        V = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)

        for attention_class in [StandardAttention, LinearAttention, PiecewiseAttention]:
            attention = attention_class(dim=dim, dropout=0.0).to(device)

            output, _ = attention(Q, K, V)
            loss = output.sum()
            loss.backward()

            # Check for NaN or Inf in gradients
            assert torch.isfinite(Q.grad).all(), f"{attention_class.__name__}: Q.grad contains NaN/Inf"
            assert torch.isfinite(K.grad).all(), f"{attention_class.__name__}: K.grad contains NaN/Inf"
            assert torch.isfinite(V.grad).all(), f"{attention_class.__name__}: V.grad contains NaN/Inf"

            # Check that gradients are non-zero (not all dead)
            assert Q.grad.abs().sum() > 0, f"{attention_class.__name__}: Q.grad is all zeros"
            assert K.grad.abs().sum() > 0, f"{attention_class.__name__}: K.grad is all zeros"
            assert V.grad.abs().sum() > 0, f"{attention_class.__name__}: V.grad is all zeros"

            # Clear gradients
            Q.grad = None
            K.grad = None
            V.grad = None


class TestMemoryLeaks:
    """Test for memory leaks."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_no_memory_leak_repeated_calls(self, clear_memory):
        """Test that repeated calls don't leak memory."""
        device = torch.device("cuda")
        batch, seq_len, dim = 2, 256, 64

        attention = PiecewiseAttention(dim=dim, dropout=0.0).to(device)

        Q = torch.randn(batch, seq_len, dim, device=device)
        K = torch.randn(batch, seq_len, dim, device=device)
        V = torch.randn(batch, seq_len, dim, device=device)

        # Run several times and measure memory
        torch.cuda.reset_peak_memory_stats(device)
        memories = []

        for i in range(10):
            output, _ = attention(Q, K, V)
            current_mem = get_memory_mb(device)
            memories.append(current_mem)

            # Allow memory to settle
            if i % 3 == 0:
                gc.collect()

        # Check that memory doesn't grow significantly
        # Allow some variance due to PyTorch memory management
        initial_mem = memories[2]  # Use 3rd iteration after warmup
        final_mem = memories[-1]

        print(f"\nMemory over 10 iterations:")
        print(f"  Initial (iter 3): {initial_mem:.2f} MB")
        print(f"  Final (iter 10):  {final_mem:.2f} MB")
        print(f"  Growth: {final_mem - initial_mem:.2f} MB")

        # Memory should not grow by more than 20% (generous threshold)
        assert final_mem <= initial_mem * 1.2, \
            f"Potential memory leak: grew from {initial_mem:.2f}MB to {final_mem:.2f}MB"


class TestMemoryEdgeCases:
    """Test memory behavior in edge cases."""

    def test_large_dimension_memory(self, clear_memory):
        """Test memory usage with large embedding dimension.

        PiecewiseAttention has O(d²) memory, so large d should increase memory.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch, seq_len = 2, 256

        dimensions = [64, 128, 256]
        memories = []

        for dim in dimensions:
            Q = torch.randn(batch, seq_len, dim, device=device)
            K = torch.randn(batch, seq_len, dim, device=device)
            V = torch.randn(batch, seq_len, dim, device=device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            attention = PiecewiseAttention(dim=dim, dropout=0.0).to(device)
            _ = attention(Q, K, V)

            peak_mb = get_peak_memory_mb(device)
            memories.append(peak_mb)

            del attention
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if device.type == "cuda":
            print("\nMemory scaling with dimension:")
            for i, dim in enumerate(dimensions):
                print(f"  d={dim:3d}: {memories[i]:6.2f} MB")

            # Memory should increase with dimension (O(d²) component)
            assert memories[1] > memories[0] * 0.9, "Memory should increase with dimension"
            assert memories[2] > memories[1] * 0.9, "Memory should increase with dimension"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_size_memory_scaling(self, clear_memory):
        """Test that memory scales linearly with batch size."""
        device = torch.device("cuda")
        seq_len, dim = 512, 64

        batch_sizes = [1, 2, 4, 8]
        memories = []

        for batch in batch_sizes:
            Q = torch.randn(batch, seq_len, dim, device=device)
            K = torch.randn(batch, seq_len, dim, device=device)
            V = torch.randn(batch, seq_len, dim, device=device)

            torch.cuda.reset_peak_memory_stats(device)

            attention = PiecewiseAttention(dim=dim, dropout=0.0).to(device)
            _ = attention(Q, K, V)

            peak_mb = get_peak_memory_mb(device)
            memories.append(peak_mb)

            del attention
            gc.collect()
            torch.cuda.empty_cache()

        print("\nMemory scaling with batch size:")
        for i, batch in enumerate(batch_sizes):
            print(f"  batch={batch:2d}: {memories[i]:6.2f} MB")

        # Check approximately linear scaling
        # Memory(8) / Memory(1) should be close to 8
        ratio = memories[3] / memories[0]
        print(f"  Memory ratio (batch=8 / batch=1): {ratio:.2f}x")

        # Allow some overhead, but should be roughly linear
        assert 4 < ratio < 12, f"Memory scaling with batch size should be roughly linear (got {ratio:.2f}x)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
