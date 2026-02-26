"""Benchmark script to compare different attention mechanisms.

Compares StandardAttention and LinearAttention in terms of:
- Time complexity (forward and backward pass)
- Memory usage
- Output quality (comparison with ground truth)
"""

import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from piecewise_linear_attention.core.attention import StandardAttention, LinearAttention


def get_memory_allocated() -> float:
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def benchmark_forward_pass(
    attention: nn.Module,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> Tuple[float, float]:
    """Benchmark forward pass time.

    Args:
        attention: Attention module to benchmark
        Q, K, V: Input tensors
        num_warmup: Number of warmup iterations
        num_runs: Number of timed iterations

    Returns:
        mean_time: Mean execution time in milliseconds
        std_time: Standard deviation of execution time
    """
    device = Q.device

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = attention(Q, K, V)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = attention(Q, K, V)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

    return torch.tensor(times).mean().item(), torch.tensor(times).std().item()


def benchmark_backward_pass(
    attention: nn.Module,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> Tuple[float, float]:
    """Benchmark backward pass time.

    Args:
        attention: Attention module to benchmark
        Q, K, V: Input tensors
        num_warmup: Number of warmup iterations
        num_runs: Number of timed iterations

    Returns:
        mean_time: Mean execution time in milliseconds
        std_time: Standard deviation of execution time
    """
    device = Q.device

    # Warmup
    for _ in range(num_warmup):
        Q_copy = Q.clone().requires_grad_(True)
        K_copy = K.clone().requires_grad_(True)
        V_copy = V.clone().requires_grad_(True)

        output, _ = attention(Q_copy, K_copy, V_copy)
        loss = output.sum()
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        Q_copy = Q.clone().requires_grad_(True)
        K_copy = K.clone().requires_grad_(True)
        V_copy = V.clone().requires_grad_(True)

        output, _ = attention(Q_copy, K_copy, V_copy)
        loss = output.sum()

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

    return torch.tensor(times).mean().item(), torch.tensor(times).std().item()


def compute_approximation_error(
    output_approx: torch.Tensor,
    output_exact: torch.Tensor,
) -> Dict[str, float]:
    """Compute various error metrics between approximate and exact outputs.

    Args:
        output_approx: Output from approximate attention mechanism
        output_exact: Output from standard (exact) attention mechanism

    Returns:
        Dictionary with error metrics (MSE, MAE, relative error, cosine similarity)
    """
    mse = torch.mean((output_approx - output_exact) ** 2).item()
    mae = torch.mean(torch.abs(output_approx - output_exact)).item()

    # Relative error (normalized by output magnitude)
    relative_error = (
        torch.norm(output_approx - output_exact) / torch.norm(output_exact)
    ).item()

    # Cosine similarity (flattened)
    flat_approx = output_approx.flatten()
    flat_exact = output_exact.flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(
        flat_approx.unsqueeze(0), flat_exact.unsqueeze(0)
    ).item()

    return {
        "mse": mse,
        "mae": mae,
        "relative_error": relative_error,
        "cosine_similarity": cosine_sim,
    }


def run_benchmark(
    batch_size: int,
    seq_len: int,
    dim: int,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """Run comprehensive benchmark for a given configuration.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        dim: Embedding dimension
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Dictionary with benchmark results for each attention type
    """
    device_obj = torch.device(device)

    # Create input tensors
    Q = torch.randn(batch_size, seq_len, dim, device=device_obj)
    K = torch.randn(batch_size, seq_len, dim, device=device_obj)
    V = torch.randn(batch_size, seq_len, dim, device=device_obj)

    # Initialize attention modules
    standard_attn = StandardAttention(dim=dim, dropout=0.0).to(device_obj)
    linear_attn = LinearAttention(dim=dim, dropout=0.0).to(device_obj)

    # Get ground truth output from standard attention
    with torch.no_grad():
        output_standard, _ = standard_attn(Q, K, V)

    results = {}

    # Benchmark StandardAttention
    print(f"Benchmarking StandardAttention (n={seq_len}, d={dim})...")
    fwd_mean, fwd_std = benchmark_forward_pass(standard_attn, Q, K, V)
    bwd_mean, bwd_std = benchmark_backward_pass(standard_attn, Q, K, V)

    results["standard"] = {
        "forward_time_ms": fwd_mean,
        "forward_std_ms": fwd_std,
        "backward_time_ms": bwd_mean,
        "backward_std_ms": bwd_std,
        "total_time_ms": fwd_mean + bwd_mean,
        "mse": 0.0,
        "mae": 0.0,
        "relative_error": 0.0,
        "cosine_similarity": 1.0,
    }

    # Benchmark LinearAttention
    print(f"Benchmarking LinearAttention (n={seq_len}, d={dim})...")
    fwd_mean, fwd_std = benchmark_forward_pass(linear_attn, Q, K, V)
    bwd_mean, bwd_std = benchmark_backward_pass(linear_attn, Q, K, V)

    with torch.no_grad():
        output_linear, _ = linear_attn(Q, K, V)

    error_metrics = compute_approximation_error(output_linear, output_standard)

    results["linear"] = {
        "forward_time_ms": fwd_mean,
        "forward_std_ms": fwd_std,
        "backward_time_ms": bwd_mean,
        "backward_std_ms": bwd_std,
        "total_time_ms": fwd_mean + bwd_mean,
        **error_metrics,
    }

    return results


def print_results(results: Dict[str, Dict[str, float]], seq_len: int, dim: int):
    """Pretty print benchmark results.

    Args:
        results: Dictionary with benchmark results
        seq_len: Sequence length
        dim: Embedding dimension
    """
    print(f"\n{'=' * 80}")
    print(f"Benchmark Results (seq_len={seq_len}, dim={dim})")
    print(f"{'=' * 80}\n")

    for attn_type, metrics in results.items():
        print(f"{attn_type.upper()}:")
        print(f"  Forward:  {metrics['forward_time_ms']:8.3f} ± {metrics['forward_std_ms']:.3f} ms")
        print(f"  Backward: {metrics['backward_time_ms']:8.3f} ± {metrics['backward_std_ms']:.3f} ms")
        print(f"  Total:    {metrics['total_time_ms']:8.3f} ms")

        if attn_type != "standard":
            print(f"  Quality Metrics:")
            print(f"    MSE:              {metrics['mse']:.6e}")
            print(f"    MAE:              {metrics['mae']:.6e}")
            print(f"    Relative Error:   {metrics['relative_error']:.6f}")
            print(f"    Cosine Similarity: {metrics['cosine_similarity']:.6f}")

        print()

    # Print speedup
    if "linear" in results and "standard" in results:
        speedup = results["standard"]["total_time_ms"] / results["linear"]["total_time_ms"]
        print(f"LinearAttention Speedup: {speedup:.2f}x")
        print()


def main():
    """Run benchmarks for various configurations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on: {device}\n")

    # Configuration: (batch_size, seq_len, dim)
    configs = [
        (2, 128, 64),
        (2, 256, 64),
        (2, 512, 64),
        (2, 1024, 64),
        (2, 512, 128),
        (2, 512, 256),
    ]

    all_results = []

    for batch_size, seq_len, dim in configs:
        print(f"\nRunning benchmark: batch={batch_size}, seq_len={seq_len}, dim={dim}")
        results = run_benchmark(batch_size, seq_len, dim, device=device)
        print_results(results, seq_len, dim)
        all_results.append((seq_len, dim, results))

    # Summary comparison
    print(f"\n{'=' * 80}")
    print("SUMMARY: Speedup vs Sequence Length")
    print(f"{'=' * 80}\n")
    print(f"{'Seq Len':>10} {'Dim':>6} {'Speedup':>10} {'Relative Error':>15}")
    print("-" * 80)

    for seq_len, dim, results in all_results:
        if "linear" in results and "standard" in results:
            speedup = results["standard"]["total_time_ms"] / results["linear"]["total_time_ms"]
            rel_error = results["linear"]["relative_error"]
            print(f"{seq_len:>10} {dim:>6} {speedup:>9.2f}x {rel_error:>14.6f}")


if __name__ == "__main__":
    main()
