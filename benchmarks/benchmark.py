"""Comprehensive benchmark comparing all attention mechanisms."""

import time
import torch

from piecewise_linear_attention import (
    StandardAttention,
    LinearAttention,
    PiecewiseAttention,
)


def benchmark_attention(attention_fn, Q, K, V, num_warmup=5, num_runs=20):
    """Benchmark an attention function."""
    # Warmup
    for _ in range(num_warmup):
        _ = attention_fn(Q, K, V)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output = attention_fn(Q, K, V)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_time": sum(times) / len(times),
        "std_time": torch.tensor(times).std().item(),
    }


def compute_error(output_approx, output_standard):
    """Compute approximation error."""
    diff = output_approx - output_standard
    rel_error = torch.norm(diff).item() / torch.norm(output_standard).item()

    batch_errors = []
    for b in range(output_standard.size(0)):
        batch_diff = output_approx[b] - output_standard[b]
        batch_rel = torch.norm(batch_diff).item() / torch.norm(output_standard[b]).item()
        batch_errors.append(batch_rel)

    return {
        "relative_error": rel_error,
        "mean_batch_error": sum(batch_errors) / len(batch_errors),
    }


def run_benchmark():
    """Run comprehensive benchmark comparing all attention mechanisms."""

    print("="*80)
    print("Piecewise Linear Attention Benchmark")
    print("="*80)
    print()
    print("Comparing StandardAttention, LinearAttention, and PiecewiseAttention")
    print("LinearAttention uses proper kernel feature maps (ELU, ReLU)")
    print()

    configs = [
        (1, 256, 64),
        (1, 512, 64),
        (1, 1024, 64),
        (1, 2048, 64),
        (4, 1024, 64),
        (8, 1024, 64),
        (16, 1024, 64),
        (8, 2048, 128),
    ]

    for batch, seq_len, dim in configs:
        print("="*80)
        print(f"Config: batch={batch}, seq_len={seq_len}, dim={dim}")
        print("="*80)

        # Generate data
        torch.manual_seed(42)
        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        # 1. Standard Attention (ground truth)
        print("\n[1/5] Standard Attention (ground truth)")
        print("-"*40)
        standard_attn = StandardAttention(dim=dim, dropout=0.0, scale=True)
        standard_fn = lambda Q, K, V: standard_attn(Q, K, V)[0]
        standard_stats = benchmark_attention(standard_fn, Q, K, V)
        standard_output = standard_fn(Q, K, V)

        print(f"  Time: {standard_stats['mean_time']*1000:.3f} ms")

        # 2. Linear Attention (ELU kernel)
        print("\n[2/4] LinearAttention with ELU kernel")
        print("-"*40)
        linear_elu = LinearAttention(dim=dim, kernel="elu", dropout=0.0)
        linear_elu_fn = lambda Q, K, V: linear_elu(Q, K, V)[0]
        linear_elu_stats = benchmark_attention(linear_elu_fn, Q, K, V)
        linear_elu_output = linear_elu_fn(Q, K, V)
        linear_elu_error = compute_error(linear_elu_output, standard_output)

        print(f"  Time: {linear_elu_stats['mean_time']*1000:.3f} ms")
        print(f"  Speedup: {standard_stats['mean_time']/linear_elu_stats['mean_time']:.2f}x")
        print(f"  Relative Error: {linear_elu_error['relative_error']:.4f} ({linear_elu_error['relative_error']*100:.1f}%)")

        # 3. Linear Attention (ReLU kernel)
        print("\n[3/4] LinearAttention with ReLU kernel")
        print("-"*40)
        linear_relu = LinearAttention(dim=dim, kernel="relu", dropout=0.0)
        linear_relu_fn = lambda Q, K, V: linear_relu(Q, K, V)[0]
        linear_relu_stats = benchmark_attention(linear_relu_fn, Q, K, V)
        linear_relu_output = linear_relu_fn(Q, K, V)
        linear_relu_error = compute_error(linear_relu_output, standard_output)

        print(f"  Time: {linear_relu_stats['mean_time']*1000:.3f} ms")
        print(f"  Speedup: {standard_stats['mean_time']/linear_relu_stats['mean_time']:.2f}x")
        print(f"  Relative Error: {linear_relu_error['relative_error']:.4f} ({linear_relu_error['relative_error']*100:.1f}%)")

        # 4. Piecewise Attention (RECOMMENDED)
        print("\n[4/4] PiecewiseAttention (RECOMMENDED)")
        print("-"*40)
        piecewise = PiecewiseAttention(dim=dim, dropout=0.0, scale=True)
        piecewise_fn = lambda Q, K, V: piecewise(Q, K, V)[0]
        piecewise_stats = benchmark_attention(piecewise_fn, Q, K, V)
        piecewise_output = piecewise_fn(Q, K, V)
        piecewise_error = compute_error(piecewise_output, standard_output)

        print(f"  Time: {piecewise_stats['mean_time']*1000:.3f} ms")
        print(f"  Speedup: {standard_stats['mean_time']/piecewise_stats['mean_time']:.2f}x")
        print(f"  Relative Error: {piecewise_error['relative_error']:.4f} ({piecewise_error['relative_error']*100:.1f}%)")

        # Comparison summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Method':<30} {'Time (ms)':<12} {'Speedup':<10} {'Error':<12}")
        print("-"*80)

        methods = [
            ("StandardAttention", standard_stats['mean_time'], 1.0, 0.0),
            ("LinearAttention (ELU)", linear_elu_stats['mean_time'],
             standard_stats['mean_time']/linear_elu_stats['mean_time'],
             linear_elu_error['relative_error']),
            ("LinearAttention (ReLU)", linear_relu_stats['mean_time'],
             standard_stats['mean_time']/linear_relu_stats['mean_time'],
             linear_relu_error['relative_error']),
            ("PiecewiseAttention", piecewise_stats['mean_time'],
             standard_stats['mean_time']/piecewise_stats['mean_time'],
             piecewise_error['relative_error']),
        ]

        for name, time_val, speedup, error in methods:
            print(f"{name:<30} {time_val*1000:<12.3f} {speedup:<10.2f} {error:<12.4f}")

        print()


if __name__ == "__main__":
    run_benchmark()
