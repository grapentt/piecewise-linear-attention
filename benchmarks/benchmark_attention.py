"""Comprehensive benchmark for standalone attention mechanisms.

This script benchmarks StandardAttention, LinearAttention, and PiecewiseAttention
in isolation (without transformer overhead) to measure pure attention performance.

Usage:
    python benchmarks/benchmark_attention.py
    python benchmarks/benchmark_attention.py --causal  # Benchmark causal masking
    python benchmarks/benchmark_attention.py --scale   # Scaling analysis
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from piecewise_linear_attention import (
    StandardAttention,
    LinearAttention,
    PiecewiseAttention,
)


def benchmark_attention(
    attention_fn,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> Dict[str, float]:
    """Benchmark an attention function.

    Args:
        attention_fn: Attention function to benchmark
        Q, K, V: Query, key, value tensors
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations

    Returns:
        Dictionary with mean_time and std_time in milliseconds
    """
    # Warmup
    for _ in range(num_warmup):
        _ = attention_fn(Q, K, V)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output, _ = attention_fn(Q, K, V)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_time_ms": sum(times) / len(times),
        "std_time_ms": torch.tensor(times).std().item(),
        "min_time_ms": min(times),
        "max_time_ms": max(times),
    }


def compute_error(output_approx: torch.Tensor, output_standard: torch.Tensor) -> Dict[str, float]:
    """Compute approximation error metrics.

    Args:
        output_approx: Output from approximate attention
        output_standard: Output from standard attention

    Returns:
        Dictionary with various error metrics
    """
    diff = output_approx - output_standard

    # Global metrics
    rel_error = torch.norm(diff).item() / torch.norm(output_standard).item()
    mean_abs_error = diff.abs().mean().item()
    max_abs_error = diff.abs().max().item()

    # Per-batch metrics
    batch_errors = []
    for b in range(output_standard.size(0)):
        batch_diff = output_approx[b] - output_standard[b]
        batch_rel = torch.norm(batch_diff).item() / torch.norm(output_standard[b]).item()
        batch_errors.append(batch_rel)

    return {
        "relative_error": rel_error,
        "mean_batch_error": sum(batch_errors) / len(batch_errors),
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
    }


def run_basic_benchmark(causal: bool = False, num_runs: int = 20) -> List[Dict]:
    """Run basic benchmark across multiple configurations.

    Args:
        causal: Whether to use causal masking
        num_runs: Number of runs per configuration

    Returns:
        List of result dictionaries
    """
    print("=" * 80)
    print(f"{'CAUSAL' if causal else 'NON-CAUSAL'} ATTENTION BENCHMARK")
    print("=" * 80)
    print()
    print("Comparing StandardAttention, LinearAttention, and PiecewiseAttention")
    print(f"Mode: {'Causal masking' if causal else 'Bidirectional'}")
    print()

    configs = [
        (1, 256, 64),
        (1, 512, 64),
        (1, 1024, 64),
        (4, 512, 64),
        (4, 1024, 64),
        (16, 1024, 64),
        (4, 2048, 64),
        (16, 2048, 64),
    ]

    results = []

    for batch, seq_len, dim in configs:
        print("=" * 80)
        print(f"Config: batch={batch}, seq_len={seq_len}, dim={dim}")
        print("=" * 80)

        # Generate data
        torch.manual_seed(42)
        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        # 1. Standard Attention (baseline)
        print("\n1. StandardAttention (baseline)")
        attn_std = StandardAttention(dim=dim, dropout=0.0, scale=True, causal=causal)
        std_results = benchmark_attention(attn_std, Q, K, V, num_runs=num_runs)
        output_std, _ = attn_std(Q, K, V)

        print(f"   Time: {std_results['mean_time_ms']:.2f} ± {std_results['std_time_ms']:.2f} ms")
        print(f"   Speedup: 1.00×")

        # 2. Linear Attention (ELU kernel)
        print("\n2. LinearAttention (ELU kernel)")
        attn_lin = LinearAttention(dim=dim, dropout=0.0, kernel_type="elu", causal=causal)
        lin_results = benchmark_attention(attn_lin, Q, K, V, num_runs=num_runs)
        output_lin, _ = attn_lin(Q, K, V)
        lin_errors = compute_error(output_lin, output_std)

        lin_speedup = std_results['mean_time_ms'] / lin_results['mean_time_ms']
        print(f"   Time: {lin_results['mean_time_ms']:.2f} ± {lin_results['std_time_ms']:.2f} ms")
        print(f"   Speedup: {lin_speedup:.2f}×")
        print(f"   Error: {lin_errors['relative_error']:.4f} (relative)")

        # 3. Piecewise Attention
        print("\n3. PiecewiseAttention")
        pseudo_query_strategy = "mean" if causal else None
        attn_pw = PiecewiseAttention(
            dim=dim,
            dropout=0.0,
            scale=True,
            causal=causal,
            causal_pseudo_query=pseudo_query_strategy if causal else "mean"
        )
        pw_results = benchmark_attention(attn_pw, Q, K, V, num_runs=num_runs)
        output_pw, _ = attn_pw(Q, K, V)
        pw_errors = compute_error(output_pw, output_std)

        pw_speedup = std_results['mean_time_ms'] / pw_results['mean_time_ms']
        print(f"   Time: {pw_results['mean_time_ms']:.2f} ± {pw_results['std_time_ms']:.2f} ms")
        print(f"   Speedup: {pw_speedup:.2f}×")
        print(f"   Error: {pw_errors['relative_error']:.4f} (relative)")

        # Summary table
        print("\n" + "-" * 80)
        print(f"{'Method':<25} {'Time (ms)':<15} {'Speedup':<12} {'Error':<12}")
        print("-" * 80)
        print(f"{'StandardAttention':<25} {std_results['mean_time_ms']:>6.2f} ± {std_results['std_time_ms']:>4.2f} {'1.00×':>11} {'0.0000':>11}")
        print(f"{'LinearAttention (ELU)':<25} {lin_results['mean_time_ms']:>6.2f} ± {lin_results['std_time_ms']:>4.2f} {lin_speedup:>10.2f}× {lin_errors['relative_error']:>11.4f}")
        print(f"{'PiecewiseAttention':<25} {pw_results['mean_time_ms']:>6.2f} ± {pw_results['std_time_ms']:>4.2f} {pw_speedup:>10.2f}× {pw_errors['relative_error']:>11.4f}")
        print("-" * 80)
        print()

        # Store results
        results.append({
            "batch": batch,
            "seq_len": seq_len,
            "dim": dim,
            "causal": causal,
            "standard": std_results,
            "linear": {**lin_results, **lin_errors, "speedup": lin_speedup},
            "piecewise": {**pw_results, **pw_errors, "speedup": pw_speedup},
        })

    return results


def run_scaling_analysis(causal: bool = False) -> List[Dict]:
    """Run scaling analysis to show how speedup increases with sequence length.

    Args:
        causal: Whether to use causal masking

    Returns:
        List of result dictionaries
    """
    print("=" * 80)
    print(f"SCALING ANALYSIS - {'CAUSAL' if causal else 'NON-CAUSAL'}")
    print("=" * 80)
    print()
    print("How does speedup scale with sequence length?")
    print()

    # Fixed batch and dim, vary sequence length
    batch = 16
    dim = 64
    seq_lens = [64, 128, 256, 512, 1024, 2048, 4096]

    results = []

    print(f"{'Seq Len':<10} {'Standard (ms)':<15} {'Linear (ms)':<15} {'Piecewise (ms)':<15} {'Linear Speedup':<15} {'PW Speedup':<15}")
    print("-" * 95)

    for seq_len in seq_lens:
        torch.manual_seed(42)
        Q = torch.randn(batch, seq_len, dim)
        K = torch.randn(batch, seq_len, dim)
        V = torch.randn(batch, seq_len, dim)

        # Benchmark each
        attn_std = StandardAttention(dim=dim, causal=causal)
        attn_lin = LinearAttention(dim=dim, causal=causal)
        attn_pw = PiecewiseAttention(dim=dim, causal=causal)

        std_time = benchmark_attention(attn_std, Q, K, V, num_runs=10)['mean_time_ms']
        lin_time = benchmark_attention(attn_lin, Q, K, V, num_runs=10)['mean_time_ms']
        pw_time = benchmark_attention(attn_pw, Q, K, V, num_runs=10)['mean_time_ms']

        lin_speedup = std_time / lin_time
        pw_speedup = std_time / pw_time

        print(f"{seq_len:<10} {std_time:>12.2f} {lin_time:>14.2f} {pw_time:>14.2f} {lin_speedup:>14.2f}× {pw_speedup:>14.2f}×")

        results.append({
            "seq_len": seq_len,
            "batch": batch,
            "dim": dim,
            "causal": causal,
            "standard_time_ms": std_time,
            "linear_time_ms": lin_time,
            "piecewise_time_ms": pw_time,
            "linear_speedup": lin_speedup,
            "piecewise_speedup": pw_speedup,
        })

    print("-" * 95)
    print()
    print(f"✨ Key Insight: Speedup increases from {results[0]['piecewise_speedup']:.1f}× to {results[-1]['piecewise_speedup']:.1f}× as n grows!")
    print()

    return results


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Benchmark attention mechanisms")
    parser.add_argument("--causal", action="store_true", help="Benchmark with causal masking")
    parser.add_argument("--scale", action="store_true", help="Run scaling analysis")
    parser.add_argument("--num-runs", type=int, default=20, help="Number of runs per config")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if args.scale:
        results = run_scaling_analysis(causal=args.causal)
    else:
        results = run_basic_benchmark(causal=args.causal, num_runs=args.num_runs)

    # Save results if requested
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {save_path}")

    print("=" * 80)
    print("✅ Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
