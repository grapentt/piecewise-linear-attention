"""Simple memory profiling for attention mechanisms.

Usage:
    # Quick run with default config
    python profile_memory.py

    # Custom configuration
    python profile_memory.py --batch 32 --seq-len 2048 --dim 128

    # Save results
    python profile_memory.py --output results.json
"""

import argparse
import gc
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

# Configuration presets
CONFIGS = {
    "small": [(1, 256, 64), (4, 256, 64)],
    "medium": [(8, 1024, 64), (16, 1024, 64)],
    "large": [(32, 2048, 128), (32, 4096, 64)],
}


def get_memory_mb(device: torch.device) -> float:
    """Get current memory usage in MB."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024**2
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        return 0.0


def profile_attention(
    attention_class: type,
    batch: int,
    seq_len: int,
    dim: int,
    device: str = "cpu",
    **kwargs
) -> Dict:
    """Profile memory and time for an attention mechanism."""
    device = torch.device(device)

    # Create attention module
    attention = attention_class(dim=dim, dropout=0.0, **kwargs).to(device)

    # Generate input
    torch.manual_seed(42)
    Q = torch.randn(batch, seq_len, dim, device=device)
    K = torch.randn(batch, seq_len, dim, device=device)
    V = torch.randn(batch, seq_len, dim, device=device)

    # Clear memory
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup
    for _ in range(3):
        _ = attention(Q, K, V)

    # Measure
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_mem = get_memory_mb(device)

    times = []
    for _ in range(10):
        start = time.perf_counter()
        output, _ = attention(Q, K, V)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else get_memory_mb(device)
    avg_time = sum(times) / len(times) * 1000  # ms

    return {
        "time_ms": avg_time,
        "memory_mb": peak_mem,
        "output": output.detach(),
    }


def run_benchmark(configs: List[Tuple[int, int, int]], device: str = "cpu") -> List[Dict]:
    """Run benchmark on multiple configurations."""
    results = []

    for batch, seq_len, dim in configs:
        print(f"\nConfig: batch={batch}, seq_len={seq_len}, dim={dim}")
        print("-" * 60)

        config_results = {
            "batch": batch,
            "seq_len": seq_len,
            "dim": dim,
            "device": device,
            "methods": {}
        }

        # Standard attention (ground truth)
        std_result = profile_attention(StandardAttention, batch, seq_len, dim, device, scale=True)
        config_results["methods"]["StandardAttention"] = {
            "time_ms": std_result["time_ms"],
            "memory_mb": std_result["memory_mb"],
            "speedup": 1.0,
            "error": 0.0,
        }
        print(f"StandardAttention:     {std_result['time_ms']:7.2f} ms | {std_result['memory_mb']:6.0f} MB")

        # Linear attention
        lin_result = profile_attention(LinearAttention, batch, seq_len, dim, device, kernel="relu")
        error_lin = (torch.norm(lin_result["output"] - std_result["output"]) /
                     torch.norm(std_result["output"])).item()
        speedup_lin = std_result["time_ms"] / lin_result["time_ms"]
        config_results["methods"]["LinearAttention"] = {
            "time_ms": lin_result["time_ms"],
            "memory_mb": lin_result["memory_mb"],
            "speedup": speedup_lin,
            "error": error_lin,
        }
        print(f"LinearAttention:       {lin_result['time_ms']:7.2f} ms | {lin_result['memory_mb']:6.0f} MB | {speedup_lin:5.2f}× | {error_lin*100:5.1f}% error")

        # Piecewise attention
        pw_result = profile_attention(PiecewiseAttention, batch, seq_len, dim, device, scale=True)
        error_pw = (torch.norm(pw_result["output"] - std_result["output"]) /
                    torch.norm(std_result["output"])).item()
        speedup_pw = std_result["time_ms"] / pw_result["time_ms"]
        config_results["methods"]["PiecewiseAttention"] = {
            "time_ms": pw_result["time_ms"],
            "memory_mb": pw_result["memory_mb"],
            "speedup": speedup_pw,
            "error": error_pw,
        }
        print(f"PiecewiseAttention:    {pw_result['time_ms']:7.2f} ms | {pw_result['memory_mb']:6.0f} MB | {speedup_pw:5.2f}× | {error_pw*100:5.1f}% error ✅")

        results.append(config_results)

    return results


def save_results(results: List[Dict], output_file: str):
    """Save results to JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple memory profiling for attention mechanisms")

    # Configuration
    parser.add_argument("--preset", choices=["small", "medium", "large", "all"], default="medium",
                        help="Use preset configurations")
    parser.add_argument("--batch", type=int, help="Custom batch size")
    parser.add_argument("--seq-len", type=int, help="Custom sequence length")
    parser.add_argument("--dim", type=int, help="Custom embedding dimension")

    # Execution
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Device to run on")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    args = parser.parse_args()

    # Determine configurations
    if args.batch and args.seq_len and args.dim:
        configs = [(args.batch, args.seq_len, args.dim)]
    elif args.preset == "all":
        configs = CONFIGS["small"] + CONFIGS["medium"] + CONFIGS["large"]
    else:
        configs = CONFIGS[args.preset]

    print("=" * 80)
    print("MEMORY PROFILING")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Configurations: {len(configs)}")

    # Run benchmark
    results = run_benchmark(configs, args.device)

    # Save results
    if args.output:
        save_results(results, args.output)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
