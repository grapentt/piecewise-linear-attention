# Benchmarks

Simple benchmarking tools for attention mechanisms.

## Quick Start

### Time & Accuracy
```bash
uv run python benchmark.py
```

### Memory Profiling
```bash
# Default (medium scale)
uv run python profile_memory.py

# Custom config
uv run python profile_memory.py --batch 32 --seq-len 4096 --dim 64

# Preset configurations
uv run python profile_memory.py --preset large
```

**Presets**: `small` | `medium` | `large` | `all`

## Output Example

```
Config: batch=32, seq_len=4096, dim=64
------------------------------------------------------------
StandardAttention:     180.02 ms |   6085 MB
LinearAttention:         5.84 ms |   6085 MB | 30.81× |  71.4% error
PiecewiseAttention:      5.59 ms |   6085 MB | 32.20× |  51.7% error ✅
```

## Key Results

**Very large scale** (batch=64, n=4096, dim=64 - 16.8M elements):
- **84× speedup** with **52% error** (PiecewiseAttention)
- 20pp better accuracy than LinearAttention
- Memory: O(d²) - constant with sequence length
- Speedup scales with batch size

See main [README.md](../README.md) for full results.

## Tests

```bash
uv run pytest piecewise_linear_attention/tests/test_memory_profile.py -v
```

## Dependencies

```bash
uv pip install -e ".[benchmark]"
```
