# Benchmarks

Tools for measuring attention mechanisms, split into two layers:

- **Evidence harness** (`harness/`) — reproducible CLIs that emit versioned JSON under [`../results/`](../results/). These are the source of the numbers in the main [README](../README.md).
- **Standalone tools** — an isolated attention benchmark and a memory profiler.

## Evidence harness

Each CLI has a `--smoke` mode (seconds) used by the test suite; drop it for a full run.

```bash
# Forward wall-clock vs sequence length -> results/scaling.json
python harness/run_scaling.py --lengths 512 1024 2048 4096 --out ../results/scaling.json

# Approximation error to exact softmax over a dispersion x length grid -> results/microbench.json
python harness/run_microbench.py --out ../results/microbench.json

# Associative recall, seed-averaged accuracy (long run) -> results/recall.json
python harness/run_recall.py --seeds 0 1 2 3 4 5 6 7 --lengths 64 \
    --steps 12000 --eval-every 1000 --out ../results/recall.json

# Per-query fidelity to softmax on real captured activations vs length
# -> results/real_activation_fidelity.json  (needs the benchmark extras)
python harness/run_real_activation_fidelity.py --out ../results/real_activation_fidelity.json

# Two-panel speed-vs-quality figure from the JSON above (PNG is not committed)
python harness/plot_results.py --out ../results/speed_vs_quality.png
```

Methods compared: `standard`, `linear`, `performer`, `piecewise` (single mean anchor), `piecewise_kmeans`, `piecewise_stride`.

### What the harness shows

- **Speed at one anchor** (`scaling.json`): the single mean-anchor forward is ~4× faster than Performer at `n=4096` and the linear-time methods cross below softmax around `n≈1024` (dim=24, B·H=256, MPS).
- **Anchor-count cost** (`anchor_cost_sweep.json`): the apply is `O(batch·m·n·d²)`, linear in anchor count `m`. At `d=64–128` the single anchor is ~3–5× faster than Performer, but `m=16` is ~3–5× slower and `m=64` is 9–44× slower — so the speed advantage is specific to `m=1`.
- **Approximation quality on synthetic inputs** (`microbench.json`): centroid anchors (mean, k-means) track softmax more closely than positional (stride), linear, or Performer.
- **Fidelity on real activations** (`real_activation_fidelity.json`): on `Q/K/V` captured from pretrained encoders, anchor count is a real accuracy lever — `m=64` cuts relative error ~3× versus `m=1` and is the only piecewise config that beats Nyströmformer at `n=8192`, while `m=1` alone is insufficient there. Opposite to the single-cluster synthetic microbenchmark.
- **Recall** (`recall.json`): the mean anchor solves the task (~0.995, 8 seeds); linear attention fails at chance.

See the main [README](../README.md#results) for the full tables and honest scope (synthetic tasks, small models, MPS timing — not CUDA or real-data end-to-end).

## Standalone tools

### Isolated attention benchmark

```bash
python benchmark_attention.py            # time + error for standard/linear/piecewise
python benchmark_attention.py --causal   # causal masking path
python benchmark_attention.py --scale    # scaling analysis
```

### Memory profiling

```bash
python profile_memory.py                                   # medium preset
python profile_memory.py --preset large
python profile_memory.py --batch 32 --seq-len 4096 --dim 64
```

Presets: `small` | `medium` | `large` | `all`. Memory is `O(d²)` in the apply — constant with sequence length.

## Tests

```bash
pytest ../piecewise_linear_attention/tests/test_evidence_cli.py -v   # harness pipeline
pytest ../piecewise_linear_attention/tests/test_memory_profile.py -v # memory profiler
```

## Dependencies

```bash
uv pip install -e ".[benchmark]"
```
