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

# Op-level attribution of the piecewise_kmeans (m>1) forward -> results/multianchor_profile.json
python harness/run_multianchor_profile.py --dims 64 128 --lengths 512 1024 2048 \
    --anchors 1 4 16 --out ../results/multianchor_profile.json

# Anchor-reuse speedup vs accuracy cost over a drifting-input loop -> results/anchor_reuse.json
python harness/run_anchor_reuse.py --out ../results/anchor_reuse.json

# Epoch-warmup diagnostic on LRA ListOps (needs experiments/lra + a GPU for the real run;
# --smoke self-tests the pipeline on CPU in seconds) -> results/lra/listops/epoch_warmup_diag.json
python harness/run_epoch_warmup_diag.py --device cuda --epochs 4 \
    --methods piecewise piecewise_kmeans_m4 piecewise_kmeans_m4_cached8 \
    --precision bf16 --out ../results/lra/listops/epoch_warmup_diag.json

# Two-panel speed-vs-quality figure from the JSON above (PNG is not committed)
python harness/plot_results.py --out ../results/speed_vs_quality.png
```

Methods compared: `standard`, `linear`, `performer`, `piecewise` (single mean anchor), `piecewise_kmeans`, `piecewise_stride`.

### What the harness shows

- **Speed at one anchor** (`scaling.json`): the single mean-anchor forward is ~4× faster than Performer at `n=4096` and the linear-time methods cross below softmax around `n≈1024` (dim=24, B·H=256, MPS).
- **Anchor-count cost** (`anchor_cost_sweep.json`): the apply is `O(batch·m·n·d²)`, linear in anchor count `m`. At `d=64–128` the single anchor is ~3–5× faster than Performer, but `m=16` is ~3–5× slower and `m=64` is 9–44× slower — so the speed advantage is specific to `m=1`.
- **Where the `m>1` time goes** (`multianchor_profile.json`): op-level attribution at the sweep shapes. The dominant term shifts with `m` — the recomputed k-means Lloyd loop is the largest single term at `m=4` (~30%), while the `O(m·n·d²)` second-moment einsum overtakes it at `m=16` (~40%). The fp32 cast is negligible on CPU.
- **Anchor reuse** (`anchor_reuse.json`): caching the detached centroids (`anchor_refresh_every>1`) recovers the clustering term — ~1.4× forward speedup at `m=4` for a +0.09% fidelity cost — but not the einsum, so it helps at low `m`, not high `m`.
- **Epoch-warmup diagnostic** (`epoch_warmup_diag.json`): on the real LRA ListOps loop, splits each epoch into data / clustering / einsum / other-compute (via a near-zero-cost native phase timer), tracks the CUDA allocator, and runs a per-iteration microscope over the first epoch. Separates the first-few-epochs slowdown into system warmup (page-cache, allocator growth, cuBLAS first-touch) from the *flat* clustering phase — the evidence that the epoch-over-epoch speedup is not k-means "converging" — and compares fresh vs cached anchors epoch-by-epoch.
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
