# Piecewise Linear Attention

> A deterministic, linear-time attention that approximates softmax by first-order Taylor expansion around representative "anchor" queries.

## Overview

**PiecewiseAttention** approximates softmax attention in linear time. Instead of the full `O(n²·d)` softmax, it computes exact softmax attention and its analytic Jacobian at a small number of *anchor* queries, then linearizes every query around its nearest anchor:

```
output[i] ≈ A(a) + J(a) · (q_i − a)
```

where `a` is the anchor for query `i`, `A(a)` is exact softmax attention at the anchor, and `J(a)` is the analytic Jacobian of the attention map. The apply is fused so that no per-query `d×d` Jacobian is ever materialized, giving `O(n·d²)` cost — linear in sequence length.

**What the evidence shows** (all numbers reproduce from the versioned files in [`results/`](results/); see [Reproducing the evidence](#reproducing-the-evidence) for the exact commands and operating points):

- **Faster than Performer at long sequences — at one anchor.** At `n=4096` (dim=24, B·H=256, MPS) the single mean-anchor forward is **~4× faster than Performer** and ~44× faster than exact softmax (12.4 ms vs 48.6 ms vs 552.6 ms). The linear-time methods cross below softmax around `n≈1024`. This speed holds for `m=1`; because the apply cost is linear in anchor count, more anchors erode and then reverse the advantage (see [the anchor-count cost trade-off](#the-anchor-count-cost-trade-off)).
- **Closer to softmax than Performer on synthetic inputs.** On the approximation microbenchmark the mean-anchor's mean relative error to exact softmax is **0.33 vs Performer's 1.70** — roughly 5× closer on average, and ~10× closer when queries are clustered. On *real* trained-model activations the single anchor is weaker and needs more anchors to lead (see [Fidelity on real activations](#fidelity-on-real-activations)).
- **Solves a task that linear attention cannot.** On synthetic associative recall (`n=64`, 12k steps, 8 seeds) the mean anchor reaches **0.995** accuracy (exact softmax = 1.000), while kernel/linear attention fails at chance (0.126).

**Honest scope.** These results are on synthetic benchmarks and captured activations with small models, timed on Apple Silicon (MPS) / CPU — not CUDA or real-data end-to-end training. At short sequences (`n<1024`) the method is *slower* than softmax; the speed win is asymptotic and holds at one anchor. The configuration that is most accurate on real activations (`m=64`) is currently several-to-many times slower than Performer — reconciling high-anchor fidelity with the single-anchor speed is the central open question. Extending to real tasks, realistic `dim`, and long context on GPU is ongoing work (see the open issues).

See [THEORY.md](THEORY.md) for the derivation.

## Key idea

Softmax attention `A(q) = Σ_i softmax(⟨k_i, q⟩/√d)·v_i` is smooth in `q`, so near an anchor `a` it is well-approximated by its first-order Taylor expansion. The method:

1. **Selects anchors** from the queries via a pluggable strategy (default: the single mean/centroid; also k-means or strided positions).
2. **Computes exact attention and the analytic Jacobian** at each anchor.
3. **Linearizes** each query around its nearest anchor: `A(q) ≈ A(a) + J(a)·(q − a)`.

The first-order truncation error grows with `‖q − a‖²`, so a good anchor keeps every query close to its expansion point. The **single mean-centroid anchor** minimizes the mean squared query-to-anchor distance by construction, which makes it the fastest configuration (one anchor, no routing) and the most accurate first-order one *on synthetic inputs with a single query cluster*. On real trained-model activations the picture is different: query distributions are less concentrated, and approximation fidelity improves substantially with more anchors — see [Fidelity on real activations](#fidelity-on-real-activations). Anchor count is therefore a genuine accuracy/cost knob, not a free parameter.

## Installation

### Using uv (recommended)

```bash
uv pip install -e .                                  # core
uv pip install -e ".[dev]"                           # + dev tooling
uv pip install -e ".[experiments]"                   # + experiment deps
uv pip install -e ".[dev,benchmark,experiments,docs]" # everything
```

### Using pip

```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[experiments]"
```

## Quick start

### Non-causal attention

```python
import torch
from piecewise_linear_attention import PiecewiseAttention

# Default: single mean-centroid anchor — fastest and most accurate first-order config.
attention = PiecewiseAttention(dim=512)

batch_size, seq_len, dim = 2, 1024, 512
Q = torch.randn(batch_size, seq_len, dim)
K = torch.randn(batch_size, seq_len, dim)
V = torch.randn(batch_size, seq_len, dim)

output, anchors = attention(Q, K, V)
print(output.shape)  # (2, 1024, 512)
```

### Choosing an anchor strategy (multi-anchor)

More anchors can converge faster on some tasks at higher cost. Strategies are pluggable; the mean centroid is the recommended default.

```python
from piecewise_linear_attention.core.anchors import KMeansAnchor, StrideAnchor

# Centroid anchors on query clusters (minimize query-to-anchor distance).
attn_kmeans = PiecewiseAttention(dim=512, anchor_strategy=KMeansAnchor(k=4, iters=3))

# Positional anchors at fixed sequence strides (cheapest selection; a baseline).
attn_stride = PiecewiseAttention(dim=512, anchor_strategy=StrideAnchor(k=4))
```

Multi-anchor mode is non-causal only.

### Causal attention (language modeling)

The causal path uses a single anchor and a cumsum formulation for `O(n·d²)` masking.

```python
# Training: mean of all queries as the anchor.
attention_train = PiecewiseAttention(dim=512, causal=True, causal_pseudo_query="mean")

# Autoregressive inference: first query as the anchor (no future leakage).
attention_infer = PiecewiseAttention(dim=512, causal=True, causal_pseudo_query="first")

output, _ = attention_train(Q, K, V)
```

### Building by name

All mechanisms are available through a registry for benchmarks and experiments:

```python
from piecewise_linear_attention.core.registry import build_attention, available_attention_types

print(available_attention_types())
# ['linear', 'performer', 'piecewise', 'piecewise_kmeans', 'piecewise_stride', 'standard']

attn = build_attention("piecewise", dim=64, dropout=0.0, causal=False)
```

## Try it on Google Colab

Run experiments with a free GPU — no installation needed:

| Experiment | Dataset | Notebook |
|------------|---------|----------|
| **Translation** | Multi30K (German→English) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grapentt/piecewise-linear-attention/blob/main/experiments/translation/colab_translation.ipynb) |
| **Vision** | CIFAR-10 (ViT) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grapentt/piecewise-linear-attention/blob/main/experiments/vision/colab_vision.ipynb) |
| **BERT** | SST-2 (Sentiment) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grapentt/piecewise-linear-attention/blob/main/benchmarks/colab_benchmark.ipynb) |
| **Benchmark** | Performance analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grapentt/piecewise-linear-attention/blob/main/benchmarks/colab_benchmark.ipynb) |

## Results

All figures below reproduce from the versioned JSON in [`results/`](results/) via the harness CLIs in [`benchmarks/harness/`](benchmarks/harness/). Timings are on Apple Silicon (MPS), PyTorch 2.10; treat them as relative comparisons on this backend, not absolute CUDA numbers.

### Speed — forward wall-clock vs sequence length

`dim=24`, `B·H=256`, mean over 50 forwards after warmup ([`results/scaling.json`](results/scaling.json)):

| n | softmax | Performer | piecewise (mean) | speedup vs Performer |
|---|---|---|---|---|
| 512 | 7.61 ms | 5.76 ms | 1.48 ms | 3.9× |
| 1024 | 30.77 ms | 11.82 ms | 2.85 ms | 4.1× |
| 2048 | 124.58 ms | 23.52 ms | 5.62 ms | 4.2× |
| 4096 | 552.62 ms | 48.59 ms | 12.43 ms | 3.9× |

The linear-time methods overtake softmax around `n≈1024`; below that, softmax's small constant wins.

### Approximation quality — relative error to exact softmax

`dim=64`, averaged over the dispersion×length grid ([`results/microbench.json`](results/microbench.json); lower is better):

| method | mean rel-err |
|---|---|
| piecewise (k-means, m=4) | 0.31 |
| **piecewise (mean, m=1)** | **0.33** |
| piecewise (stride, m=4) | 0.56 |
| linear (ELU) | 0.84 |
| Performer | 1.70 |

Centroid placement (mean, k-means) beats positional placement (stride); on this synthetic grid the single mean anchor nearly matches 4-way k-means at a quarter of the anchor cost. This parity is specific to synthetic inputs — on real activations (below) fidelity keeps improving with anchor count well past `m=4`.

### Task capability — associative recall

`n=64`, 12k steps, 8 seeds ([`results/recall.json`](results/recall.json); chance = 0.0625):

| method | accuracy (mean) |
|---|---|
| softmax (exact) | 1.000 |
| piecewise (stride, m=4) | 0.997 |
| **piecewise (mean, m=1)** | **0.995** |
| piecewise (k-means, m=4) | 0.984 |
| Performer | 0.981 |
| linear | 0.126 (fails) |

Linear/kernel attention is the only method that fails recall. Performer's occasional *early* accuracy edge over softmax during training is a convergence-speed transient (it leads by up to +0.64 accuracy around step 4k, is overtaken by ~step 6k, and plateaus below exact softmax at convergence) — not a capability edge.

### Fidelity on real activations

The synthetic microbenchmark above uses inputs with a single tight query cluster, which flatters the single-anchor approximation. To probe the harder, realistic regime, this benchmark measures per-query relative error to exact softmax on `Q/K/V` captured from pretrained encoders (MiniLM, `d=32`; bert-tiny, `d=64`) across sequence lengths `n ∈ {512, 2048, 8192}` ([`results/real_activation_fidelity.json`](results/real_activation_fidelity.json)). Worst-layer mean relative error at `n=8192` (lower is better):

| method | rel-err @ n=8192 |
|---|---|
| piecewise (k-means, m=64) | **0.24** |
| piecewise (k-means, m=16) | 0.43 |
| Nyströmformer (256 landmarks) | 0.52 |
| piecewise (mean, m=1) | 0.67 |
| Nyströmformer (64 landmarks) | 0.67 |
| Performer (512 features) | 0.94 |

Two findings, both opposite to the synthetic microbenchmark:

- **Anchor count is a real accuracy lever on real data.** `m=64` cuts relative error ~3× versus `m=1` and is the only piecewise configuration that clearly beats Nyströmformer; on the cleaner bert-tiny layer it reaches near-exact fidelity (`m=1: 0.59 → m=64: 0.07`).
- **The single mean anchor is *not* sufficient here.** At `m=1` piecewise sits above a 0.5 relative-error ceiling at `n=8192` and loses to Nyströmformer — unlike on synthetic inputs where `m=1` was the most accurate first-order config. Fidelity is otherwise flat-to-improving with length (measured query dispersion `E‖q−a‖²` barely moves with `n`).

**Honest scope of this probe.** Because no available offline checkpoint exceeds 512 positions, the `n ∈ {2048, 8192}` inputs are built by concatenating spans from *distinct* documents (a cross-document proxy, flagged `is_real=false` in the JSON), which if anything *inflates* dispersion and makes the task harder, not easier. These are relative-error comparisons on captured activations, not end-to-end task accuracy.

### The anchor-count cost trade-off

The accuracy gain from more anchors is not free: the apply is `O(batch · m · n · d²)`, linear in anchor count `m`. Forward wall-clock relative to Performer at realistic head dims ([`results/anchor_cost_sweep.json`](results/anchor_cost_sweep.json), CPU, `B·H=64`; a ratio `<1` means piecewise is faster):

| config | vs Performer, d=64 | vs Performer, d=128 |
|---|---|---|
| piecewise (mean, m=1) | **0.17–0.21× (≈5× faster)** | **0.34–0.36× (≈3× faster)** |
| piecewise (k-means, m=16) | 2.7–3.7× slower | 4.9–5.2× slower |
| piecewise (k-means, m=64) | 9–11× slower | 16–44× slower |

So the two axes pull apart: the configuration that is fastest (`m=1`, several times faster than Performer) is not the most accurate on real activations, and the configuration that is most accurate (`m=64`) is currently several-to-many times slower than Performer. Closing this gap — keeping high-anchor fidelity while cutting the `d²` apply cost (e.g. a diagonal or low-rank Jacobian) — is the central open efficiency question, and remains to be validated on GPU at these head dims.

A two-panel speed-vs-quality figure can be regenerated with `python benchmarks/harness/plot_results.py` (the PNG is not committed, matching the repo's image policy).

## Reproducing the evidence

```bash
# Speed vs sequence length (writes results/scaling.json)
python benchmarks/harness/run_scaling.py --lengths 512 1024 2048 4096 --out results/scaling.json

# Approximation error vs dispersion and length (writes results/microbench.json)
python benchmarks/harness/run_microbench.py --out results/microbench.json

# Associative recall, seed-averaged (writes results/recall.json; long run)
python benchmarks/harness/run_recall.py --seeds 0 1 2 3 4 5 6 7 --lengths 64 \
    --steps 12000 --eval-every 1000 --out results/recall.json

# Per-query fidelity to softmax on real captured activations vs length
# (writes results/real_activation_fidelity.json; needs the benchmark extras)
python benchmarks/harness/run_real_activation_fidelity.py \
    --out results/real_activation_fidelity.json

# Speed-vs-quality figure from the JSON above
python benchmarks/harness/plot_results.py --out results/speed_vs_quality.png
```

Each CLI has a `--smoke` mode (tiny, seconds) used by the test suite to guard the pipeline.

## Attention mechanisms

- **PiecewiseAttention** — first-order Taylor approximation around anchor queries; `O(n·d²)`. Pluggable anchor strategies (`MeanAnchor` default, `KMeansAnchor`, `StrideAnchor`); causal support via cumsum.
- **StandardAttention** — exact softmax baseline; `O(n²·d)`.
- **LinearAttention** — kernel-feature-map linear attention (ELU/ReLU/softplus).
- **PerformerAttention** — FAVOR+ random-feature softmax estimator; unbiased but stochastic.

## Development

```bash
# Tests
pytest
pytest --cov=piecewise_linear_attention --cov-report=html

# Formatting / types
black piecewise_linear_attention/ benchmarks/
isort piecewise_linear_attention/ benchmarks/
mypy piecewise_linear_attention/
```

## Algorithm details

### Non-causal (multi-anchor)

```
1. Select m anchors from Q via the anchor strategy (default: mean/centroid, m=1).
2. At each anchor a_j compute the exact attention output o_j = A(a_j) and the
   analytic-Jacobian factors (M_j = Σ_n α_jn v_n k_nᵀ, weighted keys wk_j).
3. Assign each query to its nearest anchor (argmin distance; non-differentiable).
4. Apply the fused first-order expansion, distributing the matmul over the
   rank-1 Jacobian so no per-query d×d matrix is formed:
     out_i = o_j + s·(M_j·Δq) − s·o_j·(wk_jᵀ·Δq),   Δq = q_i − a_j,  s = 1/√d
```

**Complexity**: `O(batch · m · n · d²)` — linear in `n` with small constant `m`, vs `O(batch · n² · d)` for softmax.

### Causal (cumsum)

Linearizing the exponential kernel *before* summation lets cumulative sums carry the per-position Jacobian:

```
1. Anchor q̄ = mean(Q) (train) or q_0 (inference).
2. Anchor weights σ_j = exp(s · q̄·k_j).
3. Cumsum terms S^A=Σσ_j v_j, S^B=Σσ_j (k_j⊗v_j), S^C=Σσ_j, S^D=Σσ_j k_j.
4. output_i = (S^A_i + s·S^B_i·(q_i−q̄)) / (S^C_i + s·S^D_iᵀ·(q_i−q̄))
```

**Complexity**: `O(n·d²)` time — the same class as linear attention, computing the exact anchor Jacobian.

## Citation

```bibtex
@software{piecewise_linear_attention,
  title = {Piecewise Linear Attention: Deterministic Linear-Time Attention via Anchor Taylor Expansion},
  year = {2026},
  url = {https://github.com/grapentt/piecewise-linear-attention}
}
```

## References

- Vaswani et al. (2017). "Attention is All You Need"
- Katharopoulos et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
- Choromanski et al. (2021). "Rethinking Attention with Performers"
- See [THEORY.md](THEORY.md) for the complete derivation and references.

## License

MIT License — see LICENSE file for details.
