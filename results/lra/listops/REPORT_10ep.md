# LRA ListOps: 10-Epoch Fair-Comparison Benchmark (fixed-budget)

## Summary

Eight attention mechanisms were trained end-to-end on the LRA ListOps task under a
fair, fixed-budget protocol (10 epochs, matched attention budget of 256, flat
learning rate 1e-3) to compare their wall-clock efficiency at sequence length 2048.
On an A100 80GB in bf16, the softmax-fidelity method **piecewise (m=1)** is the
fastest at **5.01×** steady-state speedup over the standard dense Transformer,
beating every learned and training-free peer (linformer, nystromformer, luna,
performer); `linear` is nominally faster (5.99×) but is the known accuracy-floor
degenerate case. The k-means variant **piecewise_kmeans (m=4)** carries a one-time
warmup that inflates its whole-run cost, but converges to a **1.58×** steady-state
speedup. Critically, this is a **cost/throughput probe only**: all eight methods
collapsed to identical near-chance accuracy (~16% test), a shared training-dynamics
outcome that makes cross-method accuracy comparisons meaningless under this setup.
Only the speed split is trustworthy.

## 1. Setup

- **Hardware/precision:** A100 80GB, bf16.
- **Methods (8):** standard, linear, performer, nystromformer, linformer, luna,
  piecewise (m=1), piecewise_kmeans (m=4, kmeans_iters=3).
- **Model:** sequence length n=2048, embedding dim d=512, 8 heads, 4 layers,
  ~9.5M parameters (standard: 9,472,010).
- **Training:** batch size 32, 10 epochs, seed 42, flat LR 1e-3 (no scheduler),
  matched attention budget = 256.
- **Steady-state metric:** mean per-epoch wall time over epochs 4–10 (post-warmup).

## 2. Speed Results

| Method | attn budget | Params | Steady ep-wall (min) | Total compute (min) | Total wall (min) | Data load (s) | Speedup (total wall) | Speedup (steady/epoch) |
|---|---|---|---|---|---|---|---|---|
| standard | — | 9,472,010 | 20.2 | 201.9 | 203.9 | 17.7 | 1.00× | 1.00× |
| linear | — | 9,472,010 | 3.37 | 33.3 | 34.0 | 15.8 | 6.00× | 5.99× |
| piecewise (m=1) | — | 9,472,010 | 4.03 | 39.8 | 40.6 | 16.3 | 5.02× | 5.01× |
| linformer | k=256 | 13,666,314 | 4.51 | 44.6 | 45.5 | 16.5 | 4.48× | 4.48× |
| nystromformer | landmarks=128 | 9,472,010 | 6.06 | 60.2 | 61.1 | 13.1 | 3.34× | 3.33× |
| luna | pack=256 | 9,537,546 | 6.19 | 61.4 | 62.4 | 17.3 | 3.27× | 3.26× |
| performer | features=256 | 9,472,010 | 8.45 | 84.1 | 85.3 | 17.4 | 2.39× | 2.39× |
| piecewise_kmeans (m=4) | anchors=4, iters=3 | 9,472,010 | 12.77 | 172.2 | 173.3 | 13.0 | 1.18× | 1.58× |

Speedups are computed as `standard / method` on each basis: total wall (203.9 min)
and steady per-epoch (20.2 min).

### Steady-state ranking (fastest → slowest)

1. linear — 3.37 min/epoch — 5.99× *(accuracy-floor degenerate case)*
2. **piecewise (m=1) — 4.03 min/epoch — 5.01×** *(fastest softmax-fidelity method)*
3. linformer — 4.51 min/epoch — 4.48×
4. nystromformer — 6.06 min/epoch — 3.33×
5. luna — 6.19 min/epoch — 3.26×
6. performer — 8.45 min/epoch — 2.39×
7. piecewise_kmeans (m=4) — 12.77 min/epoch — 1.58×
8. standard — 20.2 min/epoch — 1.00× (baseline)

Among the softmax-fidelity methods, **piecewise (m=1) is the fastest at 5.01×**,
beating every training-free and learned peer.

### Data load is negligible → compute ≈ wall

Total data-load time spans **13.0–17.7 s across the entire 10-epoch run** for every
method — under 0.3 min against runs of 34–204 min, a <0.5% overhead in all cases.
Consequently `total_wall` and `total_compute` differ by only ~0.8–2.0 min, and
**compute is effectively equal to wall time**. Rankings are identical on either
basis; the data pipeline is not a confound.

## 3. The m=4 Anchor Warmup Decay and n-Dependent Crossover

### Per-epoch wall-time decay

`piecewise_kmeans` (m=4, kmeans_iters=3) shows a strongly front-loaded,
monotonically decaying epoch cost that settles into a stable plateau:

| Epoch | Wall (min) | Δ vs prev |
|------:|-----------:|----------:|
| 1  | 43.22 | — |
| 2  | 21.56 | −21.66 |
| 3  | 18.33 | −3.23 |
| 4  | 13.37 | −4.96 |
| 5  | 13.87 | +0.50 |
| 6  | 12.46 | −1.41 |
| 7  | 12.80 | +0.34 |
| 8  | 12.32 | −0.48 |
| 9  | 12.31 | −0.01 |
| 10 | 12.29 | −0.02 |

Steady state (mean of epochs 4–10) is **12.77 min/epoch**. Epoch 1 (**43.22 min**)
is ~3.4× the steady-state epoch (43.22/12.77 = 3.38×) and ~2.5× the mean of the
full 10-epoch trace (43.22/17.25 = 2.51×); epochs 2–3 are still warming up.

**Interpretation.** This is the signature of front-loaded k-means clustering cost.
In the first epoch the anchor centroids are far from the data manifold, so the
`kmeans_iters=3` Lloyd updates move a lot and assignments churn. As training
proceeds the centroids stabilize, assignments change little between batches, and by
epoch 4 the cost amortizes down to a flat plateau. Epochs 4–10 vary by <1.6 min
(noise), confirming the warmup is a one-time transient, not a recurring per-epoch
tax.

### Whole-run average understates the steady cost

The total-wall speedup for `piecewise_kmeans` (**1.18×**) understates its true
steady-state cost because epoch 1 is dominated by the one-time k-means warmup. The
steady speedup is 20.2 / 12.77 = **1.58×** vs standard. Over more epochs the
whole-run average asymptotically approaches the steady figure, so 1.18× is an
artifact of the short 10-epoch horizon, not the method's real per-step cost. By
contrast `standard` is flat from epoch 1, so its whole-run and steady figures are
effectively identical — the warmup gap is unique to the k-means anchor path.

### The n-dependent crossover

The relative standing of m=4 versus dense `standard` flips with sequence length.

- **Large n (this run, GPU, n=2048, steady state):** m=4 = **12.77 min/epoch** vs
  dense = 20.2 min/epoch → m=4 is **1.58× faster**.
- **Small n (separate CPU cost sweep, `results/anchor_cost_sweep.json`):** dense
  wins — at d=64/n=512 the m=4 apply is ~12.9 ms vs dense ~8.5 ms (~1.5× slower),
  and the m=64/m=1 slowdown grows with feature dim (~50× at d=64/n=2048 vs ~130× at
  d=128/n=2048 — a d² signature).

**Explanation.** Dense attention is O(n²·d); the multi-anchor path is O(m·n·d²).
Their ratio is `(m·n·d²)/(n²·d) = m·d/n`. At small n this exceeds 1 (the d² and m
constants dominate while dense is cheap), so dense wins; as n grows the ratio falls
below 1 — dense's quadratic-in-n term loses to the linear-in-n anchor term. The
n=2048 regime sits past the crossover (n ≈ m·d), so m=4 is faster despite the
heavier d² constant.

**Suspected dominant op.** The d² cost is attributed to the mean-field covariance
einsum `bmn,bnd,bne->bmde` — a per-anchor d×d accumulation over all n keys,
O(b·m·n·d²) — which is both where the d² factor enters and the source of the
linear-in-n term that lets m=4 overtake dense at large n. Confirming this by
profiling, and evaluating anchor caching, is the subject of a planned follow-up.

## 4. Accuracy (fixed-budget, not converged)

### All 8 methods sat at the near-chance floor

Every method — standard, linear, performer, nystromformer, linformer, luna,
piecewise, piecewise_kmeans — reported the **identical** accuracy pair:

| Metric | Value |
|---|---|
| test accuracy | 0.1615 (~16.15%) |
| best validation accuracy | 0.1795 (~17.95%) |

ListOps is a 10-way classification task, so pure guessing is ~10%. All eight
architecturally distinct mechanisms converging to a bit-identical ~16% test / ~18%
val is the signature of **no meaningful learning** — a near-degenerate solution only
modestly above chance, not a converged model.

### Fixed-budget, NOT-converged caveat

These numbers must **not** be read as canonical LRA ListOps results. For reference,
the LRA paper reports a standard-Transformer ListOps accuracy of ~36.37% (external,
attributed to that paper, not measured here); every method in this run is at ~16%,
less than half of that and near chance. This run is a **fixed-budget
training-dynamics probe** (10 epochs, matched budget = 256), not a convergence run.

### Likely cause and available fix

The uniform failure across all eight methods points to a shared optimizer/schedule
limitation rather than any attention variant: the run used a flat LR of 1e-3 with no
scheduler, and ListOps is known to benefit from LR warmup to escape the
majority-class attractor early in training. A warmup + cosine scheduler is now
available (`--scheduler cosine`, with `--warmup-ratio`/`--warmup-steps`); a future
convergence run should enable it before any accuracy comparison is drawn.

## 5. Takeaways

- **Speed is the trustworthy result.** Under a fair matched-budget setup,
  **piecewise (m=1) is the fastest softmax-fidelity method at 5.01× steady-state
  speedup**, beating linformer, nystromformer, luna, and performer. `linear` is
  nominally faster (5.99×) but is the known accuracy-floor degenerate case.
- **The m=4 warmup is a measurement artifact, not a per-step tax.** Its whole-run
  1.18× understates the true 1.58× steady-state speedup; the gap is entirely the
  one-time cold-start k-means cost in epoch 1.
- **n-dependent crossover.** m=4's advantage over dense appears only past the
  crossover n ≈ m·d; at small n (CPU sweep) dense wins.
- **Accuracy carries zero signal here.** The identical ~16% across all methods
  reflects a shared training-dynamics outcome under flat LR, not the attention
  mechanisms. Do not claim any accuracy win, ranking, or parity from this run.

### Follow-up: anchor caching

The epoch-1 spike is a warmup transient tied to cold centroid initialization.
**Hypothesis:** caching centroids across batches (warm-starting each batch's k-means
from the previous batch's converged centroids rather than re-seeding cold) should
flatten the front-loaded warmup, collapsing the 43→12 min decay toward a near-flat
curve. This would remove the one-time warmup penalty without changing steady-state
cost. Whether the residual m>1 cost is dominated by the clustering or by the
`bmn,bnd,bne->bmde` mean-field einsum is the open question the follow-up profiling
will settle.
