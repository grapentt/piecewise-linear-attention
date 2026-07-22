# Research Status

A living summary of where anchor-Taylor (piecewise-linear) attention stands on
the two axes that matter — **accuracy** relative to exact softmax, and **speed**
relative to the efficient-attention baselines. See `THEORY.md` for the
derivation and `results/` for the reproducible measurements and their operating
points.

## One-line state

Piecewise attention is uniformly faster than standard softmax at long context.
At a single anchor (`m=1`) it also beats the linear-attention baselines on
speed but is not accurate enough on real trained-model activations. At high
anchor count (`m=64`) it beats every efficient baseline on real-activation
accuracy but is still slower than Performer on CPU. **The accuracy win and the
speed loss are the same configuration.** Whether a middle `m` lands in a
genuinely non-dominated (faster *and* competitively accurate) region is open and
gated on a GPU measurement at realistic per-head dimension.

## Speed

Cost is `O(batch · m · n · d²)` — linear in sequence length `n`, linear in
anchor count `m`. Ratios below are piecewise ÷ Performer forward-pass wall-clock
(CPU, batch=64, Performer `num_features=256`); `> 1` means piecewise is slower.

| config | ratio vs Performer | reading |
|---|---|---|
| `m=1` (mean anchor) | 0.19–0.35× | ~3–5× **faster** than Performer |
| `m=16` | 1.80–3.29× | 1.8–3.3× slower |
| `m=64` | 4.58–14.1× | 4.6–14× slower |

- **vs standard softmax** (`O(n²·d)`): every piecewise config is sub-quadratic,
  so all beat softmax past the crossover (`n≈1024`). At `n=4096`, the mean
  anchor is ~45× faster than standard softmax.
- **vs linear/Performer**: only `m=1` and raw linear attention beat Performer;
  the accuracy-competitive `m≥16` configs are slower.

The high-`m` apply was recently reformulated to keep the second-moment operator
compact at `(batch, m, d, d)` and route queries through a capacity-bucket
group-by-anchor matmul rather than gathering a per-query `(d, d)` Jacobian. This
is **exact** (bit-identical to the naive reference, maxerr ≤ 5e-7) and roughly
**halved** the high-`m` slowdown — a pure Pareto improvement, no accuracy cost.
It narrowed but did not close the gap: `m=64` remains slower than Performer on
CPU. The remaining floors are (a) building the compact operator is still
`O(m·n·d²)`, and (b) the `d²` factor itself, which worsens at `d=128`.

## Accuracy

Relative error to exact softmax, on two input regimes that give **opposite**
verdicts on whether anchor count matters:

- **Synthetic clustered inputs** — anchor count barely matters (one centroid
  already covers a tight query cloud). Piecewise is ~10× more accurate than
  linear/Performer and solves associative recall where linear attention fails.
  Anchor count here is a convergence-speed knob, not a capability knob.
- **Real trained-model activations** — anchor count is a genuine accuracy lever
  (query distributions are spread out, so more anchors → strictly lower error).
  Worst-layer relative error at `n=8192`:

  | method | rel-err |
  |---|---|
  | piecewise `m=1` | 0.67 |
  | piecewise `m=16` | 0.43 |
  | **piecewise `m=64`** | **0.24** |
  | Nyström L=256 | 0.52 |
  | Performer M=512 | 0.94 |

  `m=64` is the only piecewise config that clearly beats the Nyström landmark
  baseline; `m=1` alone loses to Nyström at long context.

## The tension

The configuration that wins accuracy (`m=64`) is the slow one; the configuration
that wins speed (`m=1`) is not accurate enough on real long-context data. No
single config yet wins both axes against the linear baselines. The exact
apply-speedup closed about half the `m=64` gap for free; the remainder is the
kind of `d²`/tensor-core effect that CPU measurement penalizes most and a GPU
measurement at `d=64–128` (with bf16) is expected to recover.

## What is not yet established

- GPU speed–accuracy Pareto at realistic per-head `d` (64–128) across
  `n∈{1k…16k}`, fp32 and bf16, forward and forward+backward — the existential
  gate for the efficiency framing.
- Whether the `d²` factor can be cut (structured / low-rank / diagonal Jacobian)
  while keeping the high-`m` fidelity gain.
- Real-data end-to-end task quality (LRA, translation) at realistic sizes.
- Chunked causal cumsum so the `O(n·d²)` causal *memory* stays bounded at large
  `n·d`.
