# Research Status

A living summary of where anchor-Taylor (piecewise-linear) attention stands on
the two axes that matter — **accuracy** relative to exact softmax, and **speed**
relative to the efficient-attention baselines. See `THEORY.md` for the
derivation and `results/` for the reproducible measurements and their operating
points.

## One-line state

Piecewise attention is uniformly faster than standard softmax at long context.
At a single anchor (`m=1`) it is Pareto-competitive with the linear-attention
baselines on GPU — **~4× faster than Performer at `d=128`**, tying Nyströmformer —
but is not accurate enough on real trained-model activations. At high anchor count
(`m=64`) it reaches the **lowest error of any method** (rel-err `0.24`) but is the
slow config and, on the current dense build, **runs out of memory at `n≥8192`**.
**The accuracy win and the speed/memory loss are the same configuration**, and the
GPU Pareto measurement (A100, `d=64/128`, `n≤16384`) confirms no single `m` is
non-dominated on both axes: anchor count is a monotone accuracy dial traded against
latency and memory. The open problem is making the accurate high-`m` build fit in
memory at long context.

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

### Build-side top-p: measured, no structural win on real data

An optional sparse-support build (`build_topp`, off by default) truncates each
anchor's second-moment build to that anchor's **nucleus** — the top-weighted
keys whose cumulative softmax mass reaches `p`. A nucleus-size measurement on
real trained-model activations (native `n=512` blocks, the only trustworthy
peakedness signal) shows the shipped **shared** cap — sized to the worst/most-
diffuse) anchor and k-means anchors are query-side centroids, the worst-anchor
nucleus stays near the full sequence (`~0.87–0.93` of the keys at `p=0.99`), so
the build shrink ceiling is only `~1.08×` and the path correctly self-guards to
dense. Anchors are query-side centroids, so at least one attends broadly across
keys — the query-centroid null holds. The **median** anchor's nucleus is much
smaller (`~0.3–0.6` at `p=0.99`), so a per-anchor (ragged) cap scaling with the
mean rather than the max has a `~1.5–2.5×` build FLOP ceiling (larger at a looser
`p=0.9`); whether that converts to a wall-clock win on GPU is an open follow-up
(see below). Reproduced by the real-activation fidelity harness; distribution in
`results/real_activation_fidelity_nucleus.json`.

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

### Cutting the `d²` factor: query-residual low rank is confirmed on real data

The dominant remaining build cost is the per-anchor second-moment operator
`M_j` — a full `d×d` matrix, so it carries a `d²` factor that dominates at the
realistic per-head `d=128`. That operator is only ever consumed as `M_j·Δq`,
where `Δq = q − a_j` is a query's offset from its anchor. Because anchors are
cluster centroids, each cluster's `Δq` vectors should span only a few directions,
so `M_j` need only be built on those `d' ≪ d` directions — replacing `d²` with
`d·d'`.

Whether `d'` is actually small is an empirical question about real activations,
and it is now measured (per-cluster SVD of the `Δq` residuals, native `n=512`
blocks, at three real per-head dimensions — MiniLM `d=32`, BERT-base `d=64`,
GPT-Neo-1.3B `d=128`):

| `d` | worst-anchor `d'@95%` (m=64) | median anchor |
|---|---|---|
| 32 | 0.50·d | 0.22·d |
| 64 | 0.25–0.56·d | 0.12·d |
| **128** | **0.20–0.30·d** | **0.07·d (≈13× fewer directions)** |

Two trends make this the right lever: the required fraction of directions
**shrinks as `d` grows** (so the reduced cost `d·d'` grows far slower than `d²`,
exactly opposite the cost trend), and **shrinks as `m` grows** (tighter clusters →
smaller `Δq` subspace), strongest in the accurate `m=64` regime. The key/operator
side stays broad on real data (the broadest anchor's softmax nucleus is ~0.99 of
all keys at `d=128`), so key-side low-rank compression is *not* viable — the win
is specifically on the query-residual axis. See issue #9 and
`results/jacobian_spectrum_gpu_d128.json`.

#### The reduced build is accurate but not faster: per-forward SVD kills it

The per-anchor **ragged-`d'`** reduced operator was then built (behind an
optional `build_reduced_d` flag: each anchor keeps its own `d'`, the residual
basis `R_j` is the right singular vectors of that cluster's `Δq`, and `M_j` is
built directly as the `(d, d')` factor `M_j^red`; fallbacks to the exact dense
`M_j` on tiny clusters, SVD failures, and non-low-rank anchors, so it trades
speed, never accuracy). Measured on the A100 at `d=128`, `m=64`, the result is a
clean split — the structure is real and the approximation is accurate, but it is
not a wall-clock win:

- **Accuracy — passes with margin.** Reduced `p=0.99` rel-err to exact softmax on
  real GPT-Neo-1.3B activations (native `n=512`, layers 6/12/18) is
  `0.0022–0.0257`, within ~1.5–2× of the exact dense `m=64` build (`0.0012–0.0133`)
  and **beating Nyström `L=256`** (`0.0285–0.0507`) at every layer. `p=0.95` is
  looser (loses to Nyström at layer 6), so `p=0.99` is the operating point at
  `d=128`. This confirms the low-rank query-residual structure end-to-end.
- **Wall-clock — fails, decisively.** The reduced build runs **~200–800× slower**
  than the dense build (build `x0.003`, forward `x0.005` at `d=128`, bf16 and
  fp32). The cause is exactly the anticipated one: obtaining `R_j` needs a
  per-anchor SVD of the residuals, i.e. `batch·m ≈ 512` tiny `torch.linalg.svd`
  calls per forward in a Python loop, each with its own launch + host sync. This
  swamps the theoretical build-FLOP saving (a real but irrelevant `~2–3×` ceiling).
  There is no basis reuse in the independent-forward benchmark, so the SVD is paid
  every forward.
- **Memory — also not a win here** (`~0.68×`, i.e. reduced uses more): `R` and
  `M_j^red` together, padded to the worst anchor's `d'_max`, exceed the single
  compact dense `M`.

Verdict: the reduced build is a **validated accuracy/structure result, not a
speed lever** on current kernels. A wall-clock win would require either amortizing
`R_j` (frozen anchors + cached basis — a different mode, absent here) or a batched
SVD/randomized-range-finder that avoids the per-anchor Python loop. The flag ships
off by default. Data: `results/reduced_build_a100.json` (time/memory/FLOP) and
`results/reduced_accuracy_gate_d128.json` (accuracy). Measured on A100 (not H100);
the accuracy verdict is hardware-independent.

## GPU speed–accuracy Pareto at realistic `d`: the existential gate, resolved

The speed claim was measured directly on an NVIDIA A100 80GB across per-head
`d ∈ {64, 128}`, `n ∈ {1024…16384}` at fixed `batch·heads = 256`, both `fp32` and
`bf16`, forward and forward+backward, non-causal and causal — the full grid the
efficiency framing depends on (`results/cuda_scaling_a100.json`, forward-pass
latency below is `bf16`). Every softmax reference (a manual `O(n²)` build and a
fair `scaled_dot_product_attention` with the flash and memory-efficient backends
disabled) runs out of memory by `n=8192` at both dimensions, so the sub-quadratic
methods are the only survivors at long context.

The headline concern — that the `d²` factor would erase the speed lead at the
realistic per-head `d=128` — **does not hold**. At `d=128` the single-anchor
`m=1` build is **~4× faster than Performer at every length** (`0.26–0.29×` its
latency) and roughly ties Nyströmformer. But it does **not dominate the frontier**:
Luna (pack 64) is ~1.65× faster still, so `m=1` is Pareto-competitive, not
Pareto-best, on latency alone.

Joining that latency with the real-activation accuracy above gives the honest
picture at the representative operating point (`d=128`, `n=8192`, `bf16`):

| method | fwd latency | peak mem | worst-layer rel-err |
|---|---|---|---|
| luna (pack 64) | 3.8 ms | 3.2 GB | ~0.9 (Performer-class) |
| nyströmformer L=64 | 5.7 ms | 3.8 GB | 0.52 |
| **piecewise `m=1`** | **6.3 ms** | **3.2 GB** | 0.67 |
| linformer k=256 | 9.2 ms | 4.4 GB | — |
| performer M=256 | 24.2 ms | 6.5 GB | 0.95 |
| **piecewise `m=64`** | **OOM** | — | **0.24 (best of all)** |
| softmax (manual / SDPA-math) | OOM | — | 0 (exact) |

The tension the whole project turns on is now measured on both axes at once: the
configuration that wins **accuracy** (`m=64`, rel-err `0.24`, best of every method)
**runs out of memory at `n≥8192`** on the current dense build, while the
configuration that wins **speed/memory** (`m=1`) is the least accurate piecewise
setting. Anchor count is a monotone accuracy dial (`m=1→64`: `0.67→0.51→0.43→0.24`)
that trades directly against latency and memory; no single `m` is non-dominated on
both axes against the linear baselines. The causal path confirms the memory ceiling
directly — causal `m=1` materialises a `(batch, n, d, d)` cumulative-sum tensor, so
it OOMs at `n=8192, d=64` in `fp32` but fits in `bf16` (and OOMs at `n=16384`).

Consequence for the framing: the defensible headline is **not "fastest"** but that
piecewise is a deterministic linear-attention method whose single anchor-count knob
sweeps a competitive accuracy–latency frontier — `m=1` sits with the fast linear
baselines (4× over Performer) and `m=64` reaches the lowest error of any method,
with the open engineering problem being to make the accurate high-`m` build fit in
memory at long context. Reproduced by `benchmarks/harness/run_cuda_scaling.py`;
measured on A100 (not H100), a hardware-independent Pareto verdict.

## What is not yet established

- Whether the accurate high-`m` build can be made to **fit in memory at long
  context** (`n≥8192`, `d=128`) — `m=64` currently OOMs on the dense
  `(batch, m, d, d)` operator, which is what keeps the accuracy-winning config off
  the long-context frontier. This is now the primary open problem for the framing.
- Whether an SVD-free variant of the reduced build (amortized/cached `R_j` under
  frozen anchors, or a batched randomized range-finder) can realize the confirmed
  query-residual low rank as a wall-clock win. The straightforward per-forward
  reduced build is accurate but ~200–800× slower (per-anchor SVD cost); see above.
- Whether a per-anchor (ragged) build cap — scaling with the mean nucleus rather
  than the worst-anchor max — converts its `~1.5–2.5×` build FLOP ceiling into a
  measured wall-clock win on GPU (the batched-cap version does not; the median
  nucleus is small but the max is not).
- Real-data end-to-end task quality (LRA, translation) at realistic sizes.
- Chunked causal cumsum so the `O(n·d²)` causal *memory* stays bounded at large
  `n·d`.
