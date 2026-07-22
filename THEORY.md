# Piecewise-Linear-Attention: Theoretical Foundation

## Abstract

This document presents the theoretical foundation for **piecewise-linear-attention**, a deterministic linear-time attention mechanism. By linearizing softmax attention with a first-order Taylor expansion around a small set of representative *anchor* queries, we reduce computational complexity from $O(n^2 \cdot d)$ to $O(n \cdot d^2)$ — the same class as linear attention, but tracking softmax far more closely because the expansion is exact *at* each anchor and its error grows only quadratically with the query-to-anchor distance. On synthetic benchmarks the mean-centroid variant approximates softmax markedly better than random-feature (Performer) and kernel (linear) attention, and it solves associative recall where linear attention fails; see the `results/` directory and the project README for the reproducible measurements and their operating points.

---

## 1. Background: Attention Mechanisms

### 1.1 Standard (Softmax) Attention

In transformer architectures, for each token we compute:
- **Query ($q$)**: "I am looking for this"
- **Key ($k$)**: "I have this, if you want my property, align in my direction"
- **Value ($v$)**: "This is the actual value vector you have to add to incorporate my property"

Standard attention computes the output for query $q$ as:

$$\Delta = \sum_{i=1}^{n} v_i \cdot \text{softmax}(\langle k_i, q \rangle)$$

**Complexity**: $O(n^2)$ for computing attention scores for all $n$ tokens
- For each of $n$ queries, we compute scores against $n$ keys: $O(n^2)$
- This becomes the bottleneck for long sequences

### 1.2 Linear Attention

Linear attention attempts to factorize the computation by building a memory matrix:

$$M = \sum_{i=1}^{n} v_i \otimes k_i^*$$

where $k_i^*$ denotes the dual of the key vector.

Querying this matrix gives:

$$\Delta = M(q) = \sum_{i=1}^{n} v_i \otimes k_i^*(q) = \sum_{i=1}^{n} v_i \cdot \langle k_i, q \rangle$$

**Complexity**: $O(n)$ once $M$ is constructed
- Build $M$ in $O(n \cdot d^2)$ where $d$ is the dimension
- Query $M$ for each token in $O(d^2)$
- Total: $O(n \cdot d^2)$ vs $O(n^2 \cdot d)$ for softmax

**Problem**: The raw dot product $\langle k_i, q \rangle$ is too noisy without the normalizing and sharpening effect of softmax, leading to degraded performance.

---

## 2. The Piecewise-Linear-Attention Approach

### 2.1 Core Idea

Instead of using raw dot products (too noisy) or computing full softmax attention (too expensive), we use **piecewise linear approximation** of the attention function.

**Key Insight**: Around any point $q_0$, we can approximate a smooth function using its first-order Taylor expansion:

$$f(q_0 + \Delta q) \approx f(q_0) + J_{q_0}f(\Delta q)$$

where $J_{q_0}f$ is the Jacobian of $f$ at $q_0$.

### 2.2 Method

1. **Select anchors**: choose $m$ anchor queries $\{a_1, \ldots, a_m\}$ with $m \ll n$ via a pluggable strategy. The default is the single **mean/centroid** anchor ($m=1$, $a = \frac{1}{n}\sum_i q_i$); alternatives place anchors on query clusters (k-means) or at fixed sequence positions (stride).

2. **Compute exact attention at anchors**: for each anchor $a_j$, compute the full softmax attention
   $$A(a_j) = \sum_{i=1}^{n} v_i \cdot \text{softmax}(\langle k_i, a_j \rangle / \sqrt{d})$$

3. **Compute the analytic Jacobian** $J_A(a_j)$ of the attention map at each anchor (closed form in §4.3).

4. **Linearize each query around its nearest anchor** $a_{j(i)}$:
   $$A(q_i) \approx A(a_{j(i)}) + J_A(a_{j(i)})\,(q_i - a_{j(i)})$$

For the default single-anchor case, step 3's nearest-anchor assignment is trivial (every query uses the one anchor) and the whole map is differentiable end-to-end. For $m>1$ the assignment is a non-differentiable `argmin`; gradients still flow through the expansion with respect to $Q, K, V$ (straight-through routing).

### 2.3 Complexity Analysis

- **Anchor phase**: exact attention and Jacobian factors at $m$ anchors: $O(m \cdot n \cdot d)$ for the outputs and weighted keys, $O(m \cdot n \cdot d^2)$ for the second-moment factor $M_j = \sum_n \alpha_{jn}\, v_n k_n^\top$.

- **Apply phase**: the expansion is applied in a **fused** form (§4.5) that distributes the matmul over the rank-1 Jacobian, so a per-query $d\times d$ matrix is never materialized: $O(m \cdot n \cdot d^2)$.

**Overall**: $O(\text{batch} \cdot m \cdot n \cdot d^2)$ — linear in $n$ with a small constant $m$ (typically 1–8), versus $O(\text{batch} \cdot n^2 \cdot d)$ for standard attention. The linear-time methods overtake softmax once $n$ is past the $n \approx d\cdot m$ FLOP crossover (empirically $n\approx1024$ on the tested backend); below that, softmax's small constant wins.

---

## 3. Theoretical Objectives

### 3.1 Primary Goals

1. **Efficiency**: Achieve sub-quadratic complexity while maintaining expressive power
2. **Accuracy**: Minimize approximation error compared to full softmax attention
3. **Learnability**: Optionally learn the anchor placement to further reduce approximation error
4. **Scalability**: Enable attention over long sequences ($n \gg 10^3$)

### 3.2 Key Research Questions

The core questions — how many anchors, where to place them, whether higher-order terms help, and deterministic vs. stochastic approximation — are addressed by the current evidence and tracked in §7 (Open Questions and Future Directions).

### 3.3 Expected Benefits

- **Long-Context Modeling**: Enable transformers to process much longer sequences
- **Inference Speed**: Faster attention computation for deployment
- **Memory Efficiency**: Reduced memory footprint for large-scale models
- **Theoretical Understanding**: Better understanding of attention mechanism geometry

---

## 4. Mathematical Formulation

### 4.1 Notation

- $n$: sequence length
- $d$: embedding dimension
- $m$: number of anchors ($m \ll n$; default $m=1$)
- $Q, K, V \in \mathbb{R}^{n \times d}$: query, key, value matrices
- $a_j \in \mathbb{R}^d$: the $j$-th anchor query
- $q_i, k_i, v_i \in \mathbb{R}^d$: individual query, key, value vectors
- $A(q) = \sum_i v_i \cdot \text{softmax}(\langle k_i, q \rangle / \sqrt d)$: attention function

### 4.2 Attention Function Properties

The softmax attention function $A: \mathbb{R}^d \to \mathbb{R}^d$ is:
- **Smooth**: Infinitely differentiable
- **Bounded**: Output is a weighted average of values
- **Non-linear**: Due to softmax normalization

### 4.3 First-Order Approximation

For anchor $a$ and actual query $q$:

$$A(q) \approx A(a) + J_A(a) \cdot (q - a)$$

Writing $s = 1/\sqrt{d}$ for the scale and $\alpha_i = \text{softmax}(s\langle k_i, a\rangle)$ for the anchor's attention weights, the **analytic Jacobian** of the softmax attention map is

$$J_A(a) = s\left[\sum_{i} \alpha_i\,(v_i \otimes k_i) \;-\; A(a) \otimes \bar{k}\right], \qquad \bar{k} = \sum_i \alpha_i k_i .$$

This is exact: it comes from differentiating the softmax weights, whose Jacobian is $\partial\alpha/\partial q = s\,(\text{diag}(\alpha) - \alpha\alpha^\top)K$. The rank-1 correction term $A(a)\otimes\bar{k}$ is the off-diagonal softmax coupling — dropping it (the naive "$\alpha(1-\alpha)$" diagonal form) gives a wrong Jacobian. The implementation and the causal cumsum derivation (§6.4) both use this exact form.

### 4.4 Error Analysis

The approximation error for query $q$ near anchor $a$ is bounded by

$$\left\|A(q) - \big[A(a) + J_A(a)\,(q - a)\big]\right\| \leq C \cdot \|q - a\|^2,$$

where $C$ depends on the Hessian bound of $A$ in the region. Empirically the mean first-order error tracks $\mathbb{E}\|q-a\|^2$ closely across dispersions, so **anchor placement is the dominant accuracy lever**: an anchor that keeps queries close to their expansion point is what matters, more than adding a curvature term (see §4.6).

### 4.5 Fused Apply (no per-query Jacobian)

Materializing $J_A(a)$ as an explicit $d\times d$ matrix per query is wasteful. Because $J_A(a)$ is a sum of a second-moment matrix and a rank-1 term, the expansion can be applied by distributing the matmul. With $o = A(a)$, $\bar{k} = \sum_i\alpha_i k_i$, $M = \sum_i \alpha_i\,v_i k_i^\top$, and $\Delta q = q - a$:

$$A(q) \approx o + s\,(M\,\Delta q) - s\,o\,(\bar{k}^\top \Delta q).$$

This is algebraically identical to $o + J_A(a)\,\Delta q$ but needs only the compact per-anchor factors $o, \bar{k}$ (each $\mathbb{R}^{d}$) and $M$ ($\mathbb{R}^{d\times d}$); no per-query $(d,d)$ tensor is ever formed. The apply is $O(m\cdot n\cdot d^2)$ and is what makes the method linear in $n$ in practice.

### 4.6 Why the Mean Centroid Is the Recommended Anchor

Since the first-order error scales with $\|q-a\|^2$, the single anchor that minimizes the mean squared query-to-anchor distance $\frac{1}{n}\sum_i\|q_i-a\|^2$ is the **centroid** $a=\frac{1}{n}\sum_i q_i$ — this is the definition of the mean. Consequently the single mean anchor ($m=1$) is simultaneously the cheapest configuration (one anchor, trivial routing) and the most accurate first-order one. Positionally placed anchors (e.g. strided sequence positions) sit at the edges of the query cloud, incur larger $\|\Delta q\|$, and are measurably worse. Adding more centroid anchors (k-means) reduces $\mathbb{E}\|\Delta q\|^2$ further and can speed convergence on some tasks, at proportionally higher cost.

---

## 5. Implementation Status

The formulation above is implemented in `piecewise_linear_attention`:

- **Baselines**: standard softmax, kernel linear attention (ELU/ReLU/softplus), and Performer (FAVOR+), all behind a common registry.
- **Anchor strategies**: `MeanAnchor` (default, $m=1$ centroid), `KMeansAnchor` (centroid clusters), `StrideAnchor` (positional) — pluggable via `anchor_strategy`.
- **Analytic Jacobian and fused apply** (§4.3, §4.5): the exact Jacobian factors are computed at each anchor and the expansion is applied without materializing a per-query $d\times d$ matrix.
- **Causal path** (§6): the cumsum formulation for $O(n\cdot d^2)$ masked attention.
- **Evidence harness**: reproducible CLIs (`benchmarks/harness/run_scaling.py`, `run_microbench.py`, `run_recall.py`, `plot_results.py`) that emit versioned JSON under `results/`.

What is **not** yet established (see the repository issues): real-data end-to-end task quality, CUDA and realistic-$d$ scaling, the long-context ($n\gg10^3$) regime, and positioning against landmark/Nyström-style methods.

---

## 6. Causal Attention: Efficient Implementation via Cumsum

### 6.1 The Challenge

For causal attention (e.g., language modeling), query at position $i$ can only attend to keys at positions $0, \ldots, i$. Naively, this requires computing a separate Jacobian for each position, leading to $O(n^2 \cdot d)$ complexity.

### 6.2 Efficient Solution: Linearize Before Summation

**Key Insight**: Both causal and non-causal compute the same Jacobian-based approximation:

$$A(q) \approx A(\bar{q}) + J_A(\bar{q}) \cdot (q - \bar{q})$$

The difference is the **computational strategy**. By linearizing the exponential kernel *before* summation, we can use cumulative sums for efficient causal masking.

### 6.3 Mathematical Derivation

For position $i$, the causal attention is:

$$A_i(q) = \frac{\sum_{j \leq i} \exp(s \cdot q \cdot k_j) v_j}{\sum_{j \leq i} \exp(s \cdot q \cdot k_j)}$$

where $s = 1/\sqrt{d}$ is the scale factor.

**Step 1: Linearize the exponential kernel** around anchor $\bar{q}$:

$$\exp(s \cdot q \cdot k_j) \approx \exp(s \cdot \bar{q} \cdot k_j) [1 + s \cdot (q - \bar{q}) \cdot k_j]$$

Let $\sigma_j = \exp(s \cdot \bar{q} \cdot k_j)$ be the anchor weights.

**Step 2: Substitute into attention**:

$$\text{Numerator}_i \approx \sum_{j \leq i} \sigma_j [1 + s \cdot (q - \bar{q}) \cdot k_j] v_j$$

$$= \sum_{j \leq i} \sigma_j v_j + s \cdot (q - \bar{q})^T \sum_{j \leq i} \sigma_j (k_j \otimes v_j)$$

Similarly for denominator:

$$\text{Denominator}_i \approx \sum_{j \leq i} \sigma_j + s \cdot (q - \bar{q})^T \sum_{j \leq i} \sigma_j k_j$$

**Step 3: Enable cumsum optimization**:

Define cumulative terms:
- $S^A_i = \sum_{j \leq i} \sigma_j v_j$ (cumulative weighted values)
- $S^B_i = \sum_{j \leq i} \sigma_j (k_j \otimes v_j)$ (cumulative weighted outer products)
- $S^C_i = \sum_{j \leq i} \sigma_j$ (cumulative weights)
- $S^D_i = \sum_{j \leq i} \sigma_j k_j$ (cumulative weighted keys)

These can be computed efficiently via `torch.cumsum`.

**Step 4: Final formula**:

$$\text{output}_i = \frac{S^A_i + s \cdot S^B_i (q_i - \bar{q})}{S^C_i + s \cdot S^D_i^T (q_i - \bar{q})}$$

### 6.4 Equivalence to Jacobian Approximation

This is mathematically equivalent to the Jacobian-based approximation! The Jacobian of the attention function at $\bar{q}$ for position $i$ is:

$$J_{A_i}(\bar{q}) = s \left[\sum_{j \leq i} \alpha_j (v_j \otimes k_j) - A_i(\bar{q}) \otimes \sum_{j \leq i} \alpha_j k_j \right]$$

where $\alpha_j = \sigma_j / \sum_{\ell \leq i} \sigma_\ell$ are the softmax weights.

Our cumsum formulation computes exactly this Jacobian, just more efficiently by exploiting the separability that comes from linearizing before summation.

### 6.5 Complexity Analysis

**Time Complexity**:
- Compute anchor weights $\sigma_j$: $O(n \cdot d)$
- Compute terms $A, B, C, D$: $O(n \cdot d^2)$ (bottleneck: outer products in $B$)
- Cumulative sums: $O(n \cdot d^2)$
- Apply to queries: $O(n \cdot d^2)$
- **Total**: $O(n \cdot d^2)$ ✓

**Space Complexity**: $O(n \cdot d^2)$ for storing $S^B_i$ at all positions

**Comparison**:
- Standard causal attention: $O(n^2 \cdot d)$ time, $O(n^2)$ memory
- Our causal piecewise: $O(n \cdot d^2)$ time, $O(n \cdot d^2)$ memory
- Linear attention: $O(n \cdot d^2)$ time, $O(d^2)$ memory

We match linear attention complexity while maintaining better approximation quality!

### 6.6 Pseudo-Query Selection Strategies

**Training** (`causal_pseudo_query="mean"`):
- Use $\bar{q} = \frac{1}{n}\sum_i q_i$ (mean of all queries)
- Better approximation quality when full sequence is available
- Small information leakage acceptable during training

**Inference** (`causal_pseudo_query="first"`):
- Use $\bar{q} = q_0$ (first query)
- Zero information leakage
- Suitable for autoregressive generation

---

## 7. Open Questions and Future Directions

**Settled by the current evidence** (see `results/` and the project issues):
- *Anchor count vs. placement*: placement dominates — the single mean centroid is the most accurate first-order config; more anchors mainly affect convergence speed (§4.6).
- *Higher-order terms*: an exact second-order term reduces error but has no linear-time ($O(n\cdot d)$) form, so it is not speed-compatible with the linear-time goal; low-rank/diagonal Jacobian approximations are refuted (the operators are full-rank).
- *Deterministic vs. stochastic*: the deterministic linearization tracks softmax more closely than Performer's random features at matched compute, and is seed-free at inference.

**Genuinely open**:
1. **Real-data task quality**: language modeling, LRA, and ViT/BERT fine-tuning at realistic model sizes.
2. **Long context and hardware**: CUDA kernels, realistic $d$ (64–128), and the $n\gg10^3$ regime where the linear-in-$n$ structure should dominate.
3. **Anchor learning**: a learned query→anchor map (vs. mean/k-means) as a training-time bet.
4. **Positioning**: head-to-head against landmark/Nyström-style low-rank attention, not only Performer/linear.
5. **Reference-frame gradient**: whether detaching the centroid (stop-gradient anchor) improves training stability.

---

## References

This project builds on ideas from:
- Transformer architecture and attention mechanisms
- Linear attention and efficient transformers
- Taylor approximation and manifold learning
- Numerical linear algebra and iterative methods

## Contact & Collaboration

This is an open research project. Contributions and discussions are welcome!
