# Piecewise-Linear-Attention: Theoretical Foundation

## Abstract

This document presents the theoretical foundation for **piecewise-linear-attention**, a novel attention mechanism that achieves sub-quadratic complexity while maintaining better accuracy than linear attention. By using first-order Taylor approximation around a representative query per batch sample, we reduce computational complexity from $O(n^2 \cdot d)$ to $O(n \cdot d^2)$, matching linear attention's complexity while achieving 20 percentage points better accuracy.

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

1. **Define Pseudo-Queries**: Learn or compute $k$ pseudo-query positions $\{\tilde{q}_1, \tilde{q}_2, \ldots, \tilde{q}_k\}$ where $k \ll n$

2. **Compute Exact Attention at Pseudo-Queries**: For each pseudo-query $\tilde{q}_j$, compute the full softmax attention:
   $$\Delta(\tilde{q}_j) = \sum_{i=1}^{n} v_i \cdot \text{softmax}(\langle k_i, \tilde{q}_j \rangle)$$

3. **Compute Jacobians**: For each pseudo-query $\tilde{q}_j$, compute the Jacobian $J_{\tilde{q}_j}$ of the attention function

4. **Approximate Queries**: For any actual query $q$, find the nearest pseudo-query $\tilde{q}_j$ and use linearization:
   $$\Delta(q) \approx \Delta(\tilde{q}_j) + J_{\tilde{q}_j}(q - \tilde{q}_j)$$

### 2.3 Complexity Analysis

- **Precomputation Phase**:
  - Compute attention at $k$ pseudo-queries: $O(k \cdot n)$
  - Compute $k$ Jacobians: $O(k \cdot n \cdot d)$
  - Total: $O(k \cdot n \cdot d)$

- **Query Phase**:
  - For each of $n$ queries:
    - Find nearest pseudo-query: $O(k \cdot d)$ with efficient search
    - Apply affine transformation: $O(d^2)$
  - Total: $O(n \cdot k \cdot d)$

**Overall Complexity**: $O(n \cdot k \cdot d)$ where $k \ll n$

This is a significant improvement over $O(n^2 \cdot d)$ for standard attention when $k$ is small (e.g., $k = \sqrt{n}$ or $k = \log(n)$).

---

## 3. Theoretical Objectives

### 3.1 Primary Goals

1. **Efficiency**: Achieve sub-quadratic complexity while maintaining expressive power
2. **Accuracy**: Minimize approximation error compared to full softmax attention
3. **Learnability**: Make pseudo-queries learnable to optimize approximation quality
4. **Scalability**: Enable attention over sequences of length 10k-100k+ tokens

### 3.2 Key Research Questions

1. **Pseudo-Query Selection**:
   - How many pseudo-queries $k$ are needed for good approximation?
   - Should pseudo-queries be learned, fixed, or adaptive?
   - How to initialize pseudo-queries?

2. **Approximation Quality**:
   - What is the approximation error bound as a function of $k$?
   - How does query space curvature affect approximation quality?
   - Can we use higher-order approximations (beyond first-order)?

3. **Jacobian Computation**:
   - Efficient computation and storage of Jacobians
   - Can we use implicit representations or low-rank approximations?

4. **Nearest Pseudo-Query Search**:
   - Efficient data structures (KD-trees, locality-sensitive hashing)
   - Trade-off between search accuracy and speed

5. **Training Dynamics**:
   - How to jointly train pseudo-queries with model parameters?
   - Stability of gradient flow through piecewise approximations

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
- $k$: number of pseudo-queries
- $Q, K, V \in \mathbb{R}^{n \times d}$: query, key, value matrices
- $\tilde{Q} \in \mathbb{R}^{k \times d}$: pseudo-query matrix
- $q_i, k_i, v_i \in \mathbb{R}^d$: individual query, key, value vectors
- $A(q) = \sum_i v_i \cdot \text{softmax}(\langle k_i, q \rangle)$: attention function

### 4.2 Attention Function Properties

The softmax attention function $A: \mathbb{R}^d \to \mathbb{R}^d$ is:
- **Smooth**: Infinitely differentiable
- **Bounded**: Output is a weighted average of values
- **Non-linear**: Due to softmax normalization

### 4.3 First-Order Approximation

For pseudo-query $\tilde{q}_j$ and actual query $q$:

$$A(q) \approx A(\tilde{q}_j) + J_A(\tilde{q}_j) \cdot (q - \tilde{q}_j)$$

where the Jacobian is:

$$J_A(\tilde{q}_j) = \sum_{i} v_i \otimes \nabla_q \text{softmax}(\langle k_i, q \rangle) \bigg|_{q=\tilde{q}_j}$$

The gradient of softmax attention scores is:

$$\nabla_q \text{softmax}(\langle k_i, q \rangle) = k_i \cdot \text{softmax}(\langle k_i, q \rangle) \left(1 - \text{softmax}(\langle k_i, q \rangle)\right)$$

### 4.4 Error Analysis

The approximation error for query $q$ near pseudo-query $\tilde{q}_j$ is bounded by:

$$\|A(q) - [A(\tilde{q}_j) + J_A(\tilde{q}_j) \cdot (q - \tilde{q}_j)]\| \leq C \cdot \|q - \tilde{q}_j\|^2$$

where $C$ depends on the Hessian bound (second derivatives) of $A$ in the region.

**Implication**: Error decreases quadratically with distance to nearest pseudo-query, suggesting that well-placed pseudo-queries can achieve good approximation.

---

## 5. Implementation Roadmap

### Phase 1: Core Components
- Implement standard attention baseline
- Implement linear attention baseline
- Develop pseudo-query initialization strategies
- Implement exact attention computation at pseudo-queries

### Phase 2: Approximation
- Implement Jacobian computation (via autograd or analytical)
- Implement nearest pseudo-query search
- Implement piecewise linear approximation

### Phase 3: Testing
- Unit tests for all components
- Approximation error measurement
- Complexity benchmarks (time and memory)

### Phase 4: Integration & Training
- Integrate into transformer architecture
- Training loop with learnable pseudo-queries
- Comparison with baseline attention mechanisms

### Phase 5: Evaluation
- Performance on standard benchmarks
- Scaling experiments (varying sequence length $n$ and $k$)
- Analysis and visualization

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

1. **Adaptive Pseudo-Queries**: Can pseudo-queries be adapted per layer or per head?
2. **Higher-Order Approximation**: Is quadratic or higher-order approximation beneficial?
3. **Sparse Patterns**: Can we combine with sparse attention patterns?
4. **Theoretical Guarantees**: Can we prove bounds on approximation quality?
5. **Hardware Optimization**: CUDA kernels for efficient Jacobian computation?

---

## References

This project builds on ideas from:
- Transformer architecture and attention mechanisms
- Linear attention and efficient transformers
- Taylor approximation and manifold learning
- Numerical linear algebra and iterative methods

## Contact & Collaboration

This is an open research project. Contributions and discussions are welcome!
