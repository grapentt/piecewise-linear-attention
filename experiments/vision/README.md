# Vision Transformer Experiments

Vision Transformer (ViT) experiments comparing attention mechanisms on image classification tasks.

## 🎯 Attention Mechanisms Compared

This experiment compares three attention mechanisms in a Vision Transformer:

### 1. **StandardAttention** (Baseline)
- **What**: Full softmax attention `Attention(Q,K,V) = softmax(QK^T/√d) V`
- **Complexity**: O(n²d) where n = number of patches
- **Memory**: O(n²)
- **When to use**: Best accuracy, but slowest
- **Flag**: `--attention-types standard`

### 2. **LinearAttention**
- **What**: Kernel-based linear attention using feature maps `φ(Q)(φ(K)^T V)`
- **Complexity**: O(nd²)
- **Memory**: O(d²)
- **When to use**: Maximum speed, some accuracy drop
- **Flag**: `--attention-types linear`

### 3. **PiecewiseAttention** (Our Method)
- **What**: First-order Taylor approximation around representative patches
- **Complexity**: O(nd²)
- **Memory**: O(d²)
- **When to use**: Best speed/accuracy tradeoff
- **Flag**: `--attention-types piecewise`

---

## 🚀 Quick Start

### Compare All Three Attention Types (Recommended)

This will train the same ViT model three times, once with each attention type, and show a comparison:

```bash
# Train with all three attention types and compare results
python experiments/vision/train.py
```

**Output**: Trains 3 models sequentially and prints a comparison table at the end.

### Train with Single Attention Type

```bash
# Standard attention (baseline)
python experiments/vision/train.py --attention-types standard

# Linear attention (fastest)
python experiments/vision/train.py --attention-types linear

# Piecewise attention (our method)
python experiments/vision/train.py --attention-types piecewise
```

### Compare Two Specific Attention Types

```bash
# Compare only standard and piecewise
python experiments/vision/train.py --attention-types standard piecewise
```

### Full Experiment with Results Saving

```bash
# Run all three, train for 50 epochs, save results
python experiments/vision/train.py \
  --attention-types standard linear piecewise \
  --epochs 50 \
  --batch-size 128 \
  --save results/vision/cifar10_comparison.json
```

---

## 📊 Understanding the Output

### During Training

For each attention type, you'll see:
- Model architecture summary (number of parameters)
- Per-epoch training and test metrics (loss, accuracy, time, throughput)
- Final results summary (test accuracy, test loss, total training time, average speed)

### Final Comparison Table

After training all attention types, you'll see a comparison table with:
- **Test Acc (%)**: Higher is better - measures classification accuracy
- **Test Loss**: Lower is better - cross-entropy loss on test set
- **Train Time (s)**: Lower is better - total wall-clock training time
- **Speed (samples/s)**: Higher is better - training throughput
- **Speedup**: How many times faster than Standard attention

---

## 🎓 When to Use Each Attention Type

### Use **StandardAttention** when:
- ✅ You need the best possible accuracy
- ✅ Training time is not a constraint
- ✅ You have sufficient compute resources
- ✅ You're establishing a baseline for comparison

### Use **LinearAttention** when:
- ⚡ Speed is the top priority
- ⚡ You can tolerate some accuracy drop (~2-5%)
- ⚡ You have limited compute budget
- ⚡ You need very fast inference

### Use **PiecewiseAttention** when:
- 🎯 You want a good balance between speed and accuracy
- 🎯 You need significant speedup (2-3×) with minimal accuracy loss (~1-3%)
- 🎯 You're deploying models where both speed and quality matter
- 🎯 You want to compare our proposed method

---

## 🔧 Experiment Configuration

### All Available Options

```bash
python experiments/vision/train.py \
  --dataset cifar10                    # Dataset (default: cifar10)
  --model-size tiny                    # Model size: tiny/small/base (default: tiny)
  --epochs 50                          # Number of epochs (default: 50)
  --batch-size 128                     # Batch size (default: 128)
  --lr 0.001                           # Learning rate (default: 0.001)
  --attention-types standard linear piecewise  # Which types to compare (default: all three)
  --device cpu                         # Device: cpu/cuda (default: cpu)
  --save results/vision/output.json   # Save results to file (optional)
```

### Model Sizes

| Size | Embed Dim | Layers | Heads | Parameters | Use Case |
|------|-----------|--------|-------|------------|----------|
| **tiny** | 192 | 12 | 3 | ~5M | Quick experiments, prototyping |
| **small** | 384 | 12 | 6 | ~22M | Balanced performance |
| **base** | 768 | 12 | 12 | ~86M | Best accuracy, research |

For CIFAR (32×32 images):
- **Patch size**: 4×4 (adapted from 16×16 for ImageNet)
- **Num patches**: 64 (8×8 grid)

### Example Configurations

```bash
# Quick test (5 minutes on CPU)
python experiments/vision/train.py --model-size tiny --epochs 5 --attention-types piecewise

# Full comparison (30-60 minutes on CPU)
python experiments/vision/train.py --epochs 50 --attention-types standard linear piecewise

# GPU training with larger model
python experiments/vision/train.py --model-size base --device cuda --epochs 100

# Save results for analysis
python experiments/vision/train.py --save results/vision/cifar10_comparison.json
```

---

## 📈 How the Comparison Works

### Sequential Training

When you specify multiple attention types:

```bash
python experiments/vision/train.py --attention-types standard linear piecewise
```

The script:
1. **Trains Standard**: Full model with standard attention
2. **Trains Linear**: Same model architecture, linear attention only
3. **Trains Piecewise**: Same model architecture, piecewise attention only
4. **Compares Results**: Prints side-by-side comparison table

### What's Kept Constant

For a fair comparison, these are **identical** across all three runs:
- ✅ Model architecture (layers, hidden dims, MLPs)
- ✅ Dataset and data augmentation
- ✅ Batch size and learning rate
- ✅ Number of epochs
- ✅ Random initialization (different per run, but same seed available)

### What Changes

The **only difference** is the attention mechanism in each transformer block:

```python
# Standard Attention
attention_scores = softmax(Q @ K^T / sqrt(d))
output = attention_scores @ V

# Linear Attention
Q_mapped = φ(Q)  # Feature map (ELU+1)
K_mapped = φ(K)
output = Q_mapped @ (K_mapped^T @ V)  # No softmax!

# Piecewise Attention
q_representative = mean(Q)  # Representative query
A_0, J = compute_attention_and_jacobian(q_representative, K, V)
output[i] = A_0 + J @ (Q[i] - q_representative)  # Linear approximation
```

---

## 🎯 Research Questions You Can Answer

### 1. Speed vs Accuracy Tradeoff

**Question**: How much accuracy do we sacrifice for faster training?

**Method**:
```bash
python experiments/vision/train.py --attention-types standard linear piecewise --save results.json
```

**Analysis**: Compare `final_test_accuracy` vs `total_train_time` in results

### 2. Does Piecewise Beat Linear?

**Question**: Is piecewise attention better than linear attention for vision?

**Method**:
```bash
python experiments/vision/train.py --attention-types linear piecewise
```

**Expected**: Piecewise should have higher accuracy with similar speed

### 3. Model Size Effect

**Question**: Do larger models benefit more from efficient attention?

**Method**:
```bash
# Compare tiny vs base
python experiments/vision/train.py --model-size tiny --attention-types standard piecewise
python experiments/vision/train.py --model-size base --attention-types standard piecewise
```

**Analysis**: Compare relative speedups (speedup may increase with model size)

### 4. Convergence Speed

**Question**: Do all attention types converge at the same rate?

**Method**: Load saved JSON and plot learning curves

```python
import json
import matplotlib.pyplot as plt

with open("results.json") as f:
    results = json.load(f)

for result in results:
    history = result["test_history"]
    accuracies = [epoch["accuracy"] for epoch in history]
    plt.plot(accuracies, label=result["attention_type"])

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.show()
```

---

## 💡 Tips & Best Practices

### For Reliable Comparisons

1. **Use same hardware**: Run all experiments on same machine
2. **Close other apps**: Minimize background processes
3. **Multiple runs**: Average across 3-5 runs with different seeds
4. **Warm-up**: First run may be slower (ignore or do warm-up run)

### For Faster Experiments

1. **Start small**: Use `--model-size tiny --epochs 5` for quick tests
2. **Reduce batch size**: If OOM, try `--batch-size 64`
3. **Use GPU**: `--device cuda` is 10-100× faster
4. **Single attention**: Test one type at a time first

### For Publication Results

1. **More epochs**: Use `--epochs 100` or `--epochs 200`
2. **Learning rate tuning**: Try `--lr 0.0003` or `--lr 0.003`
3. **Report statistics**: Mean ± std from multiple runs
4. **Save all results**: Use `--save` for reproducibility

---

## 📚 Related Documentation

- **[TIMING_METRICS.md](../../TIMING_METRICS.md)** - How training time is measured
- **[EXPERIMENTS.md](../../EXPERIMENTS.md)** - Overview of all experiments
- **[piecewise_linear_attention/models/vit.py](../../piecewise_linear_attention/models/vit.py)** - ViT implementation
- **[piecewise_linear_attention/core/attention.py](../../piecewise_linear_attention/core/attention.py)** - Attention mechanisms

---

## ✅ Implementation Status

**Status**: ✅ **READY TO RUN**

All components are implemented:
- ✅ Vision Transformer model
- ✅ Patch embedding layer
- ✅ Position embeddings for 2D patches
- ✅ Classification head
- ✅ Training script with data augmentation
- ✅ Evaluation metrics (accuracy, loss)
- ✅ Timing and throughput tracking
- ✅ Comparison summary output

Ready to run experiments and compare attention mechanisms! 🚀
