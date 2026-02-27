# Transformer Integration Experiments

This directory contains experiments comparing PiecewiseAttention against StandardAttention in a full transformer architecture for German→English machine translation.

## Quick Start

### 1. Run Quick Test (5-10 minutes)

```bash
# Test piecewise attention only, minimal data
python train_comparison.py \
  --attention piecewise \
  --epochs 3 \
  --batch-size 32 \
  --max-train 1000 \
  --max-eval 100 \
  --device cpu
```

### 2. Run Full Comparison (30-60 minutes)

```bash
# Train both standard and piecewise, compare results
python train_comparison.py \
  --attention both \
  --epochs 5 \
  --batch-size 32 \
  --max-train 2000 \
  --max-eval 200 \
  --device cpu
```

### 3. Evaluate and Plot

```bash
# Generate comparison plots and statistics
python evaluate_comparison.py
```

## Files

- `multihead.py` (to be created) - Multi-head attention wrapper compatible with Homework 5
- `transformer_homework5.py` (to be created) - Modified transformer with pluggable attention
- `train_comparison.py` (to be created) - Training script for both attention types
- `evaluate_comparison.py` (to be created) - Evaluation and plotting
- `results/` - Saved models and results (created automatically)

## Directory Structure

```
experiments/
├── README.md                    # This file
├── multihead.py                 # Multi-head wrapper for PiecewiseAttention
├── transformer_homework5.py     # Configurable transformer
├── train_comparison.py          # Training script
├── evaluate_comparison.py       # Evaluation script
└── results/                     # Output directory
    ├── standard/
    │   ├── best_model.pt
    │   └── results.json
    ├── piecewise/
    │   ├── best_model.pt
    │   └── results.json
    └── comparison.png
```

## Expected Output

After running experiments, you'll get:

1. **Trained models**: `results/{standard,piecewise}/best_model.pt`
2. **Results JSON**: Training/eval losses, timing info
3. **Comparison plot**: Side-by-side training curves
4. **Console summary**: Speedup, perplexity comparison

## Integration Guide

For complete implementation details, see:
**`/Users/I745505/private/piecewise-linear-attention/INTEGRATION_WITH_TRANSFORMER.md`**

That document includes:
- Full code implementations
- Architecture diagrams
- Troubleshooting guide
- Expected results
- Next steps

## Requirements

Before running experiments:

1. **Install piecewise-linear-attention**:
   ```bash
   cd /Users/I745505/private/piecewise-linear-attention
   uv pip install -e .
   ```

2. **Dataset access**: Will automatically download Multi30k from HuggingFace

3. **Homework 5 code**: Must exist at:
   `/Users/I745505/private/NLP/exam_prep/homework_solutions/nlp-exercise-5-solution-main/`

## Configuration Options

All scripts support these arguments:

- `--attention`: `standard`, `piecewise`, or `both`
- `--device`: `cpu` or `cuda`
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 64)
- `--max-train`: Max training samples (default: 5000)
- `--max-eval`: Max eval samples (default: 500)

## GPU Training

To run on GPU (much faster):

```bash
python train_comparison.py \
  --attention both \
  --epochs 10 \
  --batch-size 64 \
  --device cuda
```

## Understanding Results

### Good Results Example

```
Standard Attention:
  Best loss: 1.50, Perplexity: 4.5, Time: 300s

Piecewise Attention:
  Best loss: 1.65, Perplexity: 5.2, Time: 75s

Speedup: 4.0×
Perplexity increase: 15% → GOOD TRADEOFF
```

### Success Criteria

- **Minimal**: Loss decreases, perplexity < 20
- **Good**: Perplexity within 20% of baseline, 2×+ speedup
- **Strong**: Perplexity within 10% of baseline, 5×+ speedup

## Troubleshooting

### Import errors
```
ModuleNotFoundError: No module named 'piecewise_linear_attention'
```
→ Run `uv pip install -e .` from project root

### Shape mismatch
```
RuntimeError: shape mismatch in attention
```
→ Check mask dimensions, verify Q/K/V shapes match expected

### NaN loss
```
Loss becomes NaN after few iterations
```
→ Lower learning rate, add gradient clipping, check for numerical issues

See full troubleshooting guide in INTEGRATION_WITH_TRANSFORMER.md

## Next Steps

After successful integration:

1. **GPU benchmarks**: Run on CUDA for realistic speedup measurements
2. **Scale up**: Train on full dataset, larger models
3. **Error analysis**: Understand where approximation causes issues
4. **Optimization**: Try different pseudo-query selection strategies

## Contact

For questions or issues, refer to:
- Main integration guide: `INTEGRATION_WITH_TRANSFORMER.md`
- Theory document: `../THEORY.md`
- Project handover: `../HANDOVER.md`
