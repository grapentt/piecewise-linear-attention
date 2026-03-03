# Translation Experiments

Encoder-decoder transformer experiments comparing attention mechanisms on machine translation tasks.

## Quick Start

```bash
# Train with all three attention types (standard, linear, piecewise)
python experiments/translation/train.py

# Train with specific attention type
python experiments/translation/train.py --attention-types piecewise

# Custom dataset (German-English Multi30K by default)
python experiments/translation/train.py --dataset bentrevett/multi30k --subset de-en --src de --tgt en

# More training epochs
python experiments/translation/train.py --epochs 10 --batch-size 64

# Save results
python experiments/translation/train.py --save results/translation/multi30k_comparison.json
```

## Model Architecture

- **Encoder**: Bidirectional transformer with configurable attention
- **Decoder**: Causal transformer with cross-attention to encoder
- **Multi-head attention**: 4 heads by default
- **Layers**: 3 encoder + 3 decoder by default
- **Hidden dim**: 256 by default

## Attention Implementations

All three attention types are compared:

1. **StandardAttention**: Full softmax attention (baseline)
   - Complexity: O(n²d)
   - Best accuracy

2. **LinearAttention**: Kernel-based linear attention
   - Complexity: O(nd²)
   - Fastest but lower accuracy

3. **PiecewiseAttention**: First-order Taylor approximation
   - Complexity: O(nd²)
   - Best speed/accuracy tradeoff

## Datasets

Supported datasets (via HuggingFace):
- `bentrevett/multi30k` - German↔English (default)
- `opus_books` - Multi-language book translations
- `wmt14` - WMT14 translation task
- Custom datasets via HuggingFace datasets

## Results

Results are saved to `results/translation/` in JSON format with:
- Training/validation loss & perplexity curves
- Training time and throughput
- Model hyperparameters
- Per-epoch metrics

## Command Line Arguments

```
--dataset           HuggingFace dataset name (default: bentrevett/multi30k)
--subset            Dataset subset (default: de-en)
--src               Source language (default: de)
--tgt               Target language (default: en)
--hidden-dim        Model dimension (default: 256)
--num-layers        Encoder/decoder layers (default: 3)
--num-heads         Attention heads (default: 4)
--epochs            Training epochs (default: 5)
--batch-size        Batch size (default: 32)
--lr                Learning rate (default: 0.0001)
--attention-types   Which attention types to compare (default: all)
--device            Device (default: cpu)
--save              Path to save results JSON
```
