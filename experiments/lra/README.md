# Long Range Arena (LRA) Benchmarks

This directory contains PyTorch implementations of the Long Range Arena benchmarks for evaluating piecewise linear attention.

## Phase 1: ListOps (Current)

The ListOps task evaluates the model's ability to parse and execute hierarchical list operations.

### Quick Start

```bash
# Generate dataset (if not already present)
python experiments/lra/data/generate_listops.py --output-dir data/lra/listops

# Quick test (2 epochs, single attention type)
python experiments/lra/tasks/listops/train.py \
    --attention-types piecewise \
    --epochs 2 \
    --device mps \
    --batch-size 16

# Full benchmark (all attention types, 20 epochs)
python experiments/lra/tasks/listops/train.py \
    --attention-types standard linear piecewise \
    --epochs 20 \
    --batch-size 32 \
    --device mps
```

### Dataset

**ListOps** is a diagnostic task that requires parsing hierarchical nested expressions like `[MAX 2 9 [MIN 4 7] 0]`.

- **Task**: 10-way classification (results are digits 0-9)
- **Sequence length**: ~500-2000 tokens (character-level)
- **Dataset size**: 96K train, 2K val, 2K test
- **Operations**: MAX, MIN, MEDIAN, SUM_MOD

**Dataset Generation**: The GCS bucket (`gs://long-range-arena/lra_release`) currently has restricted access (403 Forbidden), so we generate the dataset locally using the official LRA generation code. This ensures reproducibility and matches the official benchmark exactly.

To manually generate the dataset:

```bash
python experiments/lra/data/generate_listops.py \
    --output-dir data/lra/listops \
    --num-train 96000 \
    --num-val 2000 \
    --num-test 2000 \
    --seed 42
```

### Results

TODO

### Architecture

All models use the same encoder-only transformer architecture:

```python
- Token embeddings (character-level)
- Positional embeddings (learnable)
- 4 transformer layers
- 8 attention heads
- 512 hidden dim
- 1024 MLP dim
- CLS token pooling
- Classification head (10 classes)
```

The only difference is the **attention mechanism** (standard, linear, or piecewise).

### Directory Structure

```
experiments/lra/
├── configs/
│   └── base_config.py          # LRA standard configuration
├── models/
│   └── encoder.py              # Generic LRA encoder
├── data/
│   ├── listops.py              # ListOps dataset loader
│   └── generate_listops.py     # Local dataset generator
├── tasks/
│   └── listops/
│       └── train.py            # Training script
└── utils/
    └── training.py             # Training utilities
```

## Roadmap

**Phase 1** (Current): ListOps
**Phase 2**: Text (IMDB) + Image (CIFAR-10 grayscale)
**Phase 3**: Retrieval (AAN) + Pathfinder
**Phase 4**: Colab notebook + documentation

## References

- [Long Range Arena Paper](https://openreview.net/forum?id=qVyeW-grC2k)
- [LRA GitHub](https://github.com/google-research/long-range-arena)
- [Piecewise Linear Attention](../../README.md)
