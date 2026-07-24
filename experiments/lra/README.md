# Long Range Arena (LRA) Benchmarks

This directory contains PyTorch implementations of the Long Range Arena benchmarks for evaluating piecewise linear attention.

## ListOps

The ListOps task evaluates the model's ability to parse and execute hierarchical list operations.

### Fair comparison across all attention methods

Every attention mechanism is trained **end-to-end** inside the same encoder, so the
learned-projection baselines (Linformer's `E`/`F`, Luna's pack queries) are actually
optimised rather than scored at random initialisation. Each method is given a
matched hyperparameter budget (comparable feature/landmark/rank/pack sizes; see
`configs/base_config.py::attention_hparams`), and padded key positions are excluded
correctly per method — masking the softmax scores for softmax-over-keys methods, and
masking the feature map `φ(K)` for the kernel methods, since a zeroed key row still
leaves a non-zero softmax weight / `φ(0)` in the normalizer. This makes the
comparison fair on both the accuracy and the padding-handling axes.

To keep the budget honest for Nyström — whose segment-mean landmarks require the
sequence length to be divisible by the landmark count — the encoder right-pads the
(CLS-extended) sequence to a multiple of that count (`pad_to_multiple_of`), applied
uniformly to every method so all train on the identical input. Without it, a
near-prime task length (e.g. 2000 + 1 CLS = 2001) would silently collapse Nyström to
far fewer landmarks than its peers' budget. The padded positions are masked, so they
cannot affect any output (a registry-wide leak test in
`piecewise_linear_attention/tests/test_key_padding_mask.py` locks this in — padded
values must not change any valid-position output, on both the causal and non-causal
paths). Note that `piecewise_kmeans`'s `num_anchors` is a linearization-granularity
dial, **not** a rank/landmark budget, and is reported separately rather than under the
shared budget.

### Quick Start

```bash
# Generate dataset (if not already present)
python experiments/lra/data/generate_listops.py --output-dir data/lra/listops

# Quick smoke (1 epoch, short sequences, all methods)
python experiments/lra/tasks/listops/train.py \
    --epochs 1 \
    --max-length 128 \
    --batch-size 16 \
    --device cpu

# Full benchmark (all methods, 20 epochs)
python experiments/lra/tasks/listops/train.py \
    --epochs 20 \
    --batch-size 32 \
    --device cuda
```

`--attention-types` defaults to all methods (`standard linear performer
nystromformer linformer luna piecewise`); pass a subset to compare fewer.

### Experiment tracking (optional)

The trainer can log to [MLflow](https://mlflow.org) with `--mlflow` (off by
default). It is an **optional** dependency: without it the trainer behaves
identically, and if `--mlflow` is passed but the package is not installed the run
prints a warning and continues untracked. Install it via the extra:

```bash
pip install -e ".[tracking]"   # adds mlflow
```

Each run records one nested MLflow run per attention method (grouped under a
parent run) with the method's matched hyperparameters, precision, padded sequence
length, per-epoch train/val loss+accuracy, and final test metrics; the aggregate
`results.json` is attached as an artifact. The tracking store defaults to a local
SQLite file (`./mlflow.db`) — self-contained and server-free — because MLflow has
deprecated the bare-directory file store; point `--mlflow-tracking-uri` at an
`http://` tracking server to log remotely.

```bash
python experiments/lra/tasks/listops/train.py \
    --epochs 10 --batch-size 32 --device cuda --precision bf16 \
    --mlflow --mlflow-experiment lra-listops --mlflow-run-name anchors-m1-m4

# Browse locally:
mlflow ui --backend-store-uri sqlite:///mlflow.db
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

### Tests

```bash
python -m pytest experiments/lra/tests/test_smoke.py
```

The smoke tests build every method, run a forward+backward, and assert that padded
token values cannot change the output (mask correctness) and that masking keys does
change the output (the mask is actually wired through).

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

The only difference is the **attention mechanism**, selected from the shared
registry (`standard`, `linear`, `performer`, `nystromformer`, `linformer`, `luna`,
`piecewise`).

### Directory Structure

```
experiments/lra/
├── configs/
│   └── base_config.py          # LRA config + matched per-method hyperparameters
├── models/
│   └── encoder.py              # Generic LRA encoder (threads attn_kwargs + mask)
├── data/
│   ├── listops.py              # ListOps dataset loader
│   └── generate_listops.py     # Local dataset generator
├── tasks/
│   └── listops/
│       └── train.py            # Training script (all methods)
├── tests/
│   └── test_smoke.py           # Build / mask-correctness smoke tests
└── utils/
    └── training.py             # Training utilities
```

## Roadmap

- **ListOps** (current)
- Text (IMDB) + Image (CIFAR-10 grayscale)
- Retrieval (AAN) + Pathfinder

## References

- [Long Range Arena Paper](https://openreview.net/forum?id=qVyeW-grC2k)
- [LRA GitHub](https://github.com/google-research/long-range-arena)
- [Piecewise Linear Attention](../../README.md)
