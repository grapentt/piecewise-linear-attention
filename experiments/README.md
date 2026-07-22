# Transformer Integration Experiments

End-to-end experiments that plug the attention mechanisms into full models on real tasks. Each subdirectory is self-contained with its own `train.py` and a Colab notebook.

| Task | Model | Dataset | Script |
|------|-------|---------|--------|
| Translation | Encoder–decoder Transformer | Multi30K (de→en) | `translation/train.py` |
| Vision | Vision Transformer (ViT) | CIFAR-10 | `vision/train.py` |
| Text classification | BERT-style encoder | GLUE SST-2 | `bert/train.py` |

All scripts select the attention variant with `--attention-types` and accept `standard`, `linear`, `performer`, and the piecewise variants (`piecewise`, `piecewise_kmeans`, `piecewise_stride`).

## Quick start

```bash
# Install with experiment dependencies (from the repo root)
uv pip install -e ".[experiments]"

# Translation — defaults train on a Multi30K subset
python experiments/translation/train.py --epochs 10 --save results/translation/results.json

# Vision — ViT on CIFAR-10
python experiments/vision/train.py --attention-types piecewise --save results/vision/cifar10.json

# Text classification — BERT on SST-2
python experiments/bert/train.py --attention-types piecewise --save results/bert/sst2.json
```

Compare multiple variants in one run by passing several to `--attention-types`, e.g.
`--attention-types standard piecewise performer`.

## On GPU

Pass `--device cuda` (where the script supports it) for realistic speedups; CPU/MPS runs are fine for correctness and small-scale comparisons but are not representative of production throughput.

## Notebooks

Each task has a Colab notebook (`colab_*.ipynb`) that clones the repo (with branch selection), installs dependencies, and runs the comparison with free GPU. Links are in the main [README](../README.md#try-it-on-google-colab).

## Shared utilities

`utils/` holds the pieces the task scripts share:

- `training.py` — `train_epoch`, `evaluate` loops.
- `data.py` — tokenization, vocabulary building, dataset/collate helpers.

## Status and caveats

These integrations exist to move the method beyond synthetic microbenchmarks toward real-data task quality. Rigorous, versioned real-task results (matched budgets, multiple seeds, CUDA timing, and comparison against strong efficient-attention baselines) are ongoing work tracked in the repository issues — treat any single run here as a smoke test of the integration, not a benchmark claim.

## See also

- Reproducible synthetic evidence: [`../benchmarks/`](../benchmarks/) and [`../results/`](../results/).
- Method and derivation: [`../THEORY.md`](../THEORY.md).
