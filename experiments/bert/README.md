# BERT Experiments

BERT (Bidirectional Encoder) experiments comparing attention mechanisms on language understanding tasks.

## Quick Start

```bash
# Train BERT with all three attention types
python experiments/bert/train.py

# Train on specific task
python experiments/bert/train.py --task glue/sst2  # Sentiment classification

# Specific attention type
python experiments/bert/train.py --attention-types piecewise --epochs 3

# Save results
python experiments/bert/train.py --save results/bert/sst2_comparison.json
```

## Model Architecture

- **BERT encoder**: Bidirectional transformer with configurable attention
- **Masked language modeling** (pretraining)
- **Classification head** (finetuning)
- **12 layers, 12 heads, 768 hidden dim** (BERT-base config)

## Tasks

Supported tasks:
- **GLUE Benchmark**: SST-2, MRPC, QQP, MNLI, etc.
- **SQuAD**: Question answering
- **NER**: Named entity recognition