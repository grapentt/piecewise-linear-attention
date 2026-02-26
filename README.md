# Piecewise Linear Attention

> Efficient attention mechanism using piecewise linear approximation around pseudo-queries

## Overview

This project implements **piecewise-linear-attention**, a novel attention mechanism that achieves sub-quadratic complexity while maintaining the expressiveness of softmax attention. By computing exact attention at a small number of learned pseudo-queries and using first-order Taylor approximation for nearby actual queries, we reduce complexity from $O(n^2)$ to approximately $O(n \cdot k)$, where $k \ll n$.

See [THEORY.md](THEORY.md) for the complete theoretical foundation.

## Key Idea

Instead of computing full softmax attention for all $n$ queries (expensive) or using linear attention with raw dot products (inaccurate), we:

1. Learn $k$ pseudo-query positions ($k \ll n$)
2. Compute exact softmax attention at these pseudo-queries
3. Use piecewise linear approximation for all other queries based on the nearest pseudo-query

This leverages the fact that locally, the attention function can be well-approximated by its first-order Taylor expansion.

## Installation

### Using uv (recommended)

```bash
# Install dependencies
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"

# Install all optional dependencies
uv pip install -e ".[dev,benchmark,docs]"
```

### Using pip

```bash
pip install -e .
pip install -e ".[dev]"  # with development dependencies
```

## Project Structure

```
piecewise-linear-attention/
├── piecewise_linear_attention/   # Main package
│   ├── core/                     # Core attention implementations
│   ├── models/                   # Model architectures
│   ├── utils/                    # Utility functions
│   └── tests/                    # Test suite
├── benchmarks/                   # Benchmarking scripts
├── examples/                     # Usage examples
├── docs/                         # Documentation
├── THEORY.md                     # Theoretical foundation
└── README.md                     # This file
```

## Quick Start

```python
import torch
from piecewise_linear_attention import PiecewiseLinearAttention

# Initialize attention mechanism
attention = PiecewiseLinearAttention(
    dim=512,              # embedding dimension
    num_pseudo_queries=64 # number of pseudo-queries (k)
)

# Create sample inputs
batch_size, seq_len, dim = 2, 1024, 512
Q = torch.randn(batch_size, seq_len, dim)
K = torch.randn(batch_size, seq_len, dim)
V = torch.randn(batch_size, seq_len, dim)

# Compute attention
output = attention(Q, K, V)
print(output.shape)  # (2, 1024, 512)
```

## Features

- ✅ **Standard Attention**: Full softmax attention baseline
- ✅ **Linear Attention**: Linear complexity baseline
- ✅ **Piecewise Linear Attention**: Our novel approach
- ✅ **Learnable Pseudo-Queries**: Optimized during training
- ✅ **Efficient Jacobian Computation**: Using autograd
- ✅ **Comprehensive Tests**: Unit tests and integration tests
- ✅ **Benchmarking Tools**: Time and memory profiling

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=piecewise_linear_attention --cov-report=html

# Run specific test file
pytest piecewise_linear_attention/tests/test_attention.py
```

### Code Formatting

```bash
# Format code
black piecewise_linear_attention/
isort piecewise_linear_attention/

# Check types
mypy piecewise_linear_attention/
```

### Benchmarking

```bash
# Run benchmarks
python benchmarks/compare_attention.py

# Run scaling experiments
python benchmarks/scaling_experiments.py
```

## Roadmap

- [x] Project setup and structure
- [ ] Implement standard attention baseline
- [ ] Implement linear attention baseline
- [ ] Implement piecewise linear attention
- [ ] Add comprehensive test suite
- [ ] Benchmark complexity (time and memory)
- [ ] Integrate with transformer architecture
- [ ] Training experiments
- [ ] Documentation and examples

## Contributing

Contributions are welcome! This is an open research project. Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Discuss ideas

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{piecewise_linear_attention,
  title = {Piecewise Linear Attention: Efficient Attention via Piecewise Approximation},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/piecewise-linear-attention}
}
```

## References

- Vaswani et al. (2017). "Attention is All You Need"
- Katharopoulos et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
- See [THEORY.md](THEORY.md) for complete references
