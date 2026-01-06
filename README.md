# PrisML

PrisML is a from-scratch **machine learning** library in Python, built to learn and demonstrate core ML algorithms.

## Goals

- Educational, readable implementations
- Minimal but practical use of NumPy
- Classical ML + basic neural networks

## Installation

Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/tiagosantosmr/prisml.git
cd prisml
pip install -e ".[dev]"
```

## Quick Start

```python
from prisml import Tensor

# Create a tensor from a list
t1 = Tensor([1, 2, 3])
print(t1.shape)  # (3,)

# Create a tensor from a scalar
t2 = Tensor(5.0)

# Element-wise operations with broadcasting
result = t1 + t2
print(result.data)  # [6, 7, 8]
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
prisml/
├── prisml/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── tensor.py
│   ├── linear_models/
│   ├── neural_nets/
│   └── ...
├── tests/
├── pyproject.toml
└── README.md
```

## Features (In Development)

- [x] Multi-dimensional Tensor class
- [x] NumPy-backed operations with broadcasting
- [ ] Linear Regression
- [ ] Logistic Regression
- [ ] Decision Trees
- [ ] Neural Networks
- [ ] Automatic Differentiation (Autograd)

## Contributing

This is a personal learning project. While I appreciate interest, I'm building this from scratch to learn ML fundamentals. I'm not accepting pull requests at this time, but feel free to fork and create your own version!

## License

MIT License - see LICENSE file for details.