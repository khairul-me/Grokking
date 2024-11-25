# Neural Network Grokking Research 🧠
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)

## Overview 📖
This research project investigates the grokking phenomenon in neural networks - where models suddenly achieve generalization after extended training. We focus on modular arithmetic tasks, implementing transformer architectures to understand this behavior.

## What is Grokking? 🤔
Grokking refers to a phenomenon where neural networks:
1. Initially memorize training data (high train accuracy, low test accuracy)
2. Go through an extended training period with no apparent improvement
3. Suddenly "understand" the underlying pattern (achieve high test accuracy)

## Features ⭐
- Implementation of modular arithmetic tasks (addition, subtraction)
- Custom transformer architecture
- HookedTransformer implementation
- Comprehensive analysis tools
- Visualization of learning dynamics
- Fourier analysis capabilities

## Project Structure 📁

grokking/
├── src/
│   ├── grokking.py          # HookedTransformer implementation
│   ├── transformer.py       # Custom transformer architecture
│   └── train_test.py        # Training utilities
├── notebooks/
│   └── analysis.ipynb       # Analysis and visualizations
├── data/
│   └── modular_arithmetic/  # Generated datasets
└── results/
└── checkpoints/         # Model checkpoints

## Setup and Installation 💻

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU support)

### Installation Steps
```bash
# Clone repository
git clone https://github.com/[your-username]/grokking-research.git
cd grokking-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Required Packages
torch>=1.9.0
einops>=0.3.0
transformer_lens>=0.6.0
tqdm>=4.62.0
matplotlib>=3.4.0
numpy>=1.21.0

Usage 🚀
Basic Training

# Run modular addition experiment
python src/grokking.py --prime 113 --train_frac 0.3 --epochs 25000

Key Parameters
# Model configuration
cfg = HookedTransformerConfig(
    n_layers = 1,
    n_heads = 4,
    d_model = 128,
    d_head = 32,
    d_mlp = 512,
    act_fn = "relu",
    normalization_type = None,
    d_vocab = p+2,
    d_vocab_out = p,
    n_ctx = 3
)

# Training parameters
lr = 1e-3          # Learning rate
wd = 1.0           # Weight decay
frac_train = 0.3   # Training data fraction
Experiment Results 📊
Training Dynamics
The project includes tools for visualizing:

Training and test loss curves
Weight evolution patterns
Fourier component analysis
Geometric representations of modular arithmetic

Key Findings

Network behavior shows distinct phases:

Memorization
Circuit formation
Cleanup/generalization


Critical factors for grokking:

Weight decay strength
Training data fraction
Model architecture



Troubleshooting 🔧
