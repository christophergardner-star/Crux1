# Cruxy Stability Engine - Examples

This folder contains lightweight examples you can run on a standard laptop to demonstrate the capabilities of the Cruxy Optimizer.

## Prerequisites

Ensure you have the package installed (or `src` in your PYTHONPATH) and the required dependencies:

```bash
pip install -e .
pip install matplotlib torchvision
```

## 1. 2D Optimization Demo (`demo_2d_optimization.py`)

A visual demonstration comparing **Adam** vs **Cruxy Meta3** on the Rosenbrock function (a non-convex function used to test optimization algorithms).

**What to look for:**
- **Adam** (Red): Often oscillates or takes a suboptimal path.
- **Cruxy Meta3** (Blue): Adapts its learning rate based on curvature and variance, often finding a smoother path to the global minimum.
- **LR Plot**: Observe how Cruxy dynamically adjusts its learning rate without a scheduler.

**Run it:**
```bash
python examples/demo_2d_optimization.py
```

## 2. MNIST Training Demo (`demo_mnist.py`)

A practical Deep Learning example training a simple Convolutional Neural Network (ConvNet) on the MNIST dataset (handwritten digits).

**Features:**
- Uses `CruxyOptimizer` in `meta3` mode.
- Demonstrates "Schedule-Free" training (no LR scheduler used).
- Logs the dynamic Learning Rate during training.

**Run it:**
```bash
python examples/demo_mnist.py
```

## 3. SOTA MNIST Demo (`demo_sota_mnist.py`)

Demonstrates the "World Beating" configuration with:
- **Decoupled Weight Decay (AdamW style)**: Better generalization.
- **Nesterov Momentum**: Faster convergence.
- **Gradient Centralization**: SOTA for Vision tasks.
- **Meta3 Controller**: Adaptive stability.

**Run it:**
```bash
python examples/demo_sota_mnist.py
```

## 4. Meta-Lion Demo (`demo_meta_lion.py`)

Demonstrates the **"Meta-Lion"** configuration:
- **Lion Optimizer Core**: Uses sign-based updates (memory efficient, fast).
- **Meta3 Controller**: Automatically tunes the Learning Rate for Lion (which is usually hard to tune).
- **Gradient Centralization**: Applied on top for extra performance.

**Run it:**
```bash
python examples/demo_meta_lion.py
```
