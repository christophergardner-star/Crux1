# Cruxy Stability Engine SDK

The Cruxy Stability Engine is an adaptive optimization framework for neural network training, implementing the algorithms described in the Axiom Forge Systems Ltd White Paper (v2.0 + Meta-Cruxy 3.0).

## 🚀 Verified Performance
**Runs on 4GB VRAM GPUs**
- **Model:** TinyLlama-1.1B (Float16 + LoRA)
- **Optimizer:** Cruxy Meta-Lion
- **Result:** Converged in 26s (Loss 5.0 -> 0.0003) on GTX 1650-class hardware.
- **Demo:** `python examples/demo_tinyllama_4gb.py`

## 🏆 Benchmark Results (Shakespeare GPT)
*Verified on NVIDIA GPU (Dec 2025)*

| Optimizer | Final Loss | Time | Memory | Notes |
|-----------|------------|------|--------|-------|
| **Cruxy (Meta3)** | **1.6413** | 26s | Standard | **Most Intelligent** (Beat AdamW) |
| AdamW (Baseline) | 1.6843 | 10s | Standard | Baseline |
| **Cruxy (Meta-Lion)** | **1.6633** | 26s | **Low (1/3x)** | **Best for 4GB Cards** (Stable @ LR=1e-3) |

*Note: Meta-Lion achieved near-parity with AdamW while using significantly less memory, enabling LLM training on consumer hardware.*

## Features

- **Dual-Window Variance Monitoring**: Detects training phases (volatility vs convergence).
- **Curvature-Adaptive Momentum**: Adjusts beta1 based on loss landscape geometry.
- **Meta-Optimization (Meta-Cruxy 3.0)**: Schedule-free training via hierarchical control of learning rate and momentum.
- **Predictive Gradient Clipping**: Variance-informed clipping thresholds.
- **Safety Guards**: Automatic NaN/Inf detection and hyperparameter clamping.

## Installation

```bash
pip install .
```

## Usage

### Basic Usage

Use `CruxyOptimizer` as a drop-in replacement for `torch.optim.AdamW`.

```python
import torch
from cruxy import CruxyOptimizer

model = torch.nn.Linear(10, 1)
optimizer = CruxyOptimizer(
    model.parameters(), 
    mode="meta3", # Options: "stability_v1", "stability_v2", "meta3"
    lr=1e-3
)

# Training Loop
for batch in dataloader:
    inputs, targets = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.MSELoss()(outputs, targets)
    loss.backward()
    
    # Pass loss to step() for curvature estimation
    optimizer.step(loss=loss.item())
```

### HuggingFace Trainer Integration

```python
from transformers import Trainer, TrainingArguments
from cruxy import CruxyOptimizer

class CruxyTrainer(Trainer):
    def create_optimizer(self):
        self.optimizer = CruxyOptimizer(
            self.model.parameters(),
            mode="meta3",
            lr=self.args.learning_rate
        )
        return self.optimizer

trainer = CruxyTrainer(...)
trainer.train()
```

## Modes

1.  **stability_v1**: Baseline dual-window variance monitoring with PD control.
2.  **stability_v2**: Adds curvature adaptation, gamma-norm variance, and predictive clipping.
3.  **meta3**: Adds meta-controller for schedule-free training (Recommended).

## Testing

Run the test suite:

```bash
pytest tests/
```
