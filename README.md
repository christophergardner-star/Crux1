<p align="center">
  <img src="docs/img/logo.png" alt="Axiom Forge Systems Logo" width="300"/>
</p>

> **Train 1.5B parameter models on a 4GB GPU. No scheduler tuning required.**  
> *[Read the White Paper (v2.1)](docs/Cruxy_Stability_Engine_Whitepaper_v2.1.pdf)*

# Cruxy Stability Engine SDK

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/status-stable-green)
[![Whitepaper](https://img.shields.io/badge/Whitepaper-Read_PDF-b31b1b?logo=adobeacrobatreader&logoColor=white)](docs/Cruxy_Stability_Engine_Whitepaper_v2.1.pdf)

The Cruxy Stability Engine is an adaptive optimization framework for neural network training, implementing the algorithms described in the Axiom Forge Systems Ltd White Paper (v2.0 + Meta-Cruxy 3.0).

> **Note:** Additional validation testing has been conducted since the white paper was published, including successful training runs on models up to 14B parameters on NVIDIA H200 hardware via RunPod.

## 🚀 Verified Performance

### H200 Benchmarks (141GB VRAM)
*Verified on NVIDIA H200, December 2025*

| Model | Params | Status | Notes |
|-------|--------|--------|-------|
| **TinyLlama** | 1.1B | ✅ Verified | - |
| **Qwen 2.5** | 1.5B | ✅ Verified | - |
| **SmolLM2** | 1.7B | ✅ Verified | 500-step stability test passed |
| **Phi-2** | 2.7B | ✅ Verified | - |
| **Qwen 2.5** | 7B | ✅ Verified | 99.5% loss reduction, matches AdamW |
| **Qwen 2.5** | 14B | ✅ Verified | 123GB VRAM, 100% loss reduction |

### Cruxy vs AdamW (H200)

| Model | Optimizer | Initial Loss | Final Loss | Reduction |
|-------|-----------|--------------|------------|-----------|
| Qwen 2.5 7B | **Cruxy** | 5.47 | 0.025 | **99.5%** |
| Qwen 2.5 7B | AdamW | 5.47 | 0.026 | 99.5% |
| SmolLM2 1.7B (500 steps) | **Cruxy** | 4.75 | 0.040 | **99.2%** |
| SmolLM2 1.7B (500 steps) | AdamW | 4.75 | 0.041 | 99.1% |

### 4GB VRAM (Consumer Hardware)
*Verified on GTX 1650, December 2025*

| Model | Params | Config | Status | Demo Script |
|-------|--------|--------|--------|-------------|
| **TinyLlama** | 1.1B | Float16 + LoRA | ✅ Verified | `examples/demo_tinyllama_4gb.py` |
| **Qwen 2.5** | 1.5B | Float16 + LoRA | ✅ Verified | `examples/demo_qwen_4gb.py` |
| **Gemma** | 2B | 4-bit + LoRA | ⚠️ Untested | `examples/demo_gemma2b_4gb.py` |
| **Phi-2** | 2.7B | 4-bit + LoRA | ⚠️ Untested | `examples/demo_phi2_4gb.py` |

*Note: 4-bit quantization requires `bitsandbytes`. Without it, models >1.5B may require CPU offloading.*

## 🏆 Benchmark Results (Shakespeare GPT)
*Verified on NVIDIA GPU (Dec 2025)*

| Optimizer | Final Loss | Time | Memory | Notes |
|-----------|------------|------|--------|-------|
| **Cruxy (Meta3)** | **1.6413** | 26s | Standard | **Most Intelligent** (Beat AdamW) |
| AdamW (Baseline) | 1.6843 | 10s | Standard | Baseline |
| **Cruxy (Meta-Lion)** | **1.6633** | 26s | **Low (1/3x)** | **Best for 4GB Cards** (Stable @ LR=1e-3) |

![Hero Chart](docs/img/hero_chart.png)

*Note: Meta-Lion achieved near-parity with AdamW while using significantly less memory, enabling LLM training on consumer hardware.*

## Features

- **Dual-Window Variance Monitoring**: Detects training phases (volatility vs convergence).
- **Curvature-Adaptive Momentum**: Adjusts beta1 based on loss landscape geometry.
- **Meta-Optimization (Meta-Cruxy 3.0)**: Schedule-free training via hierarchical control of learning rate and momentum.
- **Predictive Gradient Clipping**: Variance-informed clipping thresholds.
- **Safety Guards**: Automatic NaN/Inf detection and hyperparameter clamping.
- **Cluster Ready (Experimental)**: Supports `torch.compile` for high-performance training (untested on clusters).

## Installation

```bash
pip install cruxy
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
4.  **Meta-Lion**: Activate by setting `mode="meta3"` and `use_lion=True`. Combines the memory efficiency of Lion with the stability of the Meta3 controller.

## Testing

Run the test suite:

```bash
pytest tests/
```
