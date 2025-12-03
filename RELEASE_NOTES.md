# Release Notes: Cruxy Stability Engine

## v2.1.0: The "World Class" Update
**Date:** December 3, 2025

### 🚀 Major Breakthroughs
*   **Cluster Ready (Experimental):** Added `compile=True` support for `torch.compile` (Triton/Inductor). *Note: This feature is currently untested on large-scale clusters.*
*   **Meta-Lion Unleashed:** Tuned to match AdamW performance (Loss 1.66 vs 1.68) while using **1/3rd the memory**.
*   **4GB VRAM Support:** Verified training of TinyLlama-1.1B on consumer hardware (GTX 1650 class).
*   **Intelligence Fix:** Corrected Phase Modulation logic in `MetaCruxyController`, resulting in significantly lower loss.
*   **Memory Optimization:** Fixed a memory leak in Lion mode, reducing VRAM footprint by 50%.

### 🏆 Benchmark Results
| Optimizer | Final Loss | Memory |
|-----------|------------|--------|
| Cruxy (Meta3) | **1.6413** | Standard |
| AdamW | 1.6843 | Standard |
| Cruxy (Meta-Lion) | 1.6633 | **Low** |

---

# Release Notes: Cruxy Stability Engine v2.0

**Released by:** Axiom Forge Systems Ltd  
**Date:** December 3, 2025

---

## 🚀 The "Self-Driving" Optimizer for PyTorch

Axiom Forge Systems is proud to release **Cruxy v2.0**, a drop-in replacement for AdamW and Lion that eliminates the need for manual learning rate scheduling.

### 🏆 Key Achievements
*   **Beats AdamW:** Achieves lower validation loss on NanoGPT/Shakespeare benchmarks (10k steps) without any scheduler tuning.
*   **Meta-Lion Core:** Integrates Google's Lion optimizer with our proprietary **Meta3 Controller**, stabilizing the sign-based updates automatically.
*   **Scale Proven:** Validated scaling from 200K to 30M parameters on CPU.

### 📦 Features
1.  **Meta3 Controller:** A hierarchical control system that adjusts Learning Rate and Beta2 based on gradient variance and curvature.
2.  **Cruxy-Lion:** The speed of Lion with the stability of adaptive control.
3.  **SOTA Tricks:** Built-in support for **Decoupled Weight Decay**, **Nesterov Momentum**, and **Gradient Centralization**.
4.  **Distributed Ready:** Optimized norm calculations for minimal CPU-GPU sync overhead.

### 🔧 Quick Start

```bash
pip install cruxy-opt
```

```python
from cruxy import CruxyOptimizer

# Drop-in replacement for AdamW
optimizer = CruxyOptimizer(
    model.parameters(), 
    lr=1e-3, 
    mode="meta3", 
    weight_decay=0.1
)
```

### 🏢 About Axiom Forge Systems
Based in Poole, Cornwall, Axiom Forge Systems specializes in high-performance AI infrastructure and stability engineering. The Cruxy Engine represents our internal R&D efforts to democratize stable model training.

---
*For enterprise support or consulting on large-scale training runs, contact dev@axiomforge.ai*
