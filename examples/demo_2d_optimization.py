import torch
import numpy as np
import matplotlib.pyplot as plt
from cruxy import CruxyOptimizer

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def run_optimization(optimizer_class, name, steps=500, lr=1e-3, **kwargs):
    # Start at a difficult point (-2, 2)
    params = torch.tensor([-2.0, 2.0], requires_grad=True)
    
    if name == "Adam":
        opt = torch.optim.Adam([params], lr=lr)
    else:
        opt = optimizer_class([params], lr=lr, **kwargs)
        
    path = []
    lrs = []
    
    for _ in range(steps):
        path.append(params.detach().numpy().copy())
        
        opt.zero_grad()
        loss = rosenbrock(params[0], params[1])
        loss.backward()
        
        if name == "Adam":
            opt.step()
            lrs.append(lr)
        else:
            # Cruxy needs loss for curvature
            opt.step(loss=loss.item())
            # Log dynamic LR
            if hasattr(opt, 'controller') and hasattr(opt.controller, 'current_lr'):
                lrs.append(opt.controller.current_lr)
            elif hasattr(opt, 'controller') and hasattr(opt.controller, 'prev_lr'):
                 lrs.append(opt.controller.prev_lr)
            else:
                lrs.append(lr)
                
    return np.array(path), np.array(lrs)

def main():
    print("Running optimization demo on Rosenbrock function...")
    
    # Run Adam
    print("Running Adam...")
    path_adam, lrs_adam = run_optimization(None, "Adam", lr=0.01)
    
    # Run Cruxy Meta3
    print("Running Cruxy Meta3...")
    path_cruxy, lrs_cruxy = run_optimization(
        CruxyOptimizer, 
        "Cruxy Meta3", 
        mode="meta3", 
        lr=0.01,
        meta_interval=5 # Frequent updates for this short demo
    )
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Trajectory Plot
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='gray', alpha=0.5)
    ax1.plot(path_adam[:, 0], path_adam[:, 1], 'r.-', label='Adam', alpha=0.7)
    ax1.plot(path_cruxy[:, 0], path_cruxy[:, 1], 'b.-', label='Cruxy Meta3', alpha=0.7)
    ax1.plot(1, 1, 'g*', markersize=15, label='Global Min')
    ax1.set_title("Optimization Trajectory (Rosenbrock)")
    ax1.legend()
    
    # 2. Learning Rate Adaptation
    ax2.plot(lrs_adam, 'r--', label='Adam (Fixed)')
    ax2.plot(lrs_cruxy, 'b-', label='Cruxy Meta3 (Adaptive)')
    ax2.set_title("Learning Rate Adaptation")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Learning Rate")
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.tight_layout()
    print("Close the plot window to finish.")
    plt.show()

if __name__ == "__main__":
    main()
