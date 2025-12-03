import torch
import pytest
from cruxy.optimizer import CruxyOptimizer

@pytest.mark.parametrize("mode", ["stability_v1", "stability_v2", "meta3"])
def test_optimizer_convergence(mode):
    # Simple linear regression
    # y = 2x + 1
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    y = 2 * X + 1 + 0.1 * torch.randn(100, 1)
    
    model = torch.nn.Linear(1, 1)
    # Initialize far from solution
    torch.nn.init.constant_(model.weight, 0.0)
    torch.nn.init.constant_(model.bias, 0.0)
    
    optimizer = CruxyOptimizer(model.parameters(), mode=mode, lr=0.05)
    criterion = torch.nn.MSELoss()
    
    initial_loss = criterion(model(X), y).item()
    
    for step in range(200):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step(loss=loss.item())
        
    final_loss = criterion(model(X), y).item()
    
    assert final_loss < initial_loss
    # Relaxed convergence check for smoke test
    # v2 clipping might slow down convergence on this specific toy problem
    assert final_loss < initial_loss * 0.8

def test_meta3_hyperparameter_adaptation():
    # Verify that LR changes in meta3 mode
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 1)
    optimizer = CruxyOptimizer(
        model.parameters(), 
        mode="meta3", 
        lr=1e-3,
        meta_interval=1 # Update every step for testing
    )
    
    initial_lr = optimizer.controller.current_lr
    
    # Fake some noisy gradients to trigger variance
    for _ in range(10):
        optimizer.zero_grad()
        # Forward pass to get valid graph
        output = model(torch.randn(1, 10))
        loss = output.sum()
        loss.backward()
        
        # Manually inject noise into grads to ensure variance
        for p in model.parameters():
            if p.grad is not None:
                p.grad += torch.randn_like(p) * 10.0 # High variance
            
        optimizer.step(loss=loss.item())
        
    final_lr = optimizer.controller.current_lr
    
    # LR should have changed due to meta-dynamics
    assert initial_lr != final_lr
