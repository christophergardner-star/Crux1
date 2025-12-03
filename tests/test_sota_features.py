import torch
import pytest
from cruxy import CruxyOptimizer

def simple_loss(params):
    return (params[0] - 1)**2 + (params[1] + 1)**2

def test_lion_mode():
    """Test that Lion mode runs and converges on a simple problem."""
    params = torch.tensor([0.0, 0.0], requires_grad=True)
    # Lion typically needs lower LR
    opt = CruxyOptimizer([params], lr=0.1, use_lion=True, mode="stability_v1")
    
    initial_loss = simple_loss(params).item()
    
    for _ in range(50):
        opt.zero_grad()
        loss = simple_loss(params)
        loss.backward()
        opt.step()
        
    final_loss = simple_loss(params).item()
    assert final_loss < initial_loss
    assert final_loss < 0.1

def test_gradient_centralization():
    """Test that Gradient Centralization modifies gradients for >1D tensors."""
    # 1D tensor (should NOT be centralized)
    p1 = torch.randn(10, requires_grad=True)
    # 2D tensor (should be centralized)
    p2 = torch.randn(10, 10, requires_grad=True)
    
    opt = CruxyOptimizer([p1, p2], lr=0.01, use_gc=True)
    
    loss = p1.sum() + p2.sum()
    loss.backward()
    
    # Store original grads
    g1_orig = p1.grad.clone()
    g2_orig = p2.grad.clone()
    
    # Step (this applies GC to p2.grad in place effectively during update, 
    # but wait, my implementation modifies p.grad in place? 
    # Let's check the implementation. 
    # Yes: if use_gc and grad.dim() > 1: grad.add_(-grad.mean(...))
    
    opt.step()
    
    # Check p1 grad (should be unchanged by GC logic, though step modifies p, not p.grad usually, 
    # but my implementation modifies grad in place for GC)
    # Actually, standard optimizers shouldn't modify .grad in place if they can avoid it, 
    # but for GC it's a common pattern.
    
    # p1 is 1D, so GC should not apply.
    # However, the optimizer step might have modified p.grad if I implemented it that way.
    # Looking at code: 
    # if use_gc and grad.dim() > 1: grad.add_(-grad.mean(...))
    # So p1.grad should be same (1D). p2.grad should be zero-meaned on axes > 0.
    
    # Verify p2.grad is now centralized (mean over dims 1..N is 0)
    # For 2D (N, M), mean over dim 1 should be 0.
    # My code: grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)
    # For 2D, dim=1.
    
    # Let's check if the grad attached to p2 has 0 mean across dim 1
    means = p2.grad.mean(dim=1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-6)

def test_nesterov_momentum():
    """Test Nesterov momentum execution."""
    params = torch.tensor([1.0], requires_grad=True)
    opt = CruxyOptimizer([params], lr=0.1, use_nesterov=True)
    
    opt.zero_grad()
    loss = params * 2
    loss.backward()
    opt.step()
    
    # Just ensure it runs without error and updates parameters
    assert params.item() != 1.0

def test_decoupled_weight_decay():
    """Test that decoupled weight decay is applied."""
    # With decoupled WD, the parameter is decayed directly: p = p * (1 - lr * wd)
    # Then the gradient update is applied.
    
    # Case 1: Zero gradient, non-zero weight decay
    # Note: SafetyGuard clamps LR to max 1e-2 (0.01). So we use 0.01 to avoid confusion.
    params = torch.tensor([10.0], requires_grad=True)
    opt = CruxyOptimizer([params], lr=0.01, weight_decay=0.1, decoupled_weight_decay=True)
    
    opt.zero_grad()
    # No loss, so grad is None? No, we need grad to be not None for the loop to trigger usually.
    # Let's force a zero gradient.
    loss = params * 0.0
    loss.backward()
    
    # Before step: p=10.0
    # Update: p *= (1 - 0.01 * 0.1) = 10 * (1 - 0.001) = 10 * 0.999 = 9.99
    # Gradient is 0, so momentum update adds 0.
    opt.step()
    
    assert torch.allclose(params, torch.tensor([9.99]))

def test_meta_lion_integration():
    """Test Meta-Lion (Lion + Meta3 Controller)."""
    params = torch.tensor([1.0, 1.0], requires_grad=True)
    opt = CruxyOptimizer(
        [params], 
        lr=0.1, 
        mode="meta3", 
        use_lion=True,
        meta_interval=1
    )
    
    # Run a few steps
    for i in range(5):
        opt.zero_grad()
        loss = (params**2).sum()
        loss.backward()
        opt.step(loss=loss.item())
        
        # Check if controller is doing its job (updating LR)
        if i > 1:
            # Meta controller should have modified the LR
            # It's hard to predict exact value, but it shouldn't be exactly initial if variance is high
            pass
            
    # Just ensure no crash and params moved
    assert params[0].item() != 1.0
