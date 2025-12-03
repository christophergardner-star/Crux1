import torch
from cruxy import CruxyOptimizer
import os

def test_checkpointing():
    """Test that optimizer state (including controller) is saved and loaded correctly."""
    params = torch.tensor([1.0, 2.0], requires_grad=True)
    opt = CruxyOptimizer([params], lr=0.1, mode="meta3")
    
    # 1. Run a few steps to change state
    for i in range(5):
        opt.zero_grad()
        loss = (params**2).sum()
        loss.backward()
        opt.step(loss=loss.item())
        
    # Capture state
    state_before = opt.state_dict()
    lr_before = opt.controller.current_lr
    var_fast_before = opt.controller.inner.variance_tracker.var_fast
    
    # 2. Create new optimizer and load state
    params2 = torch.tensor([1.0, 2.0], requires_grad=True)
    opt2 = CruxyOptimizer([params2], lr=0.1, mode="meta3")
    
    opt2.load_state_dict(state_before)
    
    # 3. Verify state restoration
    assert opt2.controller.current_lr == lr_before
    assert opt2.controller.inner.variance_tracker.var_fast == var_fast_before
    assert opt2.controller.step_count == opt.controller.step_count
    
    print("Checkpointing test passed!")

if __name__ == "__main__":
    test_checkpointing()
