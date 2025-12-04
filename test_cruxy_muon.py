import torch
import torch.nn as nn
from cruxy.optimizer import CruxyOptimizer

def test_muon_simple():
    print("Testing Muon Optimizer...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup simple problem
    torch.manual_seed(42)
    # Muon works on >= 2D params. Linear weight is 2D.
    model = nn.Linear(10, 10, bias=False).to(device)
    target = torch.randn(10, 10).to(device)
    inputs = torch.randn(10, 10).to(device)
    
    # Initialize Optimizer
    # Muon usually needs a higher LR than Adam? Or similar?
    # Paper suggests 0.02 for Muon? Let's try 0.01.
    optimizer = CruxyOptimizer(model.parameters(), lr=0.01, mode="muon")
    
    print(f"Optimizer initialized in mode: {optimizer.mode}")
    
    # Training loop
    initial_loss = None
    for step in range(10):
        optimizer.zero_grad()
        output = model(inputs)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        
        print(f"Step {step}: Loss = {loss.item():.6f}")
        
    if loss.item() < initial_loss:
        print("SUCCESS: Loss decreased with Muon.")
    else:
        print("FAILURE: Loss did not decrease.")

if __name__ == "__main__":
    test_muon_simple()
