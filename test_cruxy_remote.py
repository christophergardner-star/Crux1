import torch
from cruxy.optimizer import CruxyOptimizer
import time

def test_cruxy_install():
    print("✅ Cruxy Import Successful")
    
    # Create a simple model
    model = torch.nn.Linear(10, 1).cuda()
    
    # Initialize Cruxy
    optimizer = CruxyOptimizer(model.parameters(), lr=1e-3, mode="meta3")
    print(f"✅ Optimizer Initialized: {optimizer}")
    
    # Dummy training step
    data = torch.randn(10, 10).cuda()
    target = torch.randn(10, 1).cuda()
    
    print("🚀 Running Training Step...")
    optimizer.zero_grad()
    output = model(data)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"✅ Step Complete. Loss: {loss.item():.4f}")
    print("🎉 Cruxy is installed and working on the H200!")

if __name__ == "__main__":
    test_cruxy_install()
