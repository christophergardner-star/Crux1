import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cruxy import CruxyOptimizer
import time

# Simple ConvNet (Same as before)
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():
    # Configuration
    BATCH_SIZE = 64
    ACCUMULATION_STEPS = 4  # Virtual Batch Size = 64 * 4 = 256
    EPOCHS = 1
    LR = 0.001
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Device: {device}")
    
    # Data
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SimpleConvNet().to(device)
    
    optimizer = CruxyOptimizer(model.parameters(), lr=LR, mode="meta3")
    
    print(f"Training with Gradient Accumulation (Steps={ACCUMULATION_STEPS})")
    print(f"Physical Batch: {BATCH_SIZE} | Virtual Batch: {BATCH_SIZE * ACCUMULATION_STEPS}")
    
    model.train()
    optimizer.zero_grad()
    
    start_time = time.time()
    
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 1. Forward
        output = model(data)
        loss = F.nll_loss(output, target)
        
        # 2. Scale Loss
        # We scale loss so gradients are averaged correctly
        loss = loss / ACCUMULATION_STEPS
        
        # 3. Backward (Accumulate Gradients)
        loss.backward()
        
        # 4. Step (Only every N steps)
        if (i + 1) % ACCUMULATION_STEPS == 0:
            # Cruxy needs the *actual* loss for curvature estimation, 
            # so we pass loss.item() * ACCUMULATION_STEPS to restore the scale
            optimizer.step(loss=loss.item() * ACCUMULATION_STEPS)
            optimizer.zero_grad()
            
            if (i + 1) % 100 == 0:
                print(f"Step {i+1} | Loss: {loss.item() * ACCUMULATION_STEPS:.4f}")
                
        if i > 200: break # Short run for demo
        
    print(f"Finished in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
