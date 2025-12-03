import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time
import os
import requests
import matplotlib.pyplot as plt
from cruxy import CruxyOptimizer

# --- 1. NanoGPT Model (Compact Definition) ---
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj  = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.size()
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.blocks(token_emb + pos_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

# --- 2. Data & Training Utils ---
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self): return len(self.data) - self.block_size
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]

def get_data():
    if not os.path.exists('input.txt'):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open('input.txt', 'w', encoding='utf-8') as f: f.write(requests.get(data_url).text)
    with open('input.txt', 'r', encoding='utf-8') as f: text = f.read()
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, len(chars)

def train_run(name, optimizer_cls, model_args, train_loader, device, steps=200, **opt_kwargs):
    print(f"--- Starting Run: {name} ---")
    torch.manual_seed(1337) # Ensure same initialization
    model = GPT(**model_args).to(device)
    
    if name == "AdamW":
        optimizer = optimizer_cls(model.parameters(), **opt_kwargs)
    else:
        # Cruxy
        optimizer = optimizer_cls(model.parameters(), **opt_kwargs)

    losses = []
    model.train()
    iter_num = 0
    
    # Create an infinite iterator
    data_iter = iter(train_loader)
    
    t0 = time.time()
    while iter_num < steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        
        if name == "Cruxy (Meta3)":
            optimizer.step(loss=loss.item())
        else:
            optimizer.step()
            
        losses.append(loss.item())
        iter_num += 1
        
        if iter_num % 100 == 0:
            print(f"{name} Step {iter_num}/{steps} | Loss: {loss.item():.4f}")

    dt = time.time() - t0
    print(f"{name} Finished in {dt:.2f}s\n")
    return losses

# --- 3. Main Execution ---
def main():
    # Config
    block_size = 64
    batch_size = 16
    n_layer = 4
    n_head = 4
    n_embd = 128
    steps = 1000  # Short race for quick validation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data, vocab_size = get_data()
    train_data = data[:int(0.9*len(data))]
    dataset = CharDataset(train_data, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model_args = dict(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size, dropout=0.0)
    
    # 1. Run AdamW (Baseline)
    # Standard settings: lr=1e-3 is common for this size
    losses_adam = train_run("AdamW", torch.optim.AdamW, model_args, loader, device, steps=steps, lr=1e-3, weight_decay=0.1)
    
    # 2. Run Cruxy (Challenger)
    losses_cruxy = train_run("Cruxy (Meta3)", CruxyOptimizer, model_args, loader, device, steps=steps, 
                             lr=1e-3, mode="meta3", weight_decay=0.1, decoupled_weight_decay=True, use_nesterov=True)
    
    # 3. Run Cruxy Meta-Lion (The Secret Weapon)
    # Lion typically needs 3x-10x lower LR than Adam. 
    # But Meta3 should handle it. We'll start at 1e-4 to be safe/fair for Lion mechanics.
    losses_lion = train_run("Cruxy (Meta-Lion)", CruxyOptimizer, model_args, loader, device, steps=steps,
                            lr=1e-4, mode="meta3", use_lion=True, weight_decay=0.1)

    # 4. Plot
    print("Generating Hero Chart...")
    plt.figure(figsize=(10, 6))
    
    # Smooth curves slightly for readability
    def smooth(scalars, weight=0.8):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    plt.plot(smooth(losses_adam), label='AdamW (Baseline)', color='gray', alpha=0.5, linestyle='--')
    plt.plot(smooth(losses_cruxy), label='Cruxy (Meta3)', color='blue', linewidth=2)
    plt.plot(smooth(losses_lion), label='Cruxy (Meta-Lion)', color='red', linewidth=2)
    
    plt.title('Training Stability: AdamW vs Cruxy vs Meta-Lion')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = 'hero_chart.png'
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    print("Open this image. If the Blue line is lower/smoother than the Gray line, you have a winner.")

if __name__ == "__main__":
    main()
