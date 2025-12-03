import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time
import json
import argparse
import os
from cruxy import CruxyOptimizer

# --- 1. Scalable NanoGPT Model ---
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
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

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
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.size()
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# --- 2. Synthetic Data ---
class SyntheticTextDataset(Dataset):
    def __init__(self, vocab_size, block_size, length=1000):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Random tokens
        data = torch.randint(0, self.vocab_size, (self.block_size + 1,))
        x = data[:-1]
        y = data[1:]
        return x, y

# --- 3. Experiment Runner ---
def run_experiment(config):
    print(f"--- Starting Experiment: {config['name']} ---")
    print(json.dumps(config, indent=2))
    
    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.get('force_cpu', False):
        device = "cpu"
    print(f"Device: {device}")

    # Model
    model = GPT(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size']
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {params:,}")
    
    # Optimizer
    optimizer = CruxyOptimizer(
        model.parameters(),
        lr=config['lr'],
        mode="meta3",
        weight_decay=0.1,
        decoupled_weight_decay=True,
        use_nesterov=True,
        use_gc=True
    )
    
    # Data
    dataset = SyntheticTextDataset(config['vocab_size'], config['block_size'], length=config['steps'] * config['batch_size'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Logging
    log_file = "scaling_log.jsonl"
    
    # Training Loop
    model.train()
    start_time = time.time()
    
    step = 0
    for x, y in loader:
        step += 1
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        optimizer.step(loss=loss.item())
        
        if step % 10 == 0:
            dt = time.time() - start_time
            tps = (step * config['batch_size'] * config['block_size']) / dt
            
            # Get LR
            lr = optimizer.param_groups[0]['lr']
            if hasattr(optimizer, 'controller'):
                lr = optimizer.controller.current_lr
            
            print(f"Step {step}/{config['steps']} | Loss: {loss.item():.4f} | LR: {lr:.6f} | Tokens/sec: {tps:.0f}")
            
            # Log to file
            log_entry = {
                "experiment": config['name'],
                "params": params,
                "step": step,
                "loss": loss.item(),
                "lr": lr,
                "tps": tps,
                "config": config
            }
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        if step >= config['steps']:
            break
            
    print(f"--- Experiment {config['name']} Complete ---\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default='1', help='Scaling step (1, 2, 3...)')
    args = parser.parse_args()
    
    # Define Scaling Steps
    steps = {
        '1': {
            'name': 'Step 1: Tiny (Validation)',
            'vocab_size': 1000,
            'block_size': 32,
            'n_embd': 64,
            'n_head': 4,
            'n_layer': 2,
            'batch_size': 4,
            'steps': 50,
            'lr': 1e-3
        },
        '2': {
            'name': 'Step 2: Small (Stress Test)',
            'vocab_size': 50257, # GPT-2 size
            'block_size': 64,
            'n_embd': 128,
            'n_head': 4,
            'n_layer': 4,
            'batch_size': 8,
            'steps': 50,
            'lr': 6e-4
        },
        '3': {
            'name': 'Step 3: Medium (CPU Limit)',
            'vocab_size': 50257,
            'block_size': 128,
            'n_embd': 256,
            'n_head': 8,
            'n_layer': 6,
            'batch_size': 8,
            'steps': 50,
            'lr': 3e-4
        },
        '4': {
            'name': 'Step 4: Large (Overload)',
            'vocab_size': 50257,
            'block_size': 256,
            'n_embd': 384,
            'n_head': 12,
            'n_layer': 12, # GPT-2 Small-ish
            'batch_size': 4,
            'steps': 20,
            'lr': 3e-4
        }
    }
    
    if args.step in steps:
        run_experiment(steps[args.step])
    else:
        print(f"Unknown step {args.step}. Available: {list(steps.keys())}")
