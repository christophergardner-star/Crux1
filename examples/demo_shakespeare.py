import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time
import os
import requests
from cruxy import CruxyOptimizer

# --- 1. NanoGPT Model (Same as Scaling Experiment) ---
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 2. Data Handling ---
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def prepare_data():
    file_path = 'input.txt'
    if not os.path.exists(file_path):
        print("Downloading Tiny Shakespeare...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Data loaded. Length: {len(text)} chars. Vocab size: {vocab_size}")
    
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    return data, vocab_size, decode

# --- 3. Main Training Script ---
def main():
    # Config for Laptop (CPU friendly but capable)
    config = {
        'block_size': 128, # Context length
        'batch_size': 16,  # Small batch for CPU
        'n_layer': 4,      # Shallow network
        'n_head': 4,
        'n_embd': 128,     # Small embedding
        'dropout': 0.0,
        'lr': 1e-3,
        'max_iters': 200,  # Short run to prove learning
        'eval_interval': 50,
        'device': 'cpu'
    }
    
    if torch.cuda.is_available():
        config['device'] = 'cuda'
        print("CUDA available! Switching to GPU.")
    
    print(f"Running on {config['device']}...")

    # Data
    data, vocab_size, decode = prepare_data()
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    train_dataset = CharDataset(train_data, config['block_size'])
    # Random sampler for infinite stream effect
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Model
    model = GPT(vocab_size, config['n_embd'], config['n_head'], config['n_layer'], config['block_size'], config['dropout'])
    model.to(config['device'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Optimizer: Cruxy Meta3
    optimizer = CruxyOptimizer(
        model.parameters(), 
        lr=config['lr'], 
        mode="meta3",
        weight_decay=0.1,
        decoupled_weight_decay=True,
        use_nesterov=True
    )

    # Training Loop
    model.train()
    iter_num = 0
    t0 = time.time()
    
    print("\n--- Starting Training ---")
    for x, y in train_loader:
        x, y = x.to(config['device']), y.to(config['device'])
        
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step(loss=loss.item())
        
        iter_num += 1
        
        if iter_num % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            lr = optimizer.param_groups[0]['lr']
            if hasattr(optimizer, 'controller'):
                lr = optimizer.controller.current_lr
            print(f"Iter {iter_num}: loss {loss.item():.4f}, lr {lr:.6f}, time {dt*100:.0f}ms/step")
            
        if iter_num >= config['max_iters']:
            break
            
    print("\n--- Training Complete. Generating Text... ---")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=config['device'])
    generated = model.generate(context, max_new_tokens=200)
    print(decode(generated[0].tolist()))

if __name__ == "__main__":
    main()
