import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cruxy.optimizer import CruxyOptimizer
import time

print("="*60)
print("CRUXY 14B TEST - QWEN2.5-14B")
print("="*60)

print(f"\nGPU: {torch.cuda.get_device_name()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB")

print("\nLoading Qwen2.5-14B (this will take a minute)...")
start = time.time()

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

load_time = time.time() - start
param_count = sum(p.numel() for p in model.parameters()) / 1e9
print(f"Loaded: {param_count:.2f}B params in {load_time:.1f}s")
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# Initialize Cruxy
optimizer = CruxyOptimizer(model.parameters(), lr=1e-5, mode="meta3")
print("\nCruxy optimizer initialized (meta3 mode)")

# Training data
texts = [
    "The future of artificial intelligence lies in",
    "Machine learning has revolutionized the way we",
    "Deep neural networks are capable of",
    "Natural language processing enables computers to",
]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")

# Training loop
print("\nTraining for 50 steps...")
model.train()
losses = []

start = time.time()
for step in range(50):
    optimizer.zero_grad()
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step(loss=loss.item())
    losses.append(loss.item())
    
    if step % 10 == 0 or step == 49:
        elapsed = time.time() - start
        print(f"Step {step+1}/50: Loss = {loss.item():.4f} | GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB | Time: {elapsed:.1f}s")

total_time = time.time() - start
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Model: Qwen2.5-14B ({param_count:.2f}B params)")
print(f"Initial Loss: {losses[0]:.4f}")
print(f"Final Loss: {losses[-1]:.4f}")
print(f"Reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
print(f"Total Time: {total_time:.1f}s ({50/total_time:.2f} steps/s)")
print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
print(f"\n✅ SUCCESS: Cruxy + 14B model working on H200!")
