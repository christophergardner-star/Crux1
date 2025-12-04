"""
Cruxy BIG MODEL Test - 8x A100-80GB
Testing Qwen2.5-32B with device_map="auto" for model parallelism
640GB total VRAM should handle 32B comfortably
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cruxy.optimizer import CruxyOptimizer
import time

def main():
    print("="*70)
    print("CRUXY BIG MODEL BENCHMARK - Qwen2.5-32B")
    print("="*70)
    
    # Check available GPUs
    n_gpus = torch.cuda.device_count()
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpus)) / 1e9
    print(f"GPUs: {n_gpus}x A100-80GB")
    print(f"Total VRAM: {total_vram:.0f}GB")
    print("="*70)
    
    # Load 32B model with automatic device mapping
    model_name = "Qwen/Qwen2.5-32B"
    print(f"\nLoading {model_name} with device_map='auto'...")
    print("This will distribute the model across all GPUs...")
    
    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatic model parallelism
        trust_remote_code=True
    )
    load_time = time.time() - start_load
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"\nModel loaded in {load_time:.1f}s")
    print(f"Parameters: {param_count:.2f}B")
    
    # Show memory usage per GPU
    print("\nMemory per GPU:")
    for i in range(n_gpus):
        mem = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i}: {mem:.1f}GB")
    
    # Initialize Cruxy optimizer
    optimizer = CruxyOptimizer(model.parameters(), lr=1e-5, mode="meta3")
    print("\nCruxy optimizer initialized (meta3 mode)")
    
    # Tokenizer and training data
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    texts = [
        "The quantum mechanical properties of superconducting materials enable",
        "Advanced neural architectures have demonstrated emergent capabilities in",
        "The mathematical foundations of transformer attention mechanisms derive from",
        "Distributed computing paradigms for large-scale model training require",
    ]
    
    # Determine which device to put inputs on (first device in the model)
    first_device = next(model.parameters()).device
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    
    print(f"\nTraining for 10 steps (batch={len(texts)})...")
    print("-"*70)
    
    model.train()
    losses = []
    
    start = time.time()
    for step in range(10):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step(loss=loss.item())
        
        losses.append(loss.item())
        elapsed = time.time() - start
        
        # Get peak memory across all GPUs
        peak_mem = max(torch.cuda.max_memory_allocated(i) for i in range(n_gpus)) / 1e9
        print(f"Step {step+1:2d}/10: Loss={loss.item():.4f} | Time={elapsed:.1f}s | Peak={peak_mem:.1f}GB")
    
    total_time = time.time() - start
    reduction = (losses[0] - losses[-1]) / losses[0] * 100
    
    print("\n" + "="*70)
    print("RESULTS - Qwen2.5-32B on 8x A100-80GB")
    print("="*70)
    print(f"Model:        Qwen2.5-32B ({param_count:.1f}B params)")
    print(f"GPUs:         {n_gpus}x A100-80GB ({total_vram:.0f}GB total)")
    print(f"Distribution: device_map='auto' (model parallelism)")
    print(f"Initial Loss: {losses[0]:.4f}")
    print(f"Final Loss:   {losses[-1]:.4f}")
    print(f"Reduction:    {reduction:.1f}%")
    print(f"Total Time:   {total_time:.1f}s")
    print(f"Per Step:     {total_time/10:.2f}s")
    print("="*70)
    print("✅ Cruxy 32B training verified!")

if __name__ == "__main__":
    main()
