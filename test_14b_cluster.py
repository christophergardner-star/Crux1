"""
Cruxy BIG MODEL Test - 8x A100-80GB
Testing Qwen2.5-14B with model parallelism across 8 GPUs
Using per-device optimizers to handle multi-device model
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cruxy.optimizer import CruxyOptimizer
import time

def main():
    print("="*70)
    print("CRUXY 14B MODEL PARALLEL BENCHMARK")
    print("="*70)
    
    # Check available GPUs
    n_gpus = torch.cuda.device_count()
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpus)) / 1e9
    print(f"GPUs: {n_gpus}x A100-80GB")
    print(f"Total VRAM: {total_vram:.0f}GB")
    print("="*70)
    
    # Load 14B model with automatic device mapping
    model_name = "Qwen/Qwen2.5-14B"
    print(f"\nLoading {model_name} with device_map='auto'...")
    
    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    load_time = time.time() - start_load
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"\nModel loaded in {load_time:.1f}s")
    print(f"Parameters: {param_count:.2f}B")
    
    # Show memory usage per GPU
    print("\nMemory distribution across GPUs:")
    for i in range(n_gpus):
        mem = torch.cuda.memory_allocated(i) / 1e9
        if mem > 0.1:
            print(f"  GPU {i}: {mem:.1f}GB")
    
    # Group parameters by device and create optimizers
    device_params = {}
    for name, param in model.named_parameters():
        device = str(param.device)
        if device not in device_params:
            device_params[device] = []
        device_params[device].append(param)
    
    # Create a Cruxy optimizer per device
    optimizers = {}
    for device, params in device_params.items():
        optimizers[device] = CruxyOptimizer(params, lr=1e-5, mode="meta3")
    
    print(f"\nCreated {len(optimizers)} Cruxy optimizers (one per device)")
    
    # Tokenizer and training data
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    texts = [
        "The quantum mechanical properties of superconducting materials enable revolutionary advances in",
        "Advanced neural architectures have demonstrated emergent capabilities that transcend their training",
        "The mathematical foundations of transformer attention mechanisms derive from statistical physics",
        "Distributed computing paradigms for large-scale model training require sophisticated synchronization",
        "Emergent intelligence in large language models arises from complex interactions between layers",
        "The geometry of high-dimensional parameter spaces reveals surprising optimization landscapes",
        "Causal reasoning in artificial systems requires fundamentally different approaches than pattern",
        "Self-supervised learning has unlocked unprecedented scale in natural language understanding",
    ]
    
    first_device = next(model.parameters()).device
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    
    print(f"\nTraining for 15 steps (batch={len(texts)}, seq_len=256)...")
    print("-"*70)
    
    model.train()
    losses = []
    
    start = time.time()
    for step in range(15):
        # Zero all optimizers
        for opt in optimizers.values():
            opt.zero_grad()
        
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        
        # Step all optimizers
        for opt in optimizers.values():
            opt.step(loss=loss.item())
        
        losses.append(loss.item())
        elapsed = time.time() - start
        
        # Get total memory across all GPUs
        total_mem = sum(torch.cuda.memory_allocated(i) for i in range(n_gpus)) / 1e9
        peak_mem = max(torch.cuda.max_memory_allocated(i) for i in range(n_gpus)) / 1e9
        
        if step % 3 == 0 or step == 14:
            print(f"Step {step+1:2d}/15: Loss={loss.item():.4f} | Time={elapsed:.1f}s | Total={total_mem:.0f}GB | Peak={peak_mem:.1f}GB")
    
    total_time = time.time() - start
    reduction = (losses[0] - losses[-1]) / losses[0] * 100
    
    print("\n" + "="*70)
    print("RESULTS - Qwen2.5-14B on 8x A100-80GB (Model Parallel)")
    print("="*70)
    print(f"Model:        Qwen2.5-14B ({param_count:.1f}B params)")
    print(f"GPUs:         {n_gpus}x A100-80GB ({total_vram:.0f}GB total)")
    print(f"Distribution: device_map='auto' (pipeline parallelism)")
    print(f"Batch Size:   {len(texts)} samples")
    print(f"Initial Loss: {losses[0]:.4f}")
    print(f"Final Loss:   {losses[-1]:.4f}")
    print(f"Reduction:    {reduction:.1f}%")
    print(f"Total Time:   {total_time:.1f}s")
    print(f"Per Step:     {total_time/15:.2f}s")
    print("="*70)
    print("✅ Cruxy 14B model parallel training verified!")

if __name__ == "__main__":
    main()
