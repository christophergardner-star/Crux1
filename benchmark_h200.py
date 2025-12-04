"""
Cruxy H200 Benchmark Suite
- 7B model validation (Mistral-7B)
- 500 step training run with loss curve
- AdamW vs Cruxy comparison
- Gemma 2B / Phi-2 verification
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from cruxy.optimizer import CruxyOptimizer
import time
import json

def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1e9

def run_training(model, tokenizer, optimizer, optimizer_name, steps=100):
    """Run training and return loss history"""
    texts = [
        "The future of artificial intelligence lies in",
        "Machine learning has revolutionized the way we",
        "Deep neural networks are capable of",
        "Natural language processing enables computers to",
        "The transformer architecture fundamentally changed",
        "Gradient descent optimization finds the minimum by",
        "Backpropagation computes gradients through",
        "Attention mechanisms allow models to focus on",
    ]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
    
    model.train()
    losses = []
    start_time = time.time()
    
    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        
        if hasattr(optimizer, 'step') and 'loss' in optimizer.step.__code__.co_varnames:
            optimizer.step(loss=loss.item())
        else:
            optimizer.step()
        
        losses.append(loss.item())
        
        if step % 50 == 0 or step == steps - 1:
            elapsed = time.time() - start_time
            print(f"  [{optimizer_name}] Step {step+1}/{steps}: Loss = {loss.item():.4f} | GPU: {get_gpu_memory():.1f}GB | Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    return {
        "optimizer": optimizer_name,
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "reduction": (losses[0] - losses[-1]) / losses[0] * 100,
        "time_seconds": total_time,
        "steps_per_second": steps / total_time
    }

def test_model(model_name, friendly_name, steps=100, compare_adamw=True):
    """Test a model with Cruxy and optionally compare to AdamW"""
    print(f"\n{'='*60}")
    print(f"TESTING: {friendly_name}")
    print(f"Model: {model_name}")
    print(f"Steps: {steps}")
    print(f"{'='*60}")
    
    # Load model
    print(f"\nLoading {friendly_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Loaded: {param_count:.2f}B params | GPU: {get_gpu_memory():.1f}GB")
    
    results = {}
    
    # Test Cruxy
    print(f"\n--- Cruxy (meta3) ---")
    optimizer = CruxyOptimizer(model.parameters(), lr=1e-5, mode="meta3")
    results["cruxy"] = run_training(model, tokenizer, optimizer, "Cruxy", steps)
    
    # Compare with AdamW
    if compare_adamw:
        # Reset model
        del model, optimizer
        torch.cuda.empty_cache()
        
        print(f"\nReloading model for AdamW comparison...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).cuda()
        
        print(f"\n--- AdamW (baseline) ---")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        results["adamw"] = run_training(model, tokenizer, optimizer, "AdamW", steps)
        
        # Comparison
        print(f"\n--- COMPARISON ---")
        print(f"Cruxy: {results['cruxy']['initial_loss']:.4f} -> {results['cruxy']['final_loss']:.4f} ({results['cruxy']['reduction']:.1f}% reduction)")
        print(f"AdamW: {results['adamw']['initial_loss']:.4f} -> {results['adamw']['final_loss']:.4f} ({results['adamw']['reduction']:.1f}% reduction)")
        print(f"Cruxy speed: {results['cruxy']['steps_per_second']:.2f} steps/s")
        print(f"AdamW speed: {results['adamw']['steps_per_second']:.2f} steps/s")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results

def main():
    print("="*60)
    print("CRUXY H200 BENCHMARK SUITE")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB")
    print("="*60)
    
    all_results = {}
    
    # 1. Quick wins - Verify open models
    print("\n" + "="*60)
    print("PHASE 1: QUICK WINS (Qwen2.5-1.5B, Phi-2)")
    print("="*60)
    
    all_results["qwen_1.5b"] = test_model(
        "Qwen/Qwen2.5-1.5B",
        "Qwen2.5 1.5B",
        steps=50,
        compare_adamw=False
    )
    
    torch.cuda.empty_cache()
    
    all_results["phi2"] = test_model(
        "microsoft/phi-2",
        "Phi-2 (2.7B)",
        steps=50,
        compare_adamw=False
    )
    
    torch.cuda.empty_cache()
    
    # 2. The big one - 7B model (using Qwen2.5-7B - fully open)
    print("\n" + "="*60)
    print("PHASE 2: 7B VALIDATION (Qwen2.5-7B)")
    print("="*60)
    
    all_results["qwen_7b"] = test_model(
        "Qwen/Qwen2.5-7B",
        "Qwen2.5 7B",
        steps=100,
        compare_adamw=True
    )
    
    torch.cuda.empty_cache()
    
    # 3. Extended run - 500 steps on SmolLM2
    print("\n" + "="*60)
    print("PHASE 3: STABILITY TEST (500 steps)")
    print("="*60)
    
    all_results["stability_500"] = test_model(
        "HuggingFaceTB/SmolLM2-1.7B",
        "SmolLM2 1.7B (500 steps)",
        steps=500,
        compare_adamw=True
    )
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    with open("/workspace/benchmark_results.json", "w") as f:
        # Convert losses to summary stats for JSON
        summary = {}
        for name, data in all_results.items():
            summary[name] = {}
            for opt, results in data.items():
                summary[name][opt] = {
                    "initial_loss": results["initial_loss"],
                    "final_loss": results["final_loss"],
                    "reduction_pct": results["reduction"],
                    "time_seconds": results["time_seconds"],
                    "steps_per_second": results["steps_per_second"]
                }
        json.dump(summary, f, indent=2)
    
    print("Results saved to /workspace/benchmark_results.json")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for name, data in all_results.items():
        print(f"\n{name}:")
        for opt, results in data.items():
            print(f"  {opt}: {results['initial_loss']:.4f} -> {results['final_loss']:.4f} ({results['reduction']:.1f}%)")

if __name__ == "__main__":
    main()
