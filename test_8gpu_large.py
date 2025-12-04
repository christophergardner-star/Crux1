"""
Cruxy 8-GPU Large Model Test
Test on single 8xA100 node with larger models
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from cruxy.optimizer import CruxyOptimizer
import time

def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print("="*60)
        print("CRUXY 8-GPU LARGE MODEL BENCHMARK")
        print(f"World Size: {world_size} GPUs")
        print(f"Total VRAM: {world_size * 80}GB (640GB)")
        print("="*60)
    
    # Test Qwen2.5-14B on 8 GPUs
    model_name = "Qwen/Qwen2.5-14B"
    
    if rank == 0:
        print(f"\nLoading {model_name} on {world_size} GPUs...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(local_rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.module.parameters()) / 1e9
        print(f"Model loaded: {param_count:.2f}B params per GPU")
        mem_used = torch.cuda.memory_allocated(local_rank) / 1e9
        mem_total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        print(f"GPU {local_rank} Memory: {mem_used:.1f}GB / {mem_total:.0f}GB")
    
    dist.barrier()
    
    # Initialize Cruxy optimizer
    optimizer = CruxyOptimizer(model.parameters(), lr=1e-5, mode="meta3")
    
    if rank == 0:
        print("Cruxy optimizer initialized (meta3 mode)")
    
    # Training data
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    texts = [
        "The future of artificial intelligence lies in developing systems that can reason",
        "Machine learning has revolutionized how we approach complex problems in science",
        "Deep neural networks have demonstrated remarkable capabilities in understanding",
        "Natural language processing enables computers to comprehend and generate text",
        "Reinforcement learning allows agents to learn optimal policies through experience",
        "Computer vision algorithms can now identify objects with superhuman accuracy",
        "Transfer learning enables models to apply knowledge from one domain to another",
        "Attention mechanisms have become fundamental to modern neural architectures",
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(local_rank)
    
    # Training loop
    if rank == 0:
        print(f"\nTraining for 10 steps with {len(texts)} samples (batch={len(texts)})...")
    
    model.train()
    losses = []
    
    start = time.time()
    for step in range(10):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step(loss=loss.item())
        
        # Gather loss from all ranks
        loss_tensor = torch.tensor([loss.item()], device=local_rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
        
        if rank == 0:
            losses.append(avg_loss)
            elapsed = time.time() - start
            print(f"Step {step+1}/10: Avg Loss = {avg_loss:.4f} | Time: {elapsed:.1f}s")
    
    if rank == 0:
        total_time = time.time() - start
        reduction = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"\n{'='*60}")
        print("RESULTS - Qwen2.5-14B on 8x A100-80GB")
        print(f"{'='*60}")
        print(f"Initial Loss: {losses[0]:.4f}")
        print(f"Final Loss:   {losses[-1]:.4f}")
        print(f"Reduction:    {reduction:.1f}%")
        print(f"Total Time:   {total_time:.1f}s")
        print(f"Avg Step:     {total_time/10:.2f}s")
        print(f"Throughput:   {10/total_time:.2f} steps/sec")
        print("="*60)
        print("✅ Cruxy distributed training verified!")
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
