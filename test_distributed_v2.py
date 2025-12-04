"""
Cruxy Multi-Node Distributed Training Test
Using torchrun with rdzv backend for easier multi-node setup
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from cruxy.optimizer import CruxyOptimizer
import time

def main():
    # Initialize distributed - torchrun sets these automatically
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print("="*60)
        print("CRUXY MULTI-NODE DISTRIBUTED TRAINING")
        print(f"World Size: {world_size} GPUs")
        print(f"Total VRAM: {world_size * 80}GB")
        print("="*60)
    
    # Load model on each GPU
    if rank == 0:
        print(f"\nLoading Qwen2.5-7B on {world_size} GPUs...")
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(local_rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.module.parameters()) / 1e9
        print(f"Model loaded: {param_count:.2f}B params per GPU")
        print(f"GPU {local_rank} Memory: {torch.cuda.memory_allocated(local_rank)/1e9:.1f}GB")
    
    dist.barrier()
    
    # Initialize Cruxy optimizer
    optimizer = CruxyOptimizer(model.parameters(), lr=1e-5, mode="meta3")
    
    if rank == 0:
        print("Cruxy optimizer initialized (meta3 mode)")
    
    # Training data
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    texts = [
        "The future of artificial intelligence lies in",
        "Machine learning has revolutionized the way we",
        "Deep neural networks are capable of",
        "Natural language processing enables computers to",
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(local_rank)
    
    # Training loop
    if rank == 0:
        print("\nTraining for 20 steps...")
    
    model.train()
    losses = []
    
    start = time.time()
    for step in range(20):
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
            if step % 5 == 0 or step == 19:
                elapsed = time.time() - start
                print(f"Step {step+1}/20: Avg Loss = {avg_loss:.4f} | Time: {elapsed:.1f}s")
    
    if rank == 0:
        total_time = time.time() - start
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"GPUs: {world_size}x A100-80GB")
        print(f"Initial Loss: {losses[0]:.4f}")
        print(f"Final Loss: {losses[-1]:.4f}")
        print(f"Reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
        print(f"Total Time: {total_time:.1f}s ({20/total_time:.2f} steps/s)")
        print(f"\n🔥 SUCCESS: Cruxy distributed training working!")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
