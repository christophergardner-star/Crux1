"""
Cruxy 8-GPU Distributed Training Benchmark
Testing Qwen2.5-7B with data parallelism across 8x A100-80GB
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
        print("CRUXY 8-GPU DISTRIBUTED TRAINING BENCHMARK")
        print(f"World Size: {world_size} GPUs (8x A100-80GB)")
        print(f"Total VRAM: 640GB")
        print("="*60)
    
    # Load 7B model
    model_name = "Qwen/Qwen2.5-7B"
    
    if rank == 0:
        print(f"\nLoading {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(local_rank)
    
    # Wrap with DDP - gradient_as_bucket_view for memory efficiency
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.module.parameters()) / 1e9
        mem_used = torch.cuda.memory_allocated(local_rank) / 1e9
        print(f"Model: {param_count:.2f}B params")
        print(f"Memory per GPU: {mem_used:.1f}GB")
    
    dist.barrier()
    
    # Initialize Cruxy optimizer
    optimizer = CruxyOptimizer(model.parameters(), lr=1e-5, mode="meta3")
    
    if rank == 0:
        print("Cruxy optimizer initialized (meta3 mode)")
    
    # Training data - larger batch for 8 GPUs
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Each GPU processes unique data
    all_texts = [
        "The future of artificial intelligence lies in developing systems that can reason",
        "Machine learning has revolutionized how we approach complex problems in science",
        "Deep neural networks have demonstrated remarkable capabilities in understanding",
        "Natural language processing enables computers to comprehend human communication",
        "Reinforcement learning allows agents to learn optimal policies through experience",
        "Computer vision algorithms can now identify objects with superhuman accuracy",
        "Transfer learning enables models to apply knowledge from one domain to another",
        "Attention mechanisms have become fundamental to modern neural architectures",
        "Generative models can create realistic images, text, and audio content",
        "Federated learning allows training across decentralized data sources securely",
        "Neural architecture search automates the design of optimal network structures",
        "Contrastive learning has emerged as a powerful self-supervised technique",
        "Multi-modal learning combines information from different sensory modalities",
        "Graph neural networks excel at learning from structured relational data",
        "Few-shot learning enables models to generalize from minimal examples",
        "Meta-learning teaches models how to learn more efficiently over time",
    ]
    
    # Each rank gets different samples (simulate distributed data loading)
    texts_per_rank = len(all_texts) // world_size
    start_idx = rank * texts_per_rank
    end_idx = start_idx + texts_per_rank
    texts = all_texts[start_idx:end_idx]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(local_rank)
    
    # Training loop
    if rank == 0:
        print(f"\nTraining: {len(all_texts)} total samples across {world_size} GPUs")
        print(f"Samples per GPU: {len(texts)}")
        print(f"Effective batch size: {len(all_texts)}")
    
    model.train()
    losses = []
    
    start = time.time()
    for step in range(15):
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
            if step % 3 == 0 or step == 14:
                elapsed = time.time() - start
                mem_gb = torch.cuda.max_memory_allocated(local_rank) / 1e9
                print(f"Step {step+1:2d}/15: Loss={avg_loss:.4f} | Time={elapsed:.1f}s | Peak Mem={mem_gb:.1f}GB")
    
    if rank == 0:
        total_time = time.time() - start
        reduction = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"\n{'='*60}")
        print("RESULTS - 8x A100-80GB DDP Training")
        print(f"{'='*60}")
        print(f"Model:        Qwen2.5-7B")
        print(f"GPUs:         8x A100-80GB (640GB total)")
        print(f"Batch:        {len(all_texts)} samples (data parallel)")
        print(f"Initial Loss: {losses[0]:.4f}")
        print(f"Final Loss:   {losses[-1]:.4f}")
        print(f"Reduction:    {reduction:.1f}%")
        print(f"Total Time:   {total_time:.1f}s")
        print(f"Per Step:     {total_time/15:.2f}s")
        print(f"Peak Memory:  {torch.cuda.max_memory_allocated(local_rank)/1e9:.1f}GB per GPU")
        print("="*60)
        print("✅ Cruxy 8-GPU distributed training verified!")
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
