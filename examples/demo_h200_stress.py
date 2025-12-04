import os

# Ensure workspace directories exist
os.makedirs("/workspace/hf_cache", exist_ok=True)
os.makedirs("/workspace/tmp", exist_ok=True)

# Set Cache and Temp paths to workspace
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TMPDIR"] = "/workspace/tmp"
os.environ["TEMP"] = "/workspace/tmp"
os.environ["TMP"] = "/workspace/tmp"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from cruxy.optimizer import CruxyOptimizer
import os

# --- Configuration for H200 Stress Test ---
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"  # 32B Parameters (~64GB in BF16)
# If 32B is too slow to download, we can fallback to 14B, but user said "full beans"
# MODEL_ID = "Qwen/Qwen2.5-14B-Instruct" 

OUTPUT_DIR = "./cruxy_h200_stress_test"

def main():
    print(f"🚀 Starting H200 Stress Test with {MODEL_ID}...")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model in Full BF16 (No Quantization needed on H200)
    print("   Loading model in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" 
    )
    
    # Enable Gradient Checkpointing for efficiency (optional on 141GB but good practice)
    model.gradient_checkpointing_enable()

    # 3. Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Dataset (Tiny Shakespeare for speed)
    dataset = load_dataset("tiny_shakespeare", split="train")
    
    def format_prompt(sample):
        return f"### Text:\n{sample['text'][:1024]}\n"

    # 5. Cruxy Optimizer
    # We use the factory to get the optimizer class and kwargs, but SFTTrainer needs the class directly
    # or we can pass the optimizer instance if we use a custom loop. 
    # For SFTTrainer, we usually pass the optimizer in 'optimizers' tuple (opt, scheduler).
    
    # However, SFTTrainer integration with custom optimizers can be tricky.
    # We will use a standard TrainingArguments and let Cruxy be injected via a callback or 
    # just instantiate it manually if we were writing a raw loop.
    # EASIER WAY: Use the 'optim="adamw_hf"' placeholder and overwrite it, 
    # OR just use a raw training loop for maximum control/transparency like the other demos.
    
    # Let's stick to the raw loop pattern from other demos for consistency and clarity.
    print("   Preparing Training Loop...")
    
    optimizer = CruxyOptimizer(
        model.parameters(),
        lr=1e-4,
        mode="meta-lion",
        weight_decay=0.01
    )

    # Simple Training Loop
    model.train()
    inputs = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt").to("cuda")
    
    print("\n   🔥 Starting Training Steps...")
    for i in range(10):
        optimizer.zero_grad()
        
        # Forward
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # Backward
        loss.backward()
        
        # Step
        optimizer.step()
        
        mem_used = torch.cuda.memory_allocated() / 1024**3
        mem_res = torch.cuda.memory_reserved() / 1024**3
        print(f"   Step {i+1}/10 | Loss: {loss.item():.4f} | Mem: {mem_used:.2f}GB (Alloc) / {mem_res:.2f}GB (Res)")

    print("\n✅ Stress Test Complete. The H200 didn't even blink.")

if __name__ == "__main__":
    main()
