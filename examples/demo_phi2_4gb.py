import torch
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from cruxy import CruxyOptimizer

def main():
    print("--- Axiom Forge Systems: Phi-2 (2.7B) 4GB VRAM Demo ---")
    print("Note: Phi-2 is 2.7B params. This requires aggressive offloading or quantization on 4GB cards.")
    
    model_id = "microsoft/phi-2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check for bitsandbytes for 4-bit loading (Best for 4GB)
    try:
        import bitsandbytes
        quantization_config = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
        print("bitsandbytes detected: Using 4-bit quantization (Fastest)")
    except ImportError:
        quantization_config = {}
        print("bitsandbytes NOT detected: Using Float16 with CPU Offloading (Slower but works)")

    # 1. Load Model & Tokenizer
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with offloading support
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto", # Critical for 4GB VRAM
        trust_remote_code=True,
        **quantization_config
    )
    
    # Enable Gradient Checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 2. Apply LoRA
    print("Applying LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=16,             # Slightly larger rank for Phi-2
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["Wqkv", "out_proj"] # Phi-2 specific modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Optimizer: Cruxy Meta-Lion
    print("Initializing Cruxy Meta-Lion...")
    optimizer = CruxyOptimizer(
        model.parameters(),
        lr=1e-4,
        mode="meta3",
        use_lion=True,
        weight_decay=0.01
    )

    # 4. Training Loop
    text = "The logic of stability is the foundation of intelligence."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    print("Starting training loop (20 steps)...")
    model.train()
    
    losses = []
    start_time = time.time()
    
    # We run fewer steps because Phi-2 is heavier
    for i in range(20):
        optimizer.zero_grad()
        
        # Forward
        # Note: inputs might need to be moved if model is offloaded, but HF handles it usually
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # Backward
        loss.backward()
        
        # Step
        optimizer.step(loss=loss.item())
        
        losses.append(loss.item())
        
        if i % 2 == 0:
            print(f"Step {i} | Loss: {loss.item():.4f}")

    print(f"Finished in {time.time() - start_time:.2f}s")
    print("Success! Phi-2 trained on your hardware.")
    
    # 5. Generate Chart
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Cruxy Meta-Lion (Phi-2)', color='purple', linewidth=2)
        plt.title('Phi-2 (2.7B) Training on Consumer Hardware')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('phi2_chart.png')
        print("Chart saved to phi2_chart.png")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
