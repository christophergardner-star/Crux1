import torch
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from cruxy import CruxyOptimizer

def main():
    print("--- Axiom Forge Systems: Gemma 2B 4GB VRAM Demo ---")
    print("Note: Requires HuggingFace Login (huggingface-cli login) for gated access.")
    
    model_id = "google/gemma-2b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check for bitsandbytes
    try:
        import bitsandbytes
        quantization_config = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
        print("bitsandbytes detected: Using 4-bit quantization")
    except ImportError:
        quantization_config = {}
        print("bitsandbytes NOT detected: Using Float16 with CPU Offloading")

    # 1. Load Model & Tokenizer
    print(f"Loading {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto",
            **quantization_config
        )
    except OSError as e:
        print("\nERROR: Could not load Gemma 2B. Make sure you have accepted the license on HuggingFace and logged in.")
        print(f"Details: {e}")
        return

    # Enable Gradient Checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 2. Apply LoRA
    print("Applying LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8,
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"] # Minimal targets for memory efficiency
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
    text = "Stability is the key to unlocking AGI."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    print("Starting training loop (20 steps)...")
    model.train()
    
    losses = []
    start_time = time.time()
    
    for i in range(20):
        optimizer.zero_grad()
        
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step(loss=loss.item())
        
        losses.append(loss.item())
        
        if i % 2 == 0:
            print(f"Step {i} | Loss: {loss.item():.4f}")

    print(f"Finished in {time.time() - start_time:.2f}s")
    print("Success! Gemma 2B trained on your hardware.")
    
    # 5. Generate Chart
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Cruxy Meta-Lion (Gemma 2B)', color='green', linewidth=2)
        plt.title('Gemma 2B Training on Consumer Hardware')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('gemma_chart.png')
        print("Chart saved to gemma_chart.png")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
