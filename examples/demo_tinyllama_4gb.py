import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from cruxy import CruxyOptimizer
import time

def main():
    print("--- Axiom Forge Systems: TinyLlama 4GB VRAM Demo ---")
    
    # 1. Configuration for 4GB VRAM
    # We use Float16 to fit the model (1.1B params * 2 bytes = 2.2GB)
    # We use LoRA to only train adapters (saving optimizer memory)
    # We use Gradient Checkpointing to save activation memory
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: This will be very slow on CPU. GPU recommended.")

    # 2. Load Model & Tokenizer
    print(f"Loading {model_id} in float16...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto" # Will use GPU if available
    )
    
    # Enable Gradient Checkpointing (Critical for 4GB VRAM)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 3. Apply LoRA
    print("Applying LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Optimizer: Cruxy Meta-Lion
    # Even though we are only training adapters, Meta-Lion is best for stability
    print("Initializing Cruxy Meta-Lion...")
    optimizer = CruxyOptimizer(
        model.parameters(),
        lr=1e-4,           # Lion likes lower LR
        mode="meta3",      # Auto-tuning
        use_lion=True,     # Memory efficient backend
        weight_decay=0.01
    )

    # 5. Dummy Training Loop
    # We'll train on a single repeated sentence to prove convergence
    text = "Axiom Forge Systems is the future of AI stability."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    print("Starting training loop...")
    model.train()
    
    losses = []
    start_time = time.time()
    for i in range(50):
        optimizer.zero_grad()
        
        # Forward
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # Backward
        loss.backward()
        
        # Step (Cruxy needs loss for curvature)
        optimizer.step(loss=loss.item())
        
        losses.append(loss.item())
        
        if i % 5 == 0:
            lr = optimizer.controller.current_lr if hasattr(optimizer, 'controller') else 1e-4
            print(f"Step {i} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

    print(f"Finished in {time.time() - start_time:.2f}s")
    print("Success! The model trained on your 4GB card.")
    
    # 6. Generate Chart
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Cruxy Meta-Lion (TinyLlama)', color='red', linewidth=2)
        plt.title('TinyLlama-1.1B Training on 4GB VRAM')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('tinyllama_chart.png')
        print("Chart saved to tinyllama_chart.png")
    except ImportError:
        print("Matplotlib not installed, skipping chart generation.")

if __name__ == "__main__":
    main()
