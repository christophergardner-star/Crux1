import torch
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from cruxy import CruxyOptimizer

def main():
    print("--- Axiom Forge Systems: Qwen 2.5 (1.5B) 4GB VRAM Demo ---")
    
    # 1. Configuration
    model_id = "Qwen/Qwen2.5-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")

    # 2. Load Model & Tokenizer
    print(f"Loading {model_id} in float16...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    # Enable Gradient Checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 3. Apply LoRA
    print("Applying LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Optimizer: Cruxy Meta-Lion
    print("Initializing Cruxy Meta-Lion...")
    optimizer = CruxyOptimizer(
        model.parameters(),
        lr=1e-4,
        mode="meta3",
        use_lion=True,
        weight_decay=0.01
    )

    # 5. Training Loop
    text = "Artificial General Intelligence requires stable optimization."
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
        
        if i % 5 == 0:
            print(f"Step {i} | Loss: {loss.item():.4f}")

    print(f"Finished in {time.time() - start_time:.2f}s")
    print("Success! Qwen 2.5-1.5B trained on your 4GB card.")
    
    # 6. Generate Chart
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Cruxy Meta-Lion (Qwen 1.5B)', color='blue', linewidth=2)
        plt.title('Qwen 2.5-1.5B Training on 4GB VRAM')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('qwen_chart.png')
    except ImportError:
        pass

if __name__ == "__main__":
    main()
