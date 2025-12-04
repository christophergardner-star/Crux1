import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cruxy.optimizer import CruxyOptimizer

print("Loading SmolLM2 1.7B...")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B", torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

optimizer = CruxyOptimizer(model.parameters(), lr=1e-5, mode="meta3")
print("Cruxy optimizer initialized (meta3 mode)")

# Quick training test
texts = ["The meaning of life is", "AI will change the world by", "Python is a great language because"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")

print("\nTraining for 5 steps...")
model.train()
for step in range(5):
    optimizer.zero_grad()
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step(loss=loss.item())
    print(f"Step {step+1}: Loss = {loss.item():.4f}")

print("\nSUCCESS: Cruxy + SmolLM2 1.7B working on H200!")
print(f"Final GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
