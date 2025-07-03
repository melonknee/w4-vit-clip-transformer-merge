from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Step 1: Load Qwen3-0.6B-base model and tokenizer from Hugging Face
model_name = "Qwen/Qwen3-0.6B-Base"  # or "Qwen/Qwen1.5-0.6B" when it's published
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()

# Step 2: Define a dummy prompt
prompt = "A photo of"

# Step 3: Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Step 4: Generate text from the prompt
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)

# Step 5: Decode and print output
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated caption:", decoded)
