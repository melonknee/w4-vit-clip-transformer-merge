import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from PIL import Image

# ====== Step 1: Load CLIP encoder ======
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# ====== Step 2: Load Qwen decoder ======
qwen_model_name = "Qwen/Qwen3-0.6B-Base"  # or "Qwen/Qwen1.5-0.6B" if released
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_name, trust_remote_code=True)
qwen_model.eval()

# ====== Step 3: Build the projection layer ======
# Qwen hidden size is 2048 (for 0.5B model); adjust if using 0.6B
qwen_hidden_size = qwen_model.config.hidden_size
projector = nn.Linear(512, qwen_hidden_size)  # CLIP: 512-d → Qwen hidden size
projector.eval()

# ====== Step 4: Load and preprocess an image ======
image = Image.open("happy_bes.jpg")  # Make sure this image exists!
clip_inputs = clip_processor(images=image, return_tensors="pt")

# ====== Step 5: Get image embedding ======
with torch.no_grad():
    image_embedding = clip_model.get_image_features(**clip_inputs)  # [1, 512]
    projected = projector(image_embedding)                          # [1, 2048]
    projected = projected.unsqueeze(1)  # → [1, 1, 2048] — like 1 fake "token"

# ====== Step 6: Tokenize a dummy prompt ======
prompt = "The image shows"  # Qwen will continue this
token_inputs = qwen_tokenizer(prompt, return_tensors="pt")
input_ids = token_inputs["input_ids"]                        # [1, T]
# input_embeds = qwen_model.transformer.wte(input_ids)         # [1, T, 2048]
input_embeds = qwen_model.get_input_embeddings()(input_ids)  # ✅ safer and works


# ====== Step 7: Concatenate image + text embeddings ======
combined_embeds = torch.cat([projected, input_embeds], dim=1)  # [1, 1+T, 2048]

# ====== Step 8: Generate text using inputs_embeds ======
with torch.no_grad():
    outputs = qwen_model.generate(inputs_embeds=combined_embeds, max_new_tokens=30)

# ====== Step 9: Decode and print ======
generated_text = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated caption:", generated_text)
