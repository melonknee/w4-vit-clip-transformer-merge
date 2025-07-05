from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Step 1: Load the CLIP model from Hugging Face
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()  # Set to evaluation mode (not training)

# Step 2: Load the image processor
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Step 3: Load and preprocess an image (example)
image = Image.open("happy_bes.jpg")  # Replace with your own image

# Step 4: Preprocess image
inputs = clip_processor(images=image, return_tensors="pt")

# Step 5: Extract image embeddings
with torch.no_grad():
    image_embeddings = clip_model.get_image_features(**inputs)
print(clip_model)
print("Image Embedding Shape:", image_embeddings.shape)  # Expect [1, 512]
