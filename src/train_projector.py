## NOTE: This training loop keeps the CLIP encoder frozen,
## partially trains the Qwen layers
## and fully trains the projector layers which bridge CLIP to Qwen

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from torchvision.transforms import ToTensor
import os
import random

from datasets import concatenate_datasets

## BATCHING STUFF
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image

num_epochs = 1000  # or more

def collate_fn(batch):
    images, captions = zip(*batch)

    # 1. Process images through CLIP processor (batched)
    clip_inputs = clip_processor(images=list(images), return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        clip_features = clip_model.get_image_features(**clip_inputs)  # [B, 512]

    # 2. Tokenize captions
    tokenized = qwen_tokenizer(
        list(captions),
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    return clip_features, tokenized["input_ids"], images, captions


class FlickrFaceDataset(Dataset):
    def __init__(self, dataset, clip_processor, tokenizer):
        self.dataset = dataset
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        caption = self.dataset[idx]["text"]
        return image, caption
##########################

# Unfreeze last N decoder layers of Qwen
def unfreeze_qwen_layers(model, num_layers_to_unfreeze=6):
    total_unfrozen = 0

    for name, module in model.named_modules():
        # Match transformer layers (like model.layers.29)
        if name.startswith("model.layers."):
            layer_num = int(name.split(".")[2])
            total_layers = model.config.num_hidden_layers
            if layer_num >= total_layers - num_layers_to_unfreeze:
                for param in module.parameters():
                    param.requires_grad = True
                total_unfrozen += 1

    print(f"✅ Unfrozen Qwen layers (total {total_unfrozen})")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
qwen_model_name = "Qwen/Qwen1.5-0.5B"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_name, trust_remote_code=True).to(device)

unfreeze_qwen_layers(qwen_model, num_layers_to_unfreeze=6)

# ✅ Set pad_token_id to eos_token_id so qwen knows how to pad
if qwen_tokenizer.pad_token_id is None:
    qwen_tokenizer.pad_token_id = qwen_tokenizer.eos_token_id

# Freeze CLIP 
for p in clip_model.parameters():
    p.requires_grad = False

# 🔓 Unfreeze first N transformer blocks of Qwen (e.g., 0–5)
# NUM_UNFROZEN_LAYERS = 6
# for name, param in qwen_model.named_parameters():
#     if any(f"transformer.h.{i}." in name for i in range(NUM_UNFROZEN_LAYERS)):
#         param.requires_grad = True

# trainable_params = [n for n, p in qwen_model.named_parameters() if p.requires_grad]
# print(f"✅ Unfrozen Qwen layers (total {len(trainable_params)}):")
# for name in trainable_params:
#     print(" -", name)

trainable_params = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)
print(f"🧠 Trainable Qwen parameters: {trainable_params:,}")


# Create projector
projector = nn.Linear(512, qwen_model.config.hidden_size).to(device)

# Optimizer (only projector's parameters)
# Gather all trainable parameters (projector + unfrozen Qwen layers)
trainable_params = list(projector.parameters()) + [
    p for n, p in qwen_model.named_parameters() if p.requires_grad
]
optimizer = optim.Adam(trainable_params, lr=1e-4)


# Load dataset
# dataset = load_dataset("alexg99/captioned_flickr_faces")["train"] # PRE-BATCHING VERSION
# # dataset["train"]: list of examples like {"image", "caption"}

# WITH BATCHING VVV
# Load both datasets
dataset1 = load_dataset("alexg99/captioned_flickr_faces")["train"]
dataset2 = load_dataset("alexg99/captioned_flickr_faces_2")["train"]

# Combine them
raw_dataset = concatenate_datasets([dataset1, dataset2])
train_dataset = FlickrFaceDataset(raw_dataset, clip_processor, qwen_tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
print(f"📊 Combined dataset size: {len(raw_dataset)} samples")

################## SAVING CHECKPOINT STUFF
# Create checkpoints directory if it doesn't exist
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SAVE_PATH = os.path.join(CHECKPOINT_DIR, "projector_checkpoint.pt")
LATEST_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")
# Resume from checkpoint if available

if os.path.exists(LATEST_PATH):
    print("🔁 Loading existing projector weights from:", LATEST_PATH)
    projector.load_state_dict(torch.load(LATEST_PATH))
elif os.path.exists(SAVE_PATH):
    print("🔁 Loading existing projector weights from:", SAVE_PATH)
    projector.load_state_dict(torch.load(SAVE_PATH))

##############################################

########### WANDB STUFF
import wandb
wandb.init(project="projector-training", name="flickr-faces-run", config={
    "learning_rate": 1e-4,
    "model": "qwen3-0.5B",
    "dataset": "captioned_flickr_faces",
    "trainable_params": sum(p.numel() for p in projector.parameters() if p.requires_grad),
})
wandb_table = wandb.Table(columns=["step", "image", "real_caption", "generated_caption"], 
                          allow_mixed_types=True, log_mode="MUTABLE")


##########################
# Training loop

### PRE-BATCHING
# for step, example in enumerate(dataset):
#     image = example["image"]
#     caption = example["text"]

#     # 1. Image → CLIP → projector
#     clip_inputs = clip_processor(images=image, return_tensors="pt").to(device)
#     with torch.no_grad():
#         clip_embedding = clip_model.get_image_features(**clip_inputs)  # [1, 512]
#     prefix_embed = projector(clip_embedding).unsqueeze(1)  # [1, 1, 2048]

#     # 2. Tokenize target caption
#     target = qwen_tokenizer(caption, return_tensors="pt")
#     target_input_ids = target["input_ids"].to(device)
#     labels = target_input_ids.clone()

#     with torch.no_grad():
#         caption_embeds = qwen_model.get_input_embeddings()(target_input_ids)  # [1, T, 2048]

#     # 3. Concatenate projector prefix + caption embeddings
#     inputs_embeds = torch.cat([prefix_embed, caption_embeds[:, :-1, :]], dim=1)  # Drop last token

#     # 4. Shift labels to match outputs
#     shifted_labels = labels[:, :]  # [1, T]
#     shifted_labels[shifted_labels == qwen_tokenizer.pad_token_id] = -100  # Ignore padding

#     # 5. Forward pass
#     outputs = qwen_model(inputs_embeds=inputs_embeds, labels=shifted_labels)

#     loss = outputs.loss
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()

#     if step % 10 == 0:

#         # SAVE EVERY 10 STEPS:
#         step_save_path = os.path.join(CHECKPOINT_DIR, f"projector_step_{step}.pt")
#         torch.save(projector.state_dict(), os.path.join(CHECKPOINT_DIR, "latest.pt"))
#         torch.save(projector.state_dict(), step_save_path)
#         print(f"💾 Saved checkpoint: {step_save_path}")

#         # =======================
#         # Print true vs generated
#         # =======================

#         # Re-generate caption from image using updated projector
#         with torch.no_grad():
#             generated = qwen_model.generate(
#                 inputs_embeds=torch.cat([prefix_embed, caption_embeds[:, :-1, :]], dim=1),
#                 max_new_tokens=30, num_beams=3,       # <--- Beam width
#             )
#             generated_text = qwen_tokenizer.decode(generated[0], skip_special_tokens=True)

#             print("🔍 Step", step)
#             print("📸 Real caption     :", caption)
#             print("🤖 Generated caption:", generated_text)
#             print("-" * 50)

#             wandb.log({"loss": loss.item(), "step": step,
#                    "real_caption": caption,
#                     "generated_caption": generated_text,})
#         print(f"[{step}] Loss: {loss.item():.4f}")

#         wandb_table.add_data(
#             step,
#             wandb.Image(image),
#             caption,
#             generated_text,
#         )
#         wandb.log({"caption_table": wandb_table})

#     if step == 100:
#         break  # For testing, limit steps
###


#### BATCHING VVVV ####
for epoch in range(num_epochs):
    print(f"🌙 Starting epoch {epoch}")
    for step, (clip_features, input_ids, images, captions) in enumerate(train_loader):

        prefix_embeds = projector(clip_features).unsqueeze(1)  # [B, 1, hidden]
        
        with torch.no_grad():
            caption_embeds = qwen_model.get_input_embeddings()(input_ids[:, :-1])  # [B, T-1, hidden]

        inputs_embeds = torch.cat([prefix_embeds, caption_embeds], dim=1)  # [B, T, hidden]
        
        labels = input_ids.clone()
        labels[labels == qwen_tokenizer.pad_token_id] = -100

        outputs = qwen_model(inputs_embeds=inputs_embeds, labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"[{step}] Loss: {loss.item():.4f}")

            # ===== SAVE CHECKPOINT =====
            step_ckpt_path = os.path.join(CHECKPOINT_DIR, f"projector_step_{step}.pt")
            torch.save(projector.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch{epoch}_step{step}.pt"))
            print(f"💾 Saved checkpoint: {step_ckpt_path}")

            # Also update latest.pt for resume
            latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
            torch.save(projector.state_dict(), latest_ckpt_path)

            
            # Generate caption from one image in the batch
            with torch.no_grad():
                # test_embed = prefix_embeds[0:1]  # Select first sample
                # attention_mask = torch.ones((1, test_embed.shape[1]), dtype=torch.long).to(device)

                # generated = qwen_model.generate(inputs_embeds=test_embed,
                #                                 max_new_tokens=30,
                #                                 num_beams=3,
                #                                 pad_token_id=qwen_tokenizer.pad_token_id,
                #                                 attention_mask=attention_mask,
                # )
                # generated_text = qwen_tokenizer.decode(generated[0], skip_special_tokens=True)

                # 👇 Step 1: Create the guiding prompt
                prompt_options = [
                    "Human: What does the person in this photo look like?\nAssistant: The image features a",
                    "Human: Describe the person's appearance.\nAssistant: The image features a",
                    "Human: Describe the face in terms of age, gender, and expression.\nAssistant: The image features a"
                ]
                prompt = random.choice(prompt_options)


                # 👇 Step 2: Tokenize prompt
                prompt_ids = qwen_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                # 👇 Step 3: Convert prompt to embeddings
                prompt_embeds = qwen_model.model.embed_tokens(prompt_ids)  # shape: [1, T, 2048]

                # 👇 Step 4: Select one sample from the batch
                prefix_embed = projector(clip_features.detach()[0:1]).unsqueeze(1)  # shape: [1, 1, 2048]

                # 👇 Step 5: Concatenate prefix and prompt
                inputs_embeds = torch.cat([prefix_embed, prompt_embeds], dim=1)  # shape: [1, 1+T, 2048]

                # 👇 Step 6: Create attention mask
                attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long).to(device)

                # 👇 Step 7: Generate caption
                generated = qwen_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    num_beams=3,
                    pad_token_id=qwen_tokenizer.pad_token_id,
                    # Optional for more diversity:
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8,
                )

                # 👇 Step 8: Decode and strip result
                generated_text = qwen_tokenizer.decode(generated[0], skip_special_tokens=True).strip()


            # =======================
            # Print true vs generated
            # =======================

            # Re-generate caption from image using updated projector
            with torch.no_grad():
                generated = qwen_model.generate(
                    inputs_embeds=torch.cat([prefix_embeds, caption_embeds[:, :-1, :]], dim=1),
                    max_new_tokens=30, num_beams=3,       # <--- Beam width
                )
                generated_text = qwen_tokenizer.decode(generated[0], skip_special_tokens=True)

                print("🔍 Step", step)
                print("📸 Real caption     :", captions[0])
                print("🤖 Generated caption:", generated_text)
                print("-" * 50)

            # Log to W&B table
            wandb_table.add_data(
                step,
                wandb.Image(images[0]),
                captions[0],
                generated_text,
            )
            wandb.log({
                "loss": loss.item(),
                "step": step,
                "caption_table": wandb_table
            })

        if step == 100:
            break
