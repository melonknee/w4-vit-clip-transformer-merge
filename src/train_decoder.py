import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from tqdm import tqdm

from dataloader import Flickr30kIterableDataset
from models import CLIPCaptioningModel

def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    pad_token_id: int,
):
    model.train()
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    for batch in tqdm(loader, desc="Training", leave=False):
        pixel_values = batch["pixel_values"].to(device)  # (B,3,H,W)
        input_ids = batch["input_ids"].to(device)     # (B,T)
        labels = batch["labels"].to(device)        # (B,T)

        # Forward
        logits = model(pixel_values, input_ids)           # (B,T,vocab_size)

        # Compute loss: flatten vocab dimension
        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == "__main__":
    # Config
    data_root    = "./data"
    ann_file     = os.path.join(data_root, "Flickr30k.token.txt")
    model_name   = "openai/clip-vit-base-patch32"
    batch_size   = 16
    num_workers  = 4
    num_epochs   = 5
    lr           = 1e-4
    max_length   = 32

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model & optimizer
    model = CLIPCaptioningModel(
        clip_model_name=model_name,
        max_length=max_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Processor
    processor = CLIPProcessor.from_pretrained(model_name)

    # Dataset & DataLoader
    ds = Flickr30kIterableDataset(
        root=data_root,
        ann_file=ann_file,
        processor=processor,
        max_length=max_length,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Training loop
    for epoch in range(1, num_epochs + 1):
        avg_loss = train(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            pad_token_id=model.tokenizer.pad_token_id,
        )
        print(f"Epoch {epoch}/{num_epochs} — avg loss: {avg_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), "data/clip_captioning.pt")
    print("Training complete. Model saved to checkpoints/clip_captioning.pt")
