import random
from typing import Tuple, List
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from transformers import CLIPProcessor
from PIL import Image
from datasets import load_dataset
import requests
from io import BytesIO

class Flickr30kHFDataset(IterableDataset):
    """
    IterableDataset that yields dicts:
      {
        'pixel_values': Tensor(3,H,W),
        'input_ids':    Tensor(T-1,),
        'labels':       Tensor(T-1,)
      }
    Shards its sample list automatically across workers.
    """
    def __init__(self, hf_dataset, processor, max_length=32):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = max_length

    def __iter__(self):
        worker = get_worker_info()
        n = len(self.dataset)
        if worker is None:
            start, end = 0, n
        else:
            per_worker = int(n / worker.num_workers)
            worker_id = worker.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker.num_workers - 1 else n

        for item in self.dataset:
            image = item["image"]
            # Download image from URL
            # response = requests.get(item["image"])
            # image = Image.open(BytesIO(response.content)).convert("RGB")
            # Each item has a list of 5 captions, you can pick one or use all
            caption = random.choice(item["caption"])  # or random.choice(item["caption"]

            out = self.processor(
                text=[caption],
                images=[image],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = out["input_ids"][0]
            pixel_values = out["pixel_values"][0]
            decoder_input = input_ids[:-1]
            label = input_ids[1:]

            yield {
                "pixel_values": pixel_values,
                "input_ids": decoder_input,
                "labels": label,
            }

if __name__ == "__main__":
    print("HELLO WORLD")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Load the Hugging Face Flickr30k dataset (test split as example)
    hf_dataset = load_dataset("nlphuji/flickr30k", split="test")
    print(hf_dataset[0])
    print("getting here?")
    ds = Flickr30kHFDataset(
        hf_dataset=hf_dataset,
        processor=processor,
        max_length=32,
    )

    # DataLoader will automatically shard ds across 4 workers
    loader = DataLoader(
        ds,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
    )

    for batch in loader:
        print(batch["pixel_values"].shape, #(B,3,H,W)
              batch["input_ids"].shape, #(B,T-1)
              batch["labels"].shape) #(B,T-1)
        break