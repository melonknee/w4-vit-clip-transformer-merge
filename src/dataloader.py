from typing import Tuple, List
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision.datasets import Flickr30k
from transformers import CLIPProcessor
from PIL import Image

class Flickr30kIterableDataset(IterableDataset):
    """
    IterableDataset that yields dicts:
      {
        'pixel_values': Tensor(3,H,W),
        'input_ids':    Tensor(T-1,),
        'labels':       Tensor(T-1,)
      }
    Shards its sample list automatically across workers.
    """
    def __init__(
        self,
        root: str,
        ann_file: str,
        processor: CLIPProcessor,
        max_length: int = 32,
    ):
        # load raw image paths + captions
        base = Flickr30k(root=root, ann_file=ann_file)
        samples: List[Tuple[str,str]] = []
        for img, caps in base:
            path = img.filename
            for cap in caps:
                samples.append((path, cap))
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start = 0
            end = len(self.samples)
        else:
            # evenly split samples across workers
            per_worker = int(len(self.samples) / worker.num_workers)
            worker_id = worker.id
            start = worker_id * per_worker
            # last worker takes the rest
            end = start + per_worker if worker_id < worker.num_workers - 1 else len(self.samples)

        for path, caption in self.samples[start:end]:
            image = Image.open(path).convert("RGB")
            out = self.processor(
                text=[caption],
                images=[image],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = out["input_ids"][0]        # (T,)
            pixel_values = out["pixel_values"][0]  # (3,H,W)
            decoder_input = input_ids[:-1]         # drop last token
            label        = input_ids[1:]           # drop first token

            yield {
                "pixel_values": pixel_values,
                "input_ids":    decoder_input,
                "labels":       label,
            }

if __name__ == "__main__":
    # instantiate processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # make the iterable dataset
    ds = Flickr30kIterableDataset(
        root="./data",
        ann_file="./data/Flickr30k.token.txt",
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
        # batch["pixel_values"]: (B,3,H,W)
        # batch["input_ids"]:    (B,T-1)
        # batch["labels"]:       (B,T-1)
        print(batch["pixel_values"].shape,
              batch["input_ids"].shape,
              batch["labels"].shape)
        break