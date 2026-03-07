import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import BaseImageProcessor, PreTrainedTokenizerBase


class Collator:
    """Batches samples into tensors using a tokenizer and image processor.

    Handles mixed-modality batches: text-only samples get tokenized,
    image samples get processed, and an image_mask tracks which batch
    entries have images so the model can route accordingly.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: BaseImageProcessor,
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | None]:
        texts = [s["text"] or "" for s in batch]
        images = [s.get("image") for s in batch]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        image_mask = torch.tensor([img is not None for img in images])

        pixel_values = None
        valid_images = [img for img in images if img is not None]
        if valid_images:
            processed = self.image_processor(valid_images, return_tensors="pt")
            pixel_values = processed["pixel_values"]

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "pixel_values": pixel_values,
            "image_mask": image_mask,
        }


def create_dataloader(
    datasets: list[Dataset],
    *,
    tokenizer: PreTrainedTokenizerBase,
    image_processor: BaseImageProcessor,
    max_length: int = 512,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Build a DataLoader from one or more HF Arrow datasets.

    Datasets are concatenated and served via memory-mapping — no full
    materialization in RAM.
    """
    combined = concatenate_datasets(datasets)
    combined.set_format("python")
    return DataLoader(
        combined,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Collator(tokenizer, image_processor, max_length),
        pin_memory=True,
    )
