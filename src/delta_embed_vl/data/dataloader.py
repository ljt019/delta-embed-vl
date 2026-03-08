from functools import partial
from typing import cast

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import Qwen3VLProcessor

from delta_embed_vl.data.dataset import (
    MultimodalEmbeddingDataset,
    NormalizedSample,
)
from delta_embed_vl.model.embedding_inputs import (
    DEFAULT_EMBED_INSTRUCTION,
    EmbeddingInput,
    build_student_batch,
    get_student_processor,
)


def create_multimodal_dataloader(
    datasets: list[tuple[str, Dataset]],
    *,
    batch_size: int = 4,
    shuffle: bool = False,
    max_length: int = 512,
    raw_datasets_by_source: dict[str, Dataset] | None = None,
) -> DataLoader:
    processor = get_student_processor()
    dataset = MultimodalEmbeddingDataset(
        datasets,
        raw_datasets_by_source=raw_datasets_by_source,
    )

    return DataLoader(
        # `MultimodalEmbeddingDataset` is a map-style dataset and works with
        # `DataLoader` at runtime, but it is duck-typed rather than a literal
        # `torch.utils.data.Dataset` subclass, so we cast for the type checker.
        dataset=cast(TorchDataset[NormalizedSample], dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(
            collate_samples,
            processor=processor,
            max_length=max_length,
        ),
    )


def collate_samples(
    samples: list[NormalizedSample],
    *,
    processor: Qwen3VLProcessor,
    max_length: int,
) -> dict[str, torch.Tensor]:
    embedding_inputs = [_to_embedding_input(sample) for sample in samples]

    # Let the existing Qwen batching path do the real work:
    # tokenization, multimodal packing, padding, truncation, etc.
    batch = build_student_batch(
        processor,
        embedding_inputs,
        max_length=max_length,
    )

    # Return a plain dict so the training loop can move tensors
    # to device later without caring about HF wrapper types.
    return {key: value for key, value in batch.items()}


def _to_embedding_input(sample: NormalizedSample) -> EmbeddingInput:
    return EmbeddingInput(
        text=sample.text,
        image=sample.image,
        instruction=DEFAULT_EMBED_INSTRUCTION,
    )
