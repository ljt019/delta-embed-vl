import re
from dataclasses import dataclass
from typing import Literal

from datasets import Dataset, concatenate_datasets
from PIL import Image
from torch.utils.data import DataLoader

from delta_embed_vl.data.media import coerce_image_to_rgb

_IMAGE_TOKEN = re.compile(r"<image>\s*")

# Some Cauldron configs contain very long QA chains. Cap usable turns per raw row
# so a few rows do not dominate the training set.
_MAX_CAULDRON_TURNS_PER_ROW = 4


def create_multimodal_dataloader(
    datasets: list[Dataset], *, batch_size: int = 4, shuffle: bool = False
) -> DataLoader:
    # set format to python for easy debugging
    combined_dataset: Dataset = concatenate_datasets(datasets)
    combined_dataset.set_format("python")

    # Huggingface 'Dataset' implements the interface of
    # torch 'Dataset', but doesn't subclass it correctly
    # https://github.com/huggingface/datasets/issues/7500
    return DataLoader(
        dataset=combined_dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=shuffle,
    )


@dataclass(frozen=True)
class NormalizedSample:
    source: str
    role: Literal["query", "document"]
    text: str | None
    image: Image.Image | None


def normalize_row(row: dict, *, source: str) -> list[NormalizedSample]:
    if source == "wikipedia":
        return _normalize_wikipedia_row(row, source=source)

    if source.startswith("cauldron/"):
        return _normalize_cauldron_row(row, source=source)

    return []


def _normalize_wikipedia_row(row: dict, *, source: str) -> list[NormalizedSample]:
    text = row.get("text")
    if not isinstance(text, str):
        return []

    text = text.strip()
    if not text:
        return []

    return [
        NormalizedSample(
            source=source,
            role="document",
            text=text,
            image=None,
        )
    ]


def _normalize_cauldron_row(row: dict, *, source: str) -> list[NormalizedSample]:
    texts = row.get("texts")
    images = row.get("images")

    if not isinstance(texts, list):
        return []

    if not isinstance(images, list):
        return []

    resolved_images = [
        resolved_image
        for image in images
        if (resolved_image := coerce_image_to_rgb(image)) is not None
    ]
    if not resolved_images:
        return []

    normalized_samples: list[NormalizedSample] = []
    emitted_turns = 0

    for turn in texts:
        if emitted_turns >= _MAX_CAULDRON_TURNS_PER_ROW:
            break
        if not isinstance(turn, dict):
            continue

        user_text = _IMAGE_TOKEN.sub("", str(turn.get("user", ""))).strip() or None
        assistant_text = (
            _IMAGE_TOKEN.sub("", str(turn.get("assistant", ""))).strip() or None
        )
        if user_text is None and assistant_text is None:
            continue

        for image in resolved_images:
            if user_text is not None:
                normalized_samples.append(
                    NormalizedSample(
                        source=source,
                        role="query",
                        text=user_text,
                        image=image,
                    )
                )

            if assistant_text is not None:
                normalized_samples.append(
                    NormalizedSample(
                        source=source,
                        role="document",
                        text=assistant_text,
                        image=image,
                    )
                )
        emitted_turns += 1

    if normalized_samples:
        return normalized_samples

    return [
        NormalizedSample(
            source=source,
            role="document",
            text=None,
            image=image,
        )
        for image in resolved_images
    ]
