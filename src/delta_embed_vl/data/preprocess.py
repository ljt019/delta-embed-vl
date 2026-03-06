import io
import logging
import re
from pathlib import Path
from typing import Any

from datasets import Dataset
from PIL import Image

from delta_embed_vl.data.download import (
    CAULDRON_CONFIGS,
    download_data,
    load_raw_cauldron,
    load_raw_wikipedia,
)
from delta_embed_vl.settings import Settings

logger = logging.getLogger(__name__)
_PROCESSED_DIR = Settings().data_dir / "processed"
_ASSETS_DIR = _PROCESSED_DIR / "assets"

### Private

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MIN_CHUNK_CHARS = 100


def _chunk_text(text: str, *, chunk_chars: int = 1200) -> list[str]:
    """Split text into non-overlapping passages, breaking on sentence boundaries.

    chunk_chars is a heuristic target, not a hard cap.
    """
    text = " ".join(text.split()).strip()
    if not text:
        return []

    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sep_len = 1 if current else 0
        if current and current_len + sep_len + len(sentence) > chunk_chars:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
            sep_len = 0

        current.append(sentence)
        current_len += sep_len + len(sentence)

    if current:
        tail = " ".join(current)
        if chunks and len(tail) < _MIN_CHUNK_CHARS:
            chunks[-1] += " " + tail
        else:
            chunks.append(tail)

    return chunks


_IMAGE_TOKEN = re.compile(r"<image>\s*")


def _extract_cauldron_text(conversation: list[dict]) -> str | None:
    """Flatten a Cauldron conversation into a single text string.

    Cauldron samples have a `texts` column shaped like:
        [{"user": "...", "assistant": "..."}, ...]

    Strips <image> placeholder tokens, concatenates all turns.
    Returns None if the result is empty.
    """
    parts: list[str] = []
    for turn in conversation:
        user = _IMAGE_TOKEN.sub("", turn.get("user", "")).strip()
        assistant = turn.get("assistant", "").strip()
        if user:
            parts.append(user)
        if assistant:
            parts.append(assistant)

    text = " ".join(parts).strip()
    return text or None


def _already_processed(path: Path) -> bool:
    return path.exists() and (path / "dataset_info.json").exists()


def _coerce_image_to_rgb(
    image: Image.Image | dict[str, Any] | str | Path | None,
) -> Image.Image | None:
    if image is None:
        return None

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            return None
        with Image.open(path) as loaded:
            return loaded.convert("RGB")

    image_bytes = image.get("bytes")
    if image_bytes is not None:
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        elif isinstance(image_bytes, bytearray):
            image_bytes = bytes(image_bytes)
        elif isinstance(image_bytes, list):
            image_bytes = bytes(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as loaded:
            return loaded.convert("RGB")

    image_path = image.get("path")
    if image_path:
        path = Path(image_path)
        if path.exists():
            with Image.open(path) as loaded:
                return loaded.convert("RGB")

    return None


def _materialize_image_asset(
    image: Image.Image | dict[str, Any] | str | Path | None,
    *,
    out_path: Path,
) -> str | None:
    normalized = _coerce_image_to_rgb(image)
    if normalized is None:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        normalized.save(out_path, format="PNG")
    return str(out_path)


### Public


def preprocess_wikipedia(*, limit: int | None = None) -> Dataset:
    """Chunk Wikipedia articles into text-only Arrow dataset on disk."""
    suffix = f"_test{limit}" if limit else ""
    out_path = _PROCESSED_DIR / f"wikipedia{suffix}"
    if _already_processed(out_path):
        return Dataset.load_from_disk(str(out_path))

    raw = load_raw_wikipedia(limit=limit)

    rows: dict[str, list] = {"text": [], "modality": []}
    for article in raw:
        for chunk in _chunk_text(article["text"]):
            rows["text"].append(chunk)
            rows["modality"].append("text")

    ds = Dataset.from_dict(rows)
    ds.save_to_disk(str(out_path))
    return ds


def preprocess_cauldron_config(config: str, *, limit: int | None = None) -> Dataset:
    """Convert one Cauldron config into a unified Arrow dataset on disk."""
    suffix = f"_test{limit}" if limit else ""
    out_path = _PROCESSED_DIR / "cauldron" / f"{config}{suffix}"
    asset_dir = _ASSETS_DIR / "cauldron" / f"{config}{suffix}"
    if _already_processed(out_path):
        return Dataset.load_from_disk(str(out_path))

    raw = load_raw_cauldron(config, limit=limit)

    rows: dict[str, list] = {"text": [], "image": [], "modality": []}
    skipped_images = 0
    for example_idx, example in enumerate(raw):
        text = _extract_cauldron_text(example["texts"])
        images = example["images"]

        if not images:
            if text:
                rows["text"].append(text)
                rows["image"].append(None)
                rows["modality"].append("text")
            continue

        for image_idx, image in enumerate(images):
            image_path = _materialize_image_asset(
                image,
                out_path=asset_dir / f"{example_idx:08d}_{image_idx:02d}.png",
            )
            if image_path is None:
                skipped_images += 1
                continue

            modality = "text_image" if text else "image"
            rows["text"].append(text)
            rows["image"].append(image_path)
            rows["modality"].append(modality)

    ds = Dataset.from_dict(rows)
    ds.save_to_disk(str(out_path))
    logger.info(
        "Processed cauldron/%s: rows=%d skipped_images=%d",
        config,
        len(ds),
        skipped_images,
    )
    return ds


def preprocess_cauldron(*, limit: int | None = None) -> list[Dataset]:
    """Process all Cauldron configs, returning a list of Arrow datasets."""
    return [
        preprocess_cauldron_config(config, limit=limit) for config in CAULDRON_CONFIGS
    ]


def preprocess_data(
    download_first: bool = False, *, limit: int | None = None
) -> tuple[Dataset, list[Dataset]]:
    """Entry point: returns (wikipedia_dataset, cauldron_datasets).

    Each is memory-mapped from disk — no RAM blow-up.
    Pass limit=N to only process the first N raw entries (for testing).
    """
    if download_first:
        download_data(limit=limit)

    wikipedia = preprocess_wikipedia(limit=limit)
    cauldron = preprocess_cauldron(limit=limit)
    return wikipedia, cauldron
