import io
import json
import logging
import re
import shutil
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
_EMPTY_DATASET_MARKER = "_empty_dataset.json"

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
    return path.exists() and (
        (path / "dataset_info.json").exists() or (path / _EMPTY_DATASET_MARKER).exists()
    )


def _load_processed_dataset(path: Path, *, empty_rows: dict[str, list]) -> Dataset:
    if (path / _EMPTY_DATASET_MARKER).exists():
        return Dataset.from_dict(empty_rows)
    return Dataset.load_from_disk(str(path))


def _save_processed_dataset(
    dataset: Dataset, path: Path, *, empty_rows: dict[str, list]
) -> None:
    if path.exists():
        shutil.rmtree(path)

    if len(dataset) == 0:
        path.mkdir(parents=True, exist_ok=True)
        (path / _EMPTY_DATASET_MARKER).write_text(
            json.dumps({"columns": list(empty_rows.keys())})
        )
        return

    dataset.save_to_disk(str(path))


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


def _chunk_wikipedia_batch(batch: dict[str, list]) -> dict[str, list]:
    rows: dict[str, list] = {"text": [], "modality": []}
    for text in batch["text"]:
        for chunk in _chunk_text(text):
            rows["text"].append(chunk)
            rows["modality"].append("text")
    return rows


### Public


def preprocess_wikipedia(*, limit: int | None = None) -> Dataset:
    """Chunk Wikipedia articles into text-only Arrow dataset on disk."""
    suffix = f"_test{limit}" if limit else ""
    out_path = _PROCESSED_DIR / f"wikipedia{suffix}"
    empty_rows: dict[str, list] = {"text": [], "modality": []}
    if _already_processed(out_path):
        try:
            return _load_processed_dataset(out_path, empty_rows=empty_rows)
        except Exception:
            logger.warning("Rebuilding corrupt processed cache at %s", out_path)
            shutil.rmtree(out_path, ignore_errors=True)

    raw = load_raw_wikipedia(limit=limit)
    ds = raw.map(
        _chunk_wikipedia_batch,
        batched=True,
        batch_size=256,
        remove_columns=raw.column_names,
        desc="Chunking wikipedia",
    )
    _save_processed_dataset(ds, out_path, empty_rows=empty_rows)
    return ds


def preprocess_cauldron_config(config: str, *, limit: int | None = None) -> Dataset:
    """Convert one Cauldron config into a unified Arrow dataset on disk."""
    suffix = f"_test{limit}" if limit else ""
    out_path = _PROCESSED_DIR / "cauldron" / f"{config}{suffix}"
    asset_dir = _ASSETS_DIR / "cauldron" / f"{config}{suffix}"
    empty_rows: dict[str, list] = {"text": [], "image": [], "modality": []}
    if _already_processed(out_path):
        try:
            return _load_processed_dataset(out_path, empty_rows=empty_rows)
        except Exception:
            logger.warning("Rebuilding corrupt processed cache at %s", out_path)
            shutil.rmtree(out_path, ignore_errors=True)
            shutil.rmtree(asset_dir, ignore_errors=True)

    raw = load_raw_cauldron(config, limit=limit)

    skipped_images = 0

    def _normalize_cauldron_batch(
        batch: dict[str, list], indices: list[int]
    ) -> dict[str, list]:
        nonlocal skipped_images
        rows: dict[str, list] = {"text": [], "image": [], "modality": []}

        for example_idx, conversation, images in zip(
            indices, batch["texts"], batch["images"], strict=False
        ):
            text = _extract_cauldron_text(conversation)

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

        return rows

    ds = raw.map(
        _normalize_cauldron_batch,
        batched=True,
        with_indices=True,
        batch_size=32,
        remove_columns=raw.column_names,
        desc=f"Normalizing cauldron/{config}",
    )
    _save_processed_dataset(ds, out_path, empty_rows=empty_rows)
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
