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
_EMPTY_DATASET_MARKER = "_empty_dataset.json"

### Private

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MIN_CHUNK_CHARS = 100


def _progress_interval(total: int) -> int:
    return max(1_000, (total + 19) // 20)


def _split_oversized_span(text: str, *, max_chars: int) -> list[str]:
    """Split a long span on whitespace, with hard fallback on raw chars."""
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    if not words:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            chunks.extend(
                word[i : i + max_chars] for i in range(0, len(word), max_chars)
            )
            continue

        sep_len = 1 if current else 0
        if current and current_len + sep_len + len(word) > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
            continue

        current.append(word)
        current_len += sep_len + len(word)

    if current:
        chunks.append(" ".join(current))

    return chunks


def _chunk_text(text: str, *, chunk_chars: int = 1200) -> list[str]:
    """Split text into non-overlapping passages, breaking on sentence boundaries.

    chunk_chars is a target and a hard cap.
    """
    text = " ".join(text.split()).strip()
    if not text:
        return []

    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        for sentence_part in _split_oversized_span(sentence, max_chars=chunk_chars):
            sep_len = 1 if current else 0
            if current and current_len + sep_len + len(sentence_part) > chunk_chars:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
                sep_len = 0

            current.append(sentence_part)
            current_len += sep_len + len(sentence_part)

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


def _has_usable_image(
    image: Image.Image | dict[str, Any] | str | Path | None,
) -> bool:
    if image is None:
        return False
    if isinstance(image, Image.Image):
        return True
    if isinstance(image, (str, Path)):
        return Path(image).exists()

    if image.get("bytes") is not None:
        return True

    image_path = image.get("path")
    return bool(image_path) and Path(image_path).exists()


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
            dataset = _load_processed_dataset(out_path, empty_rows=empty_rows)
            logger.info("wikipedia: cached rows=%d", len(dataset))
            return dataset
        except Exception:
            logger.warning("Rebuilding corrupt processed cache at %s", out_path)
            shutil.rmtree(out_path, ignore_errors=True)

    raw = load_raw_wikipedia(limit=limit)
    total_articles = len(raw)
    progress_interval = _progress_interval(total_articles)
    processed_articles = 0
    next_progress_log = progress_interval

    logger.info("wikipedia: preprocessing %d articles", total_articles)

    def _chunk_wikipedia_batch_with_progress(batch: dict[str, list]) -> dict[str, list]:
        nonlocal processed_articles, next_progress_log
        rows = _chunk_wikipedia_batch(batch)
        processed_articles += len(batch["text"])
        if (
            processed_articles >= next_progress_log
            or processed_articles == total_articles
        ):
            logger.info(
                "wikipedia: %d / %d articles", processed_articles, total_articles
            )
            while next_progress_log <= processed_articles:
                next_progress_log += progress_interval
        return rows

    ds = raw.map(
        _chunk_wikipedia_batch_with_progress,
        batched=True,
        batch_size=256,
        remove_columns=raw.column_names,
    )
    _save_processed_dataset(ds, out_path, empty_rows=empty_rows)
    logger.info("wikipedia: done rows=%d", len(ds))
    return ds


def preprocess_cauldron_config(config: str, *, limit: int | None = None) -> Dataset:
    """Convert one Cauldron config into a unified Arrow dataset on disk."""
    suffix = f"_test{limit}" if limit else ""
    out_path = _PROCESSED_DIR / "cauldron" / f"{config}{suffix}"
    empty_rows: dict[str, list] = {"text": [], "modality": []}
    if _already_processed(out_path):
        try:
            dataset = _load_processed_dataset(out_path, empty_rows=empty_rows)
            logger.info("cauldron/%s: cached rows=%d", config, len(dataset))
            return dataset
        except Exception:
            logger.warning("Rebuilding corrupt processed cache at %s", out_path)
            shutil.rmtree(out_path, ignore_errors=True)

    raw = load_raw_cauldron(config, limit=limit)
    total_examples = len(raw)
    progress_interval = _progress_interval(total_examples)
    processed_examples = 0
    next_progress_log = progress_interval

    skipped_images = 0
    logger.info("cauldron/%s: preprocessing %d examples", config, total_examples)

    def _normalize_cauldron_batch(batch: dict[str, list]) -> dict[str, list]:
        nonlocal next_progress_log, processed_examples, skipped_images
        rows: dict[str, list] = {"text": [], "modality": []}

        for conversation, images in zip(batch["texts"], batch["images"], strict=False):
            text = _extract_cauldron_text(conversation)

            if not images:
                if text:
                    rows["text"].append(text)
                    rows["modality"].append("text")
                continue

            for image in images:
                if not _has_usable_image(image):
                    skipped_images += 1
                    continue

                modality = "text_image" if text else "image"
                rows["text"].append(text)
                rows["modality"].append(modality)

        processed_examples += len(batch["texts"])
        if (
            processed_examples >= next_progress_log
            or processed_examples == total_examples
        ):
            logger.info(
                "cauldron/%s: %d / %d examples",
                config,
                processed_examples,
                total_examples,
            )
            while next_progress_log <= processed_examples:
                next_progress_log += progress_interval
        return rows

    ds = raw.map(
        _normalize_cauldron_batch,
        batched=True,
        batch_size=32,
        remove_columns=raw.column_names,
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
