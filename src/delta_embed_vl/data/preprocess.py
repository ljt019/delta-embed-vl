import json
import logging
import re
import shutil
from pathlib import Path

from datasets import Dataset

from delta_embed_vl.artifacts import versioned_name
from delta_embed_vl.data.download import (
    CAULDRON_CONFIGS,
    download_data,
    load_raw_cauldron,
    load_raw_wikipedia,
)
from delta_embed_vl.data.media import has_usable_image
from delta_embed_vl.model.embedding_inputs import (
    EmbeddingInput,
    count_teacher_prompt_tokens,
    teacher_prompt_token_limit,
)
from delta_embed_vl.progress import ProgressBar
from delta_embed_vl.settings import Settings

logger = logging.getLogger(__name__)
_PROCESSED_DIR = Settings().data_dir / "processed"
_EMPTY_DATASET_MARKER = "_empty_dataset.json"

### Private

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MIN_CHUNK_CHARS = 100


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


def _fits_teacher_prompt(text: str) -> bool:
    return (
        count_teacher_prompt_tokens(EmbeddingInput(text=text))
        <= teacher_prompt_token_limit()
    )


def _split_raw_teacher_safe(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        low = 1
        high = len(text) - start
        best = 0

        while low <= high:
            mid = (low + high) // 2
            candidate = text[start : start + mid].strip()
            if candidate and _fits_teacher_prompt(candidate):
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        if best == 0:
            best = 1

        chunk = text[start : start + best].strip()
        if chunk:
            chunks.append(chunk)
        start += best

    return chunks


def _ensure_teacher_safe(text: str) -> list[str]:
    if _fits_teacher_prompt(text):
        return [text]

    words = text.split()
    if len(words) <= 1:
        return _split_raw_teacher_safe(text)

    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        if not current:
            if _fits_teacher_prompt(word):
                current = [word]
            else:
                chunks.extend(_split_raw_teacher_safe(word))
            continue

        candidate = " ".join([*current, word])
        if _fits_teacher_prompt(candidate):
            current.append(word)
            continue

        chunks.append(" ".join(current))
        if _fits_teacher_prompt(word):
            current = [word]
        else:
            chunks.extend(_split_raw_teacher_safe(word))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def _chunk_text(text: str, *, chunk_chars: int = 1200) -> list[str]:
    """Split text into passages and enforce teacher-safe prompt lengths."""
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

    safe_chunks: list[str] = []
    for chunk in chunks:
        safe_chunks.extend(_ensure_teacher_safe(chunk))

    return safe_chunks


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
    out_path = _PROCESSED_DIR / versioned_name("wikipedia", limit=limit)
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
    processed_articles = 0
    emitted_rows = 0
    progress = ProgressBar(
        label="wikipedia",
        total=total_articles,
        unit="articles",
    )

    logger.info("wikipedia: preprocessing %d articles", total_articles)

    def _chunk_wikipedia_batch_with_progress(batch: dict[str, list]) -> dict[str, list]:
        nonlocal emitted_rows, processed_articles
        rows = _chunk_wikipedia_batch(batch)
        processed_articles += len(batch["text"])
        emitted_rows += len(rows["text"])
        progress.update(
            processed_articles,
            extra=f"rows={emitted_rows:,}",
        )
        return rows

    ds = raw.map(
        _chunk_wikipedia_batch_with_progress,
        batched=True,
        batch_size=256,
        remove_columns=raw.column_names,
    )
    progress.close(extra=f"rows={emitted_rows:,}")
    _save_processed_dataset(ds, out_path, empty_rows=empty_rows)
    logger.info("wikipedia: done rows=%d", len(ds))
    return ds


def preprocess_cauldron_config(config: str, *, limit: int | None = None) -> Dataset:
    """Convert one Cauldron config into a unified Arrow dataset on disk."""
    out_path = _PROCESSED_DIR / "cauldron" / versioned_name(config, limit=limit)
    empty_rows: dict[str, list] = {
        "text": [],
        "modality": [],
        "source_index": [],
        "image_index": [],
    }
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
    processed_examples = 0

    skipped_images = 0
    emitted_rows = 0
    progress = ProgressBar(
        label=f"cauldron/{config}",
        total=total_examples,
        unit="examples",
    )
    logger.info("cauldron/%s: preprocessing %d examples", config, total_examples)

    def _normalize_cauldron_batch(batch: dict[str, list]) -> dict[str, list]:
        nonlocal emitted_rows, processed_examples, skipped_images
        rows: dict[str, list] = {
            "text": [],
            "modality": [],
            "source_index": [],
            "image_index": [],
        }
        batch_start = processed_examples

        for offset, (conversation, images) in enumerate(
            zip(batch["texts"], batch["images"], strict=False)
        ):
            source_index = batch_start + offset
            text = _extract_cauldron_text(conversation)

            if not images:
                if text:
                    rows["text"].append(text)
                    rows["modality"].append("text")
                    rows["source_index"].append(source_index)
                    rows["image_index"].append(-1)
                continue

            for image_index, image in enumerate(images):
                if not has_usable_image(image):
                    skipped_images += 1
                    continue

                modality = "text_image" if text else "image"
                rows["text"].append(text)
                rows["modality"].append(modality)
                rows["source_index"].append(source_index)
                rows["image_index"].append(image_index)

        processed_examples += len(batch["texts"])
        emitted_rows += len(rows["text"])
        progress.update(
            processed_examples,
            extra=f"rows={emitted_rows:,}, skipped_images={skipped_images:,}",
        )
        return rows

    ds = raw.map(
        _normalize_cauldron_batch,
        batched=True,
        batch_size=32,
        remove_columns=raw.column_names,
    )
    progress.close(extra=f"rows={emitted_rows:,}, skipped_images={skipped_images:,}")
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
