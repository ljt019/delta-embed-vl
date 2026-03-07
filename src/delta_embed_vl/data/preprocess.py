import json
import logging
import re
import shutil
from collections.abc import Callable
from pathlib import Path

from datasets import Dataset
from transformers import Qwen3VLProcessor

from delta_embed_vl.artifacts import versioned_name
from delta_embed_vl.data.download import (
    CAULDRON_CONFIGS,
    download_data,
    load_raw_cauldron,
    load_raw_wikipedia,
)
from delta_embed_vl.data.media import ImageLike, coerce_image_to_rgb, has_usable_image
from delta_embed_vl.model.embedding_inputs import (
    EmbeddingInput,
    count_teacher_processed_tokens,
    count_teacher_prompt_tokens,
    get_student_processor,
    sample_fits_student,
    student_batch_fit_flags,
    teacher_processed_token_lengths,
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


def _fits_teacher_processed(sample: EmbeddingInput) -> bool:
    return count_teacher_processed_tokens(sample) <= teacher_prompt_token_limit()


def _split_raw_safe(text: str, *, fits: Callable[[str], bool]) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        low = 1
        high = len(text) - start
        best = 0

        while low <= high:
            mid = (low + high) // 2
            candidate = text[start : start + mid].strip()
            if candidate and fits(candidate):
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


def _split_raw_teacher_safe(text: str) -> list[str]:
    return _split_raw_safe(text, fits=_fits_teacher_prompt)


def _ensure_text_safe(text: str, *, fits: Callable[[str], bool]) -> list[str]:
    if fits(text):
        return [text]

    words = text.split()
    if len(words) <= 1:
        return _split_raw_safe(text, fits=fits)

    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        if not current:
            if fits(word):
                current = [word]
            else:
                chunks.extend(_split_raw_safe(word, fits=fits))
            continue

        candidate = " ".join([*current, word])
        if fits(candidate):
            current.append(word)
            continue

        chunks.append(" ".join(current))
        if fits(word):
            current = [word]
        else:
            chunks.extend(_split_raw_safe(word, fits=fits))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def _ensure_teacher_safe(text: str) -> list[str]:
    return _ensure_text_safe(text, fits=_fits_teacher_prompt)


def _ensure_multimodal_joint_safe(
    text: str,
    image: ImageLike,
    *,
    processor: Qwen3VLProcessor,
    max_length: int,
) -> list[str]:
    return _ensure_text_safe(
        text,
        fits=lambda candidate: (
            _fits_teacher_processed(EmbeddingInput(text=candidate, image=image))
            and sample_fits_student(
                processor,
                EmbeddingInput(text=candidate, image=image),
                max_length=max_length,
            )
        ),
    )


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


def load_cauldron_embedding_input(
    raw_dataset: Dataset,
    row: dict,
    *,
    config: str | None = None,
    row_index: int | None = None,
) -> EmbeddingInput:
    text = row.get("text") or None
    image_index = int(row["image_index"])
    if image_index < 0:
        return EmbeddingInput(text=text)

    source_index = int(row["source_index"])
    source_images = raw_dataset[source_index]["images"]
    if image_index >= len(source_images):
        location = f"source={source_index} image={image_index}"
        if config is not None:
            location = f"cauldron/{config} {location}"
        if row_index is not None:
            location = f"{location} row={row_index}"
        raise ValueError(f"Image index out of range for {location}.")

    image = coerce_image_to_rgb(source_images[image_index])
    if image is None:
        location = f"source={source_index} image={image_index}"
        if config is not None:
            location = f"cauldron/{config} {location}"
        if row_index is not None:
            location = f"{location} row={row_index}"
        raise ValueError(f"Could not resolve image for {location}.")
    return EmbeddingInput(text=text, image=image)


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


def _append_cauldron_row(
    rows: dict[str, list],
    *,
    text: str | None,
    modality: str,
    source_index: int,
    image_index: int,
) -> None:
    rows["text"].append(text)
    rows["modality"].append(modality)
    rows["source_index"].append(source_index)
    rows["image_index"].append(image_index)


def _batched_processed_lengths(
    samples: list[EmbeddingInput], *, batch_size: int = 8
) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(samples), batch_size):
        lengths.extend(
            teacher_processed_token_lengths(samples[start : start + batch_size])
        )
    return lengths


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


def preprocess_cauldron_config(
    config: str,
    *,
    limit: int | None = None,
    student_max_length: int = 8192,
) -> Dataset:
    """Convert one Cauldron config into a unified Arrow dataset on disk."""
    out_path = (
        _PROCESSED_DIR
        / "cauldron"
        / f"{versioned_name(config, limit=limit)}_student{student_max_length}"
    )
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
    skipped_overlength = 0
    skipped_student_overlength = 0
    split_rows = 0
    emitted_rows = 0
    progress = ProgressBar(
        label=f"cauldron/{config}",
        total=total_examples,
        unit="examples",
    )
    logger.info("cauldron/%s: preprocessing %d examples", config, total_examples)
    student_processor = get_student_processor()

    def _normalize_cauldron_batch(batch: dict[str, list]) -> dict[str, list]:
        nonlocal \
            emitted_rows, \
            processed_examples, \
            skipped_images, \
            skipped_overlength, \
            skipped_student_overlength, \
            split_rows
        rows: dict[str, list] = {
            "text": [],
            "modality": [],
            "source_index": [],
            "image_index": [],
        }
        batch_start = processed_examples
        candidate_samples: list[EmbeddingInput] = []
        candidate_metadata: list[tuple[int, int, str, str | None, ImageLike]] = []

        for offset, (conversation, images) in enumerate(
            zip(batch["texts"], batch["images"], strict=False)
        ):
            source_index = batch_start + offset
            text = _extract_cauldron_text(conversation)

            if not images:
                if text:
                    safe_chunks = _ensure_teacher_safe(text)
                    split_rows += max(0, len(safe_chunks) - 1)
                    for chunk in safe_chunks:
                        _append_cauldron_row(
                            rows,
                            text=chunk,
                            modality="text",
                            source_index=source_index,
                            image_index=-1,
                        )
                continue

            for image_index, image in enumerate(images):
                if not has_usable_image(image):
                    skipped_images += 1
                    continue

                modality = "text_image" if text else "image"
                candidate_samples.append(EmbeddingInput(text=text, image=image))
                candidate_metadata.append(
                    (source_index, image_index, modality, text, image)
                )

        if candidate_samples:
            processed_lengths = _batched_processed_lengths(candidate_samples)
            student_fit_flags = student_batch_fit_flags(
                student_processor,
                candidate_samples,
                max_length=student_max_length,
            )
            for (
                source_index,
                image_index,
                modality,
                text,
                image,
            ), processed_tokens, student_fits in zip(
                candidate_metadata,
                processed_lengths,
                student_fit_flags,
                strict=False,
            ):
                if (
                    processed_tokens <= teacher_prompt_token_limit()
                    and student_fits
                ):
                    _append_cauldron_row(
                        rows,
                        text=text,
                        modality=modality,
                        source_index=source_index,
                        image_index=image_index,
                    )
                    continue

                if modality != "text_image" or not text:
                    skipped_overlength += 1
                    if not student_fits:
                        skipped_student_overlength += 1
                    logger.warning(
                        "Skipping overlength cauldron/%s source=%d image=%d processed_tokens=%d student_fits=%s max_length=%d",
                        config,
                        source_index,
                        image_index,
                        processed_tokens,
                        student_fits,
                        student_max_length,
                    )
                    continue

                safe_chunks = _ensure_multimodal_joint_safe(
                    text,
                    image,
                    processor=student_processor,
                    max_length=student_max_length,
                )
                if not safe_chunks:
                    skipped_overlength += 1
                    if not student_fits:
                        skipped_student_overlength += 1
                    logger.warning(
                        "Skipping unsplittable cauldron/%s source=%d image=%d processed_tokens=%d student_fits=%s max_length=%d",
                        config,
                        source_index,
                        image_index,
                        processed_tokens,
                        student_fits,
                        student_max_length,
                    )
                    continue
                split_rows += max(0, len(safe_chunks) - 1)
                for chunk in safe_chunks:
                    _append_cauldron_row(
                        rows,
                        text=chunk,
                        modality=modality,
                        source_index=source_index,
                        image_index=image_index,
                    )

        processed_examples += len(batch["texts"])
        emitted_rows += len(rows["text"])
        progress.update(
            processed_examples,
            extra=(
                f"rows={emitted_rows:,}, split_rows={split_rows:,}, "
                f"skipped_images={skipped_images:,}, skipped_overlength={skipped_overlength:,}, "
                f"skipped_student_overlength={skipped_student_overlength:,}"
            ),
        )
        return rows

    ds = raw.map(
        _normalize_cauldron_batch,
        batched=True,
        batch_size=32,
        remove_columns=raw.column_names,
    )
    progress.close(
        extra=(
            f"rows={emitted_rows:,}, split_rows={split_rows:,}, "
            f"skipped_images={skipped_images:,}, skipped_overlength={skipped_overlength:,}, "
            f"skipped_student_overlength={skipped_student_overlength:,}"
        )
    )
    _save_processed_dataset(ds, out_path, empty_rows=empty_rows)
    logger.info(
        "Processed cauldron/%s: rows=%d split_rows=%d skipped_images=%d skipped_overlength=%d skipped_student_overlength=%d",
        config,
        len(ds),
        split_rows,
        skipped_images,
        skipped_overlength,
        skipped_student_overlength,
    )
    return ds


def preprocess_cauldron(
    *,
    limit: int | None = None,
    student_max_length: int = 8192,
) -> list[Dataset]:
    """Process all Cauldron configs, returning a list of Arrow datasets."""
    return [
        preprocess_cauldron_config(
            config,
            limit=limit,
            student_max_length=student_max_length,
        )
        for config in CAULDRON_CONFIGS
    ]


def preprocess_data(
    download_first: bool = False,
    *,
    limit: int | None = None,
    student_max_length: int = 8192,
) -> tuple[Dataset, list[Dataset]]:
    """Entry point: returns (wikipedia_dataset, cauldron_datasets).

    Each is memory-mapped from disk — no RAM blow-up.
    Pass limit=N to only process the first N raw entries (for testing).
    """
    if download_first:
        download_data(limit=limit)

    wikipedia = preprocess_wikipedia(limit=limit)
    cauldron = preprocess_cauldron(
        limit=limit,
        student_max_length=student_max_length,
    )
    return wikipedia, cauldron
