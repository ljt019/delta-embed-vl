from __future__ import annotations

import io
import json
import logging
import re
import shutil
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, cast

from datasets import Dataset
from PIL import Image
from transformers import Qwen3VLProcessor

from delta_embed_vl.artifacts import versioned_name
from delta_embed_vl.data.download import (
    CAULDRON_CONFIGS,
    download_data,
    load_raw_cauldron,
    load_raw_wikipedia,
)
from delta_embed_vl.model.embedding_inputs import (
    EmbeddingInput,
    count_teacher_processed_tokens,
    count_teacher_prompt_tokens,
    get_student_processor,
    sample_fits_student,
    teacher_prompt_token_limit,
)
from delta_embed_vl.settings import Settings

logger = logging.getLogger(__name__)

_PREPARED_DIR = Settings().data_dir / "prepared"
_EMPTY_DATASET_MARKER = "_empty_dataset.json"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_IMAGE_TOKEN = re.compile(r"<image>\s*")
_MIN_CHUNK_CHARS = 100
_MAX_CAULDRON_TURNS_PER_ROW = 4
_GLOBAL_IMAGE_LONGEST_EDGE_CAP: int | None = 1024

_PREPARED_COLUMNS = {
    "source": [],
    "role": [],
    "text": [],
    "source_row": [],
    "image_index": [],
}

ImageLike = Image.Image | dict[str, Any] | str | Path | None


class CauldronTurn(dict[str, str]):
    pass


def _image_cap_cache_suffix() -> str:
    if _GLOBAL_IMAGE_LONGEST_EDGE_CAP is None:
        return "_imgcap-original"
    return f"_imgcap{_GLOBAL_IMAGE_LONGEST_EDGE_CAP}"


def _resize_image_to_max_longest_edge(
    image: Image.Image,
    *,
    max_longest_edge: int,
) -> Image.Image:
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= max_longest_edge:
        return image

    scale = max_longest_edge / longest_edge
    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    return image.resize(resized_size, Image.Resampling.BICUBIC)


def _apply_image_cap(image: Image.Image) -> Image.Image:
    if _GLOBAL_IMAGE_LONGEST_EDGE_CAP is None:
        return image
    return _resize_image_to_max_longest_edge(
        image,
        max_longest_edge=_GLOBAL_IMAGE_LONGEST_EDGE_CAP,
    )


def resolve_image_path(image_path: str | Path) -> Path | None:
    path = Path(image_path)
    if path.exists():
        return path

    normalized_path = str(image_path).replace("\\", "/")
    resolved = _resolve_cached_image_path(normalized_path)
    if resolved is None:
        return None
    return Path(resolved)


@lru_cache(maxsize=65536)
def _resolve_cached_image_path(normalized_path: str) -> str | None:
    marker = "/downloads/extracted/"
    suffixes: list[str] = []

    if marker in normalized_path:
        suffixes.append(normalized_path.split(marker, maxsplit=1)[1])

    stripped = normalized_path.lstrip("/")
    if stripped:
        suffixes.append(stripped)

    parts = [part for part in PurePosixPath(stripped or normalized_path).parts if part]
    for width in range(min(len(parts), 6), 0, -1):
        suffixes.append("/".join(parts[-width:]))

    seen: set[str] = set()
    for suffix in suffixes:
        if suffix in seen:
            continue
        seen.add(suffix)
        resolved = next((Settings().data_dir / "raw").glob(f"**/{suffix}"), None)
        if resolved is not None and resolved.exists():
            return str(resolved)

    return None


def coerce_image_to_rgb(image: ImageLike) -> Image.Image | None:
    if image is None:
        return None

    if isinstance(image, Image.Image):
        return _apply_image_cap(image.convert("RGB"))

    if isinstance(image, (str, Path)):
        resolved_path = resolve_image_path(image)
        if resolved_path is None:
            return None
        with Image.open(resolved_path) as loaded:
            return _apply_image_cap(loaded.convert("RGB"))

    image_bytes = image.get("bytes")
    if image_bytes is not None:
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        elif isinstance(image_bytes, bytearray):
            image_bytes = bytes(image_bytes)
        elif isinstance(image_bytes, list):
            image_bytes = bytes(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as loaded:
            return _apply_image_cap(loaded.convert("RGB"))

    image_path = image.get("path")
    if image_path:
        resolved_path = resolve_image_path(image_path)
        if resolved_path is None:
            return None
        with Image.open(resolved_path) as loaded:
            return _apply_image_cap(loaded.convert("RGB"))

    return None


def _split_oversized_span(text: str, *, max_chars: int) -> list[str]:
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
    image: Image.Image,
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
    text = " ".join(text.split()).strip()
    if not text:
        return []

    sentences = [sentence for sentence in _SENTENCE_SPLIT.split(text) if sentence]
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


def _normalize_cauldron_text(text: str) -> str | None:
    normalized = _IMAGE_TOKEN.sub("", text).strip()
    return normalized or None


def _append_prepared_row(
    rows: dict[str, list[Any]],
    *,
    source: str,
    role: str,
    text: str | None,
    source_row: int,
    image_index: int,
) -> None:
    rows["source"].append(source)
    rows["role"].append(role)
    rows["text"].append(text)
    rows["source_row"].append(source_row)
    rows["image_index"].append(image_index)


def _empty_rows() -> dict[str, list[Any]]:
    return {key: [] for key in _PREPARED_COLUMNS}


def _already_prepared(path: Path) -> bool:
    return path.exists() and (
        (path / "dataset_info.json").exists() or (path / _EMPTY_DATASET_MARKER).exists()
    )


def _load_prepared_dataset(path: Path) -> Dataset:
    if (path / _EMPTY_DATASET_MARKER).exists():
        return Dataset.from_dict(_empty_rows())
    return Dataset.load_from_disk(str(path))


def _save_prepared_dataset(dataset: Dataset, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)

    if len(dataset) == 0:
        path.mkdir(parents=True, exist_ok=True)
        (path / _EMPTY_DATASET_MARKER).write_text(
            json.dumps({"columns": list(_PREPARED_COLUMNS.keys())}),
            encoding="utf-8",
        )
        return

    dataset.save_to_disk(str(path))


def _prepare_wikipedia_batch(
    batch: dict[str, list[Any]],
    indices: list[int],
) -> dict[str, list[Any]]:
    rows = _empty_rows()
    for source_row, text in zip(indices, batch["text"], strict=False):
        if not isinstance(text, str):
            continue

        for chunk in _chunk_text(text):
            _append_prepared_row(
                rows,
                source="wikipedia",
                role="document",
                text=chunk,
                source_row=source_row,
                image_index=-1,
            )
    return rows


def _validate_cauldron_turns(
    *,
    source: str,
    source_row: int,
    texts: object,
    images: object,
) -> tuple[list[CauldronTurn], list[Any]]:
    if not isinstance(texts, list):
        raise ValueError(f"{source} row={source_row}: expected texts to be a list.")
    if not isinstance(images, list):
        raise ValueError(f"{source} row={source_row}: expected images to be a list.")

    validated_turns: list[CauldronTurn] = []
    for turn_idx, turn in enumerate(texts[:_MAX_CAULDRON_TURNS_PER_ROW]):
        if not isinstance(turn, dict):
            raise ValueError(
                f"{source} row={source_row} turn={turn_idx}: expected dict turn."
            )
        turn_dict = cast(dict[str, object], turn)
        if "user" not in turn_dict or "assistant" not in turn_dict:
            raise ValueError(
                f"{source} row={source_row} turn={turn_idx}: missing user/assistant."
            )
        user = turn_dict["user"]
        assistant = turn_dict["assistant"]
        if not isinstance(user, str) or not isinstance(assistant, str):
            raise ValueError(
                f"{source} row={source_row} turn={turn_idx}: expected string user/assistant."
            )
        validated_turns.append(CauldronTurn(user=user, assistant=assistant))

    return validated_turns, images


def _prepare_cauldron_batch(
    batch: dict[str, list[Any]],
    indices: list[int],
    *,
    config: str,
    processor: Qwen3VLProcessor,
    student_max_length: int,
) -> dict[str, list[Any]]:
    rows = _empty_rows()
    source = f"cauldron/{config}"

    for source_row, texts, images in zip(
        indices,
        batch["texts"],
        batch["images"],
        strict=False,
    ):
        turns, raw_images = _validate_cauldron_turns(
            source=source,
            source_row=source_row,
            texts=texts,
            images=images,
        )

        resolved_images: list[tuple[int, Image.Image]] = []
        for image_index, image in enumerate(raw_images):
            resolved_image = coerce_image_to_rgb(image)
            if resolved_image is None:
                continue
            resolved_images.append((image_index, resolved_image))

        if not resolved_images:
            continue

        emitted_any = False
        for turn in turns:
            query_text = _normalize_cauldron_text(turn["user"])
            document_text = _normalize_cauldron_text(turn["assistant"])

            for image_index, image in resolved_images:
                if query_text is not None:
                    query_chunks = _ensure_multimodal_joint_safe(
                        query_text,
                        image,
                        processor=processor,
                        max_length=student_max_length,
                    )
                    for chunk in query_chunks:
                        _append_prepared_row(
                            rows,
                            source=source,
                            role="query",
                            text=chunk,
                            source_row=source_row,
                            image_index=image_index,
                        )
                        emitted_any = True

                if document_text is not None:
                    document_chunks = _ensure_multimodal_joint_safe(
                        document_text,
                        image,
                        processor=processor,
                        max_length=student_max_length,
                    )
                    for chunk in document_chunks:
                        _append_prepared_row(
                            rows,
                            source=source,
                            role="document",
                            text=chunk,
                            source_row=source_row,
                            image_index=image_index,
                        )
                        emitted_any = True

        if emitted_any:
            continue

        for image_index, _ in resolved_images:
            _append_prepared_row(
                rows,
                source=source,
                role="document",
                text=None,
                source_row=source_row,
                image_index=image_index,
            )

    return rows


def load_prepared_cauldron_embedding_input(
    raw_dataset: Dataset,
    row: dict[str, Any],
    *,
    config: str | None = None,
    row_index: int | None = None,
) -> EmbeddingInput:
    text = row.get("text") or None
    image_index = int(row["image_index"])
    if image_index < 0:
        return EmbeddingInput(text=text)

    source_row = int(row["source_row"])
    source_images = raw_dataset[source_row]["images"]
    if image_index >= len(source_images):
        location = f"source_row={source_row} image={image_index}"
        if config is not None:
            location = f"cauldron/{config} {location}"
        if row_index is not None:
            location = f"{location} row={row_index}"
        raise ValueError(f"Image index out of range for {location}.")

    image = coerce_image_to_rgb(source_images[image_index])
    if image is None:
        location = f"source_row={source_row} image={image_index}"
        if config is not None:
            location = f"cauldron/{config} {location}"
        if row_index is not None:
            location = f"{location} row={row_index}"
        raise ValueError(f"Could not resolve image for {location}.")

    return EmbeddingInput(text=text, image=image)


def prepare_dataset(
    source: str,
    raw_dataset: Dataset,
    *,
    student_max_length: int = 8192,
) -> Dataset:
    if source == "wikipedia":
        return raw_dataset.map(
            _prepare_wikipedia_batch,
            batched=True,
            with_indices=True,
            batch_size=256,
            remove_columns=raw_dataset.column_names,
        )

    if source.startswith("cauldron/"):
        config = source.split("/", maxsplit=1)[1]
        processor = get_student_processor()
        return raw_dataset.map(
            lambda batch, indices: _prepare_cauldron_batch(
                batch,
                indices,
                config=config,
                processor=processor,
                student_max_length=student_max_length,
            ),
            batched=True,
            with_indices=True,
            batch_size=32,
            remove_columns=raw_dataset.column_names,
        )

    raise ValueError(f"Unsupported source: {source}")


def prepare_wikipedia(*, limit: int | None = None) -> Dataset:
    out_path = _PREPARED_DIR / versioned_name("wikipedia", limit=limit)
    if _already_prepared(out_path):
        return _load_prepared_dataset(out_path)

    raw = load_raw_wikipedia(limit=limit)
    dataset = prepare_dataset("wikipedia", raw)
    _save_prepared_dataset(dataset, out_path)
    logger.info("Prepared wikipedia rows=%d", len(dataset))
    return dataset


def prepare_cauldron_config(
    config: str,
    *,
    limit: int | None = None,
    student_max_length: int = 8192,
) -> Dataset:
    out_path = (
        _PREPARED_DIR
        / "cauldron"
        # Prepared rows depend on the current image cap because fit checks and
        # chunking run against the resized image seen by the student path.
        / (
            f"{versioned_name(config, limit=limit)}"
            f"_student{student_max_length}"
            f"{_image_cap_cache_suffix()}"
        )
    )
    if _already_prepared(out_path):
        return _load_prepared_dataset(out_path)

    raw = load_raw_cauldron(config, limit=limit)
    dataset = prepare_dataset(
        f"cauldron/{config}",
        raw,
        student_max_length=student_max_length,
    )
    _save_prepared_dataset(dataset, out_path)
    logger.info("Prepared cauldron/%s rows=%d", config, len(dataset))
    return dataset


def prepare_cauldron(
    *,
    limit: int | None = None,
    student_max_length: int = 8192,
) -> list[Dataset]:
    return [
        prepare_cauldron_config(
            config,
            limit=limit,
            student_max_length=student_max_length,
        )
        for config in CAULDRON_CONFIGS
    ]


def prepare_data(
    download_first: bool = False,
    *,
    limit: int | None = None,
    student_max_length: int = 8192,
) -> tuple[Dataset, list[Dataset]]:
    if download_first:
        download_data(limit=limit)

    wikipedia = prepare_wikipedia(limit=limit)
    cauldron = prepare_cauldron(
        limit=limit,
        student_max_length=student_max_length,
    )
    return wikipedia, cauldron
