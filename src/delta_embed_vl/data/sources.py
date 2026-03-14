from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from PIL import Image

from delta_embed_vl import cfg
from delta_embed_vl.data.download import (
    CAULDRON_CONFIGS,
    load_raw_cauldron,
    load_raw_wikipedia,
)
from delta_embed_vl.model.tokenization import (
    DEFAULT_EMBED_INSTRUCTION,
    EmbeddingInput,
    get_student_processor,
    student_batch_fit_flags,
)

logger = logging.getLogger(__name__)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_IMAGE_TOKEN = re.compile(r"<image>\s*")
_MIN_CHUNK_CHARS = 100
_MAX_CAULDRON_TURNS_PER_ROW = 4
_GLOBAL_IMAGE_LONGEST_EDGE_CAP: int | None = 1024
_CAULDRON_STUDENT_BATCH_SIZE = 8
_WIKIPEDIA_STUDENT_BATCH_SIZE = 32

ImageLike = Image.Image | dict[str, Any] | str | Path | None


class CauldronTurn(dict[str, str]):
    pass


@dataclass(frozen=True)
class NormalizedSample:
    source: str
    role: Literal["query", "document"]
    text: str | None
    image: Image.Image | None
    instruction: str = DEFAULT_EMBED_INSTRUCTION

    def to_embedding_input(self) -> EmbeddingInput:
        return EmbeddingInput(
            text=self.text,
            image=self.image,
            instruction=self.instruction,
        )


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


@lru_cache(maxsize=8192)
def _resolve_image_path_cached(image_path_str: str) -> str | None:
    path = Path(image_path_str)
    if path.exists():
        return str(path)

    normalized_path = image_path_str.replace("\\", "/")
    resolved = _resolve_cached_image_path(normalized_path)
    if resolved is None:
        return None
    return resolved


def resolve_image_path(image_path: str | Path) -> Path | None:
    resolved = _resolve_image_path_cached(str(image_path))
    if resolved is None:
        return None
    return Path(resolved)


@lru_cache(maxsize=8192)
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
        resolved = next((Path("data") / "raw").glob(f"**/{suffix}"), None)
        if resolved is not None and resolved.exists():
            return str(resolved)

    return None


def _coerce_opened_image_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        image.load()
        return _apply_image_cap(image)
    return _apply_image_cap(image.convert("RGB"))


def coerce_image_to_rgb(image: ImageLike) -> Image.Image | None:
    if image is None:
        return None

    if isinstance(image, Image.Image):
        return _coerce_opened_image_to_rgb(image)

    if isinstance(image, (str, Path)):
        resolved_path = resolve_image_path(image)
        if resolved_path is None:
            return None
        with Image.open(resolved_path) as loaded:
            return _coerce_opened_image_to_rgb(loaded)

    image_bytes = image.get("bytes")
    if image_bytes is not None:
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        elif isinstance(image_bytes, bytearray):
            image_bytes = bytes(image_bytes)
        elif isinstance(image_bytes, list):
            image_bytes = bytes(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as loaded:
            return _coerce_opened_image_to_rgb(loaded)

    image_path = image.get("path")
    if image_path:
        resolved_path = resolve_image_path(image_path)
        if resolved_path is None:
            return None
        with Image.open(resolved_path) as loaded:
            return _coerce_opened_image_to_rgb(loaded)

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


def _chunk_text(text: str, *, chunk_chars: int = 4096) -> list[str]:
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

    return chunks


def _normalize_cauldron_text(text: str) -> str | None:
    normalized = _IMAGE_TOKEN.sub("", text).strip()
    return normalized or None


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


def _yield_fitting_samples(
    candidates: list[NormalizedSample],
    *,
    student_max_length: int,
    batch_size: int,
) -> Iterator[NormalizedSample]:
    if not candidates:
        return

    processor = get_student_processor()
    fits = student_batch_fit_flags(
        processor,
        [
            EmbeddingInput(
                text=sample.text,
                image=sample.image,
                instruction=sample.instruction,
            )
            for sample in candidates
        ],
        max_length=student_max_length,
        batch_size=batch_size,
    )
    for sample, fits_student in zip(candidates, fits, strict=True):
        if fits_student:
            yield sample


def wikipedia_samples(
    *,
    limit: int | None = None,
    student_max_length: int = cfg["max_length"],
    max_output_samples: int | None = None,
    no_stream: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Iterator[NormalizedSample]:
    raw_dataset = load_raw_wikipedia(limit=limit, no_stream=no_stream)
    if num_shards > 1:
        raw_dataset = raw_dataset.shard(
            num_shards=num_shards,
            index=shard_index,
            contiguous=True,
        )
    buffer: list[NormalizedSample] = []
    emitted = 0

    def flush_buffer() -> Iterator[NormalizedSample]:
        nonlocal buffer, emitted
        if not buffer:
            return
        for sample in _yield_fitting_samples(
            buffer,
            student_max_length=student_max_length,
            batch_size=_WIKIPEDIA_STUDENT_BATCH_SIZE,
        ):
            yield sample
            emitted += 1
            if max_output_samples is not None and emitted >= max_output_samples:
                return
        buffer = []

    for row in raw_dataset:
        text = row.get("text")
        if not isinstance(text, str):
            continue
        for chunk in _chunk_text(text):
            buffer.append(
                NormalizedSample(
                    source="wikipedia",
                    role="document",
                    text=chunk,
                    image=None,
                )
            )
            if len(buffer) < 256:
                continue
            yield from flush_buffer()
            if max_output_samples is not None and emitted >= max_output_samples:
                return

    yield from flush_buffer()


def cauldron_config_samples(
    config: str,
    *,
    limit: int | None = None,
    student_max_length: int = cfg["max_length"],
    no_stream: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Iterator[NormalizedSample]:
    source = f"cauldron/{config}"
    raw_dataset = load_raw_cauldron(config, limit=limit, no_stream=no_stream)
    if num_shards > 1:
        raw_dataset = raw_dataset.shard(
            num_shards=num_shards,
            index=shard_index,
            contiguous=True,
        )

    for source_row, raw_row in enumerate(raw_dataset):
        try:
            yield from _process_cauldron_row(
                raw_row,
                source=source,
                source_row=source_row,
                student_max_length=student_max_length,
            )
        except Exception:
            logger.debug("Skipping %s row=%d: %s", source, source_row, _exc_oneliner())
            continue


def _exc_oneliner() -> str:
    import sys

    exc = sys.exc_info()[1]
    if exc is None:
        return "unknown error"
    msg = str(exc).replace("\n", " ").strip()
    return f"{type(exc).__name__}: {msg[:200]}"


def _process_cauldron_row(
    raw_row: Any,
    *,
    source: str,
    source_row: int,
    student_max_length: int,
) -> Iterator[NormalizedSample]:
    turns, raw_images = _validate_cauldron_turns(
        source=source,
        source_row=source_row,
        texts=raw_row["texts"],
        images=raw_row["images"],
    )

    resolved_images: list[Image.Image] = []
    for image in raw_images:
        resolved_image = coerce_image_to_rgb(image)
        if resolved_image is not None:
            resolved_images.append(resolved_image)
    if not resolved_images:
        return

    candidates: list[NormalizedSample] = []
    for turn in turns:
        query_text = _normalize_cauldron_text(turn["user"])
        document_text = _normalize_cauldron_text(turn["assistant"])
        for image in resolved_images:
            if query_text is not None:
                candidates.append(
                    NormalizedSample(
                        source=source,
                        role="query",
                        text=query_text,
                        image=image,
                    )
                )
            if document_text is not None:
                candidates.append(
                    NormalizedSample(
                        source=source,
                        role="document",
                        text=document_text,
                        image=image,
                    )
                )

    emitted_any = False
    for sample in _yield_fitting_samples(
        candidates,
        student_max_length=student_max_length,
        batch_size=_CAULDRON_STUDENT_BATCH_SIZE,
    ):
        emitted_any = True
        yield sample

    if emitted_any:
        return

    for image in resolved_images:
        yield NormalizedSample(
            source=source,
            role="document",
            text=None,
            image=image,
        )


def normalization_source_names() -> list[str]:
    return ["wikipedia", *[f"cauldron/{config}" for config in CAULDRON_CONFIGS]]


def iter_source_samples(
    source: str,
    *,
    limit: int | None = None,
    student_max_length: int = cfg["max_length"],
    no_stream: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Iterator[NormalizedSample]:
    if source == "wikipedia":
        yield from wikipedia_samples(
            limit=limit,
            student_max_length=student_max_length,
            no_stream=no_stream,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        return

    if source.startswith("cauldron/"):
        yield from cauldron_config_samples(
            source.removeprefix("cauldron/"),
            limit=limit,
            student_max_length=student_max_length,
            no_stream=no_stream,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        return

    supported_sources = ", ".join(normalization_source_names())
    raise ValueError(
        f"Unknown normalization source: {source}. Supported: {supported_sources}"
    )
