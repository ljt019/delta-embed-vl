from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import httpx
import numpy as np
from datasets import Dataset
from numpy.lib.format import open_memmap
from PIL import Image

from delta_embed_vl.artifacts import versioned_name
from delta_embed_vl.data.download import CAULDRON_CONFIGS, load_raw_cauldron
from delta_embed_vl.data.media import (
    coerce_image_to_rgb,
    has_usable_image,
    image_to_data_uri,
)
from delta_embed_vl.data.preprocess import (
    _extract_cauldron_text,
    preprocess_cauldron_config,
    preprocess_wikipedia,
)
from delta_embed_vl.model.embedding_inputs import (
    DEFAULT_EMBED_INSTRUCTION,
    EmbeddingInput,
)
from delta_embed_vl.progress import ProgressLogger
from delta_embed_vl.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

_EMBEDDINGS_DIR = settings.data_dir / "embeddings"
_BATCH_SIZE = 256


def get_embedding(
    text: str | None = None,
    image: Image.Image | None = None,
) -> list[float]:
    """Get a single teacher embedding. Useful for testing / ad-hoc queries."""
    payload = _build_payload(EmbeddingInput(text=text, image=image))
    resp = httpx.post(
        f"{settings.teacher_base_url}/embeddings",
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def embed_wikipedia(*, limit: int | None = None) -> np.ndarray:
    """Generate teacher embeddings for all Wikipedia chunks."""
    ds = preprocess_wikipedia(limit=limit)
    out_name = versioned_name("wikipedia", limit=limit)
    return asyncio.run(
        _embed_inputs(
            _iter_dataset_inputs(ds),
            total_rows=len(ds),
            out_path=_EMBEDDINGS_DIR / f"{out_name}.npy",
            label="wikipedia",
        )
    )


def embed_cauldron_config(config: str, *, limit: int | None = None) -> np.ndarray:
    """Generate teacher embeddings for one Cauldron config."""
    ds = preprocess_cauldron_config(config, limit=limit)
    raw = load_raw_cauldron(config, limit=limit)
    out_name = versioned_name(config, limit=limit)
    return asyncio.run(
        _embed_inputs(
            _iter_cauldron_inputs(raw),
            total_rows=len(ds),
            out_path=_EMBEDDINGS_DIR / "cauldron" / f"{out_name}.npy",
            label=f"cauldron/{config}",
        )
    )


def embed_all(*, limit: int | None = None) -> None:
    """Generate teacher embeddings for all datasets."""
    embed_wikipedia(limit=limit)
    for config in CAULDRON_CONFIGS:
        embed_cauldron_config(config, limit=limit)


def _progress_interval(total: int) -> int:
    return max(1_000, (total + 19) // 20)


def _load_cached_embeddings(out_path: Path, *, expected_rows: int) -> np.ndarray | None:
    if not out_path.exists():
        return None

    try:
        cached = np.load(str(out_path), mmap_mode="r")
    except Exception:
        logger.warning("Removing corrupt embedding cache at %s", out_path)
        out_path.unlink(missing_ok=True)
        return None

    if cached.shape[0] != expected_rows:
        logger.warning(
            "Removing stale embedding cache at %s (expected %d rows, got %d)",
            out_path,
            expected_rows,
            cached.shape[0],
        )
        out_path.unlink(missing_ok=True)
        return None

    return cached


def _build_payload(sample: EmbeddingInput) -> dict[str, object]:
    """Build a vLLM chat-embedding request body for Qwen3-VL."""
    user_content: list[dict[str, object]] = []
    resolved_image = coerce_image_to_rgb(sample.image)
    if sample.image is not None and resolved_image is None:
        raise ValueError("Could not resolve image for teacher embedding request.")

    if resolved_image is not None:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_uri(resolved_image)},
            }
        )

    user_content.append({"type": "text", "text": sample.text or ""})

    return {
        "model": settings.teacher_model,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": sample.instruction or DEFAULT_EMBED_INSTRUCTION,
                    }
                ],
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ],
        "encoding_format": "float",
        "continue_final_message": True,
        "add_special_tokens": True,
    }


def _iter_dataset_inputs(dataset: Dataset) -> Iterator[EmbeddingInput]:
    for row in dataset:
        yield EmbeddingInput(text=row.get("text") or None)


def _iter_cauldron_inputs(raw: Dataset) -> Iterator[EmbeddingInput]:
    for example in raw:
        text = _extract_cauldron_text(example["texts"])
        images = example["images"]

        if not images:
            if text:
                yield EmbeddingInput(text=text)
            continue

        for image in images:
            if not has_usable_image(image):
                continue
            yield EmbeddingInput(text=text, image=image)


async def _request_embedding(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    sample: EmbeddingInput,
) -> list[float]:
    async with semaphore:
        resp = await client.post(
            f"{settings.teacher_base_url}/embeddings",
            json=_build_payload(sample),
        )
        if resp.is_error:
            response_body = resp.text[:500]
            raise RuntimeError(
                f"Teacher embedding request failed with {resp.status_code}: {response_body}"
            )
        return resp.json()["data"][0]["embedding"]


async def _write_input_batch(
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    samples: list[EmbeddingInput],
    embedding_memmap: np.memmap | None,
    start_index: int,
    total_rows: int,
    tmp_path: Path,
) -> tuple[np.memmap, int]:
    results = await asyncio.gather(
        *[_request_embedding(client, semaphore, sample) for sample in samples]
    )
    batch_array = np.asarray(results, dtype=np.float32)
    if embedding_memmap is None:
        embedding_memmap = open_memmap(
            str(tmp_path),
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, batch_array.shape[1]),
        )
    end_index = start_index + len(batch_array)
    embedding_memmap[start_index:end_index] = batch_array
    embedding_memmap.flush()
    return embedding_memmap, end_index


async def _embed_inputs(
    input_iter: Iterable[EmbeddingInput],
    *,
    total_rows: int,
    out_path: Path,
    label: str,
) -> np.ndarray:
    cached = _load_cached_embeddings(out_path, expected_rows=total_rows)
    if cached is not None:
        logger.info("%s: cached shape=%s", label, cached.shape)
        return cached

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("%s: embedding %d samples", label, total_rows)

    if total_rows == 0:
        arr = np.zeros((0, 0), dtype=np.float32)
        np.save(str(out_path), arr)
        logger.info("%s: done shape=%s", label, arr.shape)
        return arr

    embedding_memmap: np.memmap | None = None
    tmp_path = out_path.with_suffix(f"{out_path.suffix}.tmp")
    tmp_path.unlink(missing_ok=True)
    semaphore = asyncio.Semaphore(settings.teacher_concurrency)
    next_index = 0
    progress = ProgressLogger(
        logger=logger,
        label=label,
        total=total_rows,
        unit="samples",
        every_items=_progress_interval(total_rows),
    )
    batch_samples: list[EmbeddingInput] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        for sample in input_iter:
            batch_samples.append(sample)
            if len(batch_samples) < _BATCH_SIZE:
                continue
            embedding_memmap, next_index = await _write_input_batch(
                client=client,
                semaphore=semaphore,
                samples=batch_samples,
                embedding_memmap=embedding_memmap,
                start_index=next_index,
                total_rows=total_rows,
                tmp_path=tmp_path,
            )
            batch_samples = []
            progress.maybe_log(next_index)

        if batch_samples:
            embedding_memmap, next_index = await _write_input_batch(
                client=client,
                semaphore=semaphore,
                samples=batch_samples,
                embedding_memmap=embedding_memmap,
                start_index=next_index,
                total_rows=total_rows,
                tmp_path=tmp_path,
            )
            progress.maybe_log(next_index, force=True)

    if embedding_memmap is None or next_index != total_rows:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Expected to write {total_rows} embeddings to {out_path}, wrote {next_index}"
        )

    tmp_path.replace(out_path)
    logger.info("%s: done shape=%s", label, embedding_memmap.shape)
    return np.load(str(out_path), mmap_mode="r")
