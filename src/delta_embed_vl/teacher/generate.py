import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from datasets import Dataset
from numpy.lib.format import open_memmap
from PIL import Image

from delta_embed_vl.data.download import CAULDRON_CONFIGS, load_raw_cauldron
from delta_embed_vl.data.preprocess import (
    _extract_cauldron_text,
    preprocess_cauldron_config,
    preprocess_wikipedia,
)
from delta_embed_vl.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


### Public


def get_embedding(
    text: str | None = None,
    image: Image.Image | None = None,
) -> list[float]:
    """Get a single teacher embedding. Useful for testing / ad-hoc queries."""
    payload = _build_payload(text, image)
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
    suffix = f"_test{limit}" if limit else ""
    return asyncio.run(_embed_dataset(ds, _EMBEDDINGS_DIR / f"wikipedia{suffix}.npy"))


def embed_cauldron_config(config: str, *, limit: int | None = None) -> np.ndarray:
    """Generate teacher embeddings for one Cauldron config."""
    ds = preprocess_cauldron_config(config, limit=limit)
    raw = load_raw_cauldron(config, limit=limit)
    suffix = f"_test{limit}" if limit else ""
    return asyncio.run(
        _embed_payloads(
            _iter_cauldron_payloads(raw),
            total_rows=len(ds),
            out_path=_EMBEDDINGS_DIR / "cauldron" / f"{config}{suffix}.npy",
        )
    )


def embed_all(*, limit: int | None = None) -> None:
    """Generate teacher embeddings for all datasets."""
    embed_wikipedia(limit=limit)
    for config in CAULDRON_CONFIGS:
        logger.info("Embedding cauldron/%s", config)
        embed_cauldron_config(config, limit=limit)


### Private


_EMBEDDINGS_DIR = settings.data_dir / "embeddings"
_RAW_DATA_DIR = settings.data_dir / "raw"
_INSTRUCTION = "Represent the user's input."
_BATCH_SIZE = 256


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


def _resolve_image_path(image_path: str | Path) -> Path | None:
    path = Path(image_path)
    if path.exists():
        return path

    marker = "/downloads/extracted/"
    normalized_path = image_path.replace("\\", "/")
    if marker in normalized_path:
        suffix = normalized_path.split(marker, maxsplit=1)[1]
        matches = _RAW_DATA_DIR.glob(f"**/downloads/extracted/{suffix}")
        resolved = next(matches, None)
        if resolved is not None and resolved.exists():
            return resolved

    return None


def _coerce_image(
    image: Image.Image | dict[str, Any] | str | Path | None,
) -> Image.Image | None:
    if image is None:
        return None

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        resolved_path = _resolve_image_path(image)
        if resolved_path is None:
            return None
        with Image.open(resolved_path) as loaded:
            return loaded.convert("RGB")

    image_bytes = image.get("bytes")
    if image_bytes is not None:
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        elif isinstance(image_bytes, bytearray):
            image_bytes = bytes(image_bytes)
        elif isinstance(image_bytes, list):
            image_bytes = bytes(image_bytes)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_path = image.get("path")
    if image_path:
        resolved_path = _resolve_image_path(image_path)
        if resolved_path is None:
            return None
        with Image.open(resolved_path) as loaded:
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
        return _resolve_image_path(image) is not None

    if image.get("bytes") is not None:
        return True

    image_path = image.get("path")
    return bool(image_path) and _resolve_image_path(image_path) is not None


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _build_payload(
    text: str | None, image: Image.Image | dict[str, Any] | str | Path | None
) -> dict:
    """Build a vLLM chat-embedding request body for Qwen3-VL."""
    user_content: list[dict] = []
    resolved_image = _coerce_image(image)

    if resolved_image is not None:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_uri(resolved_image)},
            }
        )

    user_content.append({"type": "text", "text": text or ""})

    return {
        "model": settings.teacher_model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": _INSTRUCTION}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ],
        "encoding_format": "float",
        "continue_final_message": True,
        "add_special_tokens": True,
    }


def _iter_cauldron_payloads(raw: Dataset):
    for example in raw:
        text = _extract_cauldron_text(example["texts"])
        images = example["images"]

        if not images:
            if text:
                yield _build_payload(text, None)
            continue

        for image in images:
            if not _has_usable_image(image):
                continue
            yield _build_payload(text, image)


async def _request_embedding(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    payload: dict,
) -> list[float]:
    async with semaphore:
        resp = await client.post(
            f"{settings.teacher_base_url}/embeddings",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


async def _write_payload_batch(
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    payloads: list[dict],
    embedding_memmap: np.memmap | None,
    start_index: int,
    total_rows: int,
    tmp_path: Path,
) -> tuple[np.memmap, int]:
    results = await asyncio.gather(
        *[_request_embedding(client, semaphore, payload) for payload in payloads]
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
    logger.info("  %d / %d", end_index, total_rows)
    return embedding_memmap, end_index


async def _embed_dataset(dataset: Dataset, out_path: Path) -> np.ndarray:
    n = len(dataset)
    cached = _load_cached_embeddings(out_path, expected_rows=n)
    if cached is not None:
        logger.info("Cached: %s", out_path)
        return cached

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Embedding %d samples → %s", n, out_path)

    if n == 0:
        arr = np.zeros((0, 0), dtype=np.float32)
        np.save(str(out_path), arr)
        logger.info("Saved %s  shape=%s", out_path, arr.shape)
        return arr

    embedding_memmap: np.memmap | None = None
    tmp_path = out_path.with_suffix(f"{out_path.suffix}.tmp")
    tmp_path.unlink(missing_ok=True)
    semaphore = asyncio.Semaphore(settings.teacher_concurrency)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        for batch_start in range(0, n, _BATCH_SIZE):
            batch_end = min(batch_start + _BATCH_SIZE, n)
            tasks = [
                _request_embedding(
                    client,
                    semaphore,
                    _build_payload(
                        dataset[i].get("text"),
                        dataset[i].get("image"),
                    ),
                )
                for i in range(batch_start, batch_end)
            ]
            results = await asyncio.gather(*tasks)
            batch_array = np.asarray(results, dtype=np.float32)
            if embedding_memmap is None:
                embedding_memmap = open_memmap(
                    str(tmp_path),
                    mode="w+",
                    dtype=np.float32,
                    shape=(n, batch_array.shape[1]),
                )
            embedding_memmap[batch_start:batch_end] = batch_array
            embedding_memmap.flush()
            logger.info("  %d / %d", batch_end, n)

    if embedding_memmap is None:
        raise RuntimeError(f"No embeddings were written for dataset at {out_path}")

    tmp_path.replace(out_path)
    logger.info("Saved %s  shape=%s", out_path, embedding_memmap.shape)
    return np.load(str(out_path), mmap_mode="r")


async def _embed_payloads(
    payload_iter: Any, *, total_rows: int, out_path: Path
) -> np.ndarray:
    cached = _load_cached_embeddings(out_path, expected_rows=total_rows)
    if cached is not None:
        logger.info("Cached: %s", out_path)
        return cached

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Embedding %d samples → %s", total_rows, out_path)

    if total_rows == 0:
        arr = np.zeros((0, 0), dtype=np.float32)
        np.save(str(out_path), arr)
        logger.info("Saved %s  shape=%s", out_path, arr.shape)
        return arr

    embedding_memmap: np.memmap | None = None
    tmp_path = out_path.with_suffix(f"{out_path.suffix}.tmp")
    tmp_path.unlink(missing_ok=True)
    semaphore = asyncio.Semaphore(settings.teacher_concurrency)
    next_index = 0
    batch_payloads: list[dict] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        for payload in payload_iter:
            batch_payloads.append(payload)
            if len(batch_payloads) < _BATCH_SIZE:
                continue
            embedding_memmap, next_index = await _write_payload_batch(
                client=client,
                semaphore=semaphore,
                payloads=batch_payloads,
                embedding_memmap=embedding_memmap,
                start_index=next_index,
                total_rows=total_rows,
                tmp_path=tmp_path,
            )
            batch_payloads = []

        if batch_payloads:
            embedding_memmap, next_index = await _write_payload_batch(
                client=client,
                semaphore=semaphore,
                payloads=batch_payloads,
                embedding_memmap=embedding_memmap,
                start_index=next_index,
                total_rows=total_rows,
                tmp_path=tmp_path,
            )

    if embedding_memmap is None or next_index != total_rows:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Expected to write {total_rows} embeddings to {out_path}, wrote {next_index}"
        )

    tmp_path.replace(out_path)
    logger.info("Saved %s  shape=%s", out_path, embedding_memmap.shape)
    return np.load(str(out_path), mmap_mode="r")
