import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from datasets import Dataset
from PIL import Image

from delta_embed_vl.data.download import CAULDRON_CONFIGS
from delta_embed_vl.data.preprocess import (
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
    suffix = f"_test{limit}" if limit else ""
    return asyncio.run(
        _embed_dataset(ds, _EMBEDDINGS_DIR / "cauldron" / f"{config}{suffix}.npy")
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
_BATCH_SIZE = 1000


def _resolve_image_path(image_path: str) -> Path:
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

    raise FileNotFoundError(
        f"Image not found at {image_path}. Expected local cache match under {_RAW_DATA_DIR}."
    )


def _coerce_image(image: Image.Image | dict[str, Any] | None) -> Image.Image | None:
    if image is None:
        return None

    if isinstance(image, Image.Image):
        return image

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
        return Image.open(_resolve_image_path(image_path)).convert("RGB")

    raise ValueError("Image payload must contain either bytes or path.")


def _image_to_data_uri(image: Image.Image | dict[str, Any]) -> str:
    image = _coerce_image(image)
    if image is None:
        raise ValueError("Expected image payload, got None.")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _build_payload(
    text: str | None, image: Image.Image | dict[str, Any] | None
) -> dict:
    """Build a vLLM chat-embedding request body for Qwen3-VL."""
    user_content: list[dict] = []

    if image is not None:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_uri(image)},
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


async def _embed_dataset(dataset: Dataset, out_path: Path) -> np.ndarray:
    if out_path.exists():
        logger.info("Cached: %s", out_path)
        return np.load(str(out_path), mmap_mode="r")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    logger.info("Embedding %d samples → %s", n, out_path)

    all_embeddings: list[list[float]] = []
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
            all_embeddings.extend(results)
            logger.info("  %d / %d", len(all_embeddings), n)

    arr = np.array(all_embeddings, dtype=np.float32)
    np.save(str(out_path), arr)
    logger.info("Saved %s  shape=%s", out_path, arr.shape)
    return arr
