from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from delta_embed_vl.evals.encoder import DeltaEmbedEncoder
from delta_embed_vl.evals.image_cache import load_remote_image
from delta_embed_vl.evals.retrieval import mean_recall_at_k
from delta_embed_vl.evals.types import EvalResult

logger = logging.getLogger(__name__)

_TEXTCAPS_VALIDATION_URL = (
    "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_val.json"
)
_TEXTCAPS_FAST_LIMIT = 500
_TEXTCAPS_CACHE_DIR = Path("data/eval_cache/textcaps")


def run_textcaps(
    encoder: DeltaEmbedEncoder,
    *,
    batch_size: int,
    fast: bool = False,
) -> list[EvalResult]:
    rows = _load_textcaps_rows()
    if fast:
        rows = rows[:_TEXTCAPS_FAST_LIMIT]

    images = []
    image_to_captions: list[set[int]] = []
    captions: list[str] = []
    caption_to_images: list[set[int]] = []
    skipped = 0

    for row in rows:
        reference_strs = [
            caption for caption in row.get("reference_strs", []) if caption
        ]
        image_url = _resolve_image_url(row)
        if not reference_strs or image_url is None:
            skipped += 1
            continue

        try:
            image = load_remote_image(image_url, cache_dir=_TEXTCAPS_CACHE_DIR)
        except Exception as exc:
            skipped += 1
            logger.warning("Skipping TextCaps image %s: %s", row.get("image_id"), exc)
            continue

        image_index = len(images)
        images.append(image)
        relevant_captions: set[int] = set()
        for caption in reference_strs:
            captions.append(caption)
            caption_to_images.append({image_index})
            relevant_captions.add(len(captions) - 1)
        image_to_captions.append(relevant_captions)

    logger.info(
        "TextCaps rows=%d usable_images=%d captions=%d skipped=%d",
        len(rows),
        len(images),
        len(captions),
        skipped,
    )
    if not images or not captions:
        raise ValueError("TextCaps evaluation produced no usable images/captions.")

    image_embeddings = encoder.encode_images(images, batch_size=batch_size)
    caption_embeddings = encoder.encode_texts(captions, batch_size=batch_size)
    text_to_image = mean_recall_at_k(
        caption_embeddings,
        image_embeddings,
        caption_to_images,
        k=1,
    )
    image_to_text = mean_recall_at_k(
        image_embeddings,
        caption_embeddings,
        image_to_captions,
        k=1,
    )
    return [
        EvalResult(
            eval_type="Text -> Image",
            benchmark="TextCaps",
            score=text_to_image,
            metric="recall@1",
        ),
        EvalResult(
            eval_type="Image -> Text",
            benchmark="TextCaps",
            score=image_to_text,
            metric="recall@1",
        ),
    ]


@lru_cache(maxsize=1)
def _load_textcaps_rows() -> list[dict[str, Any]]:
    response = httpx.get(_TEXTCAPS_VALIDATION_URL, timeout=120.0)
    response.raise_for_status()
    payload = response.json()["data"]

    deduped_rows: list[dict[str, Any]] = []
    seen_image_ids: set[str] = set()
    for row in payload:
        image_id = str(row["image_id"])
        if image_id in seen_image_ids:
            continue
        seen_image_ids.add(image_id)
        deduped_rows.append(row)
    return deduped_rows


def _resolve_image_url(row: dict[str, Any]) -> str | None:
    for key in ("flickr_300k_url", "flickr_original_url"):
        candidate = row.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    return None
