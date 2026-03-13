from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from delta_embed_vl.evals.encoder import DeltaEmbedEncoder
from delta_embed_vl.evals.image_cache import load_remote_image
from delta_embed_vl.evals.retrieval import mean_recall_at_k
from delta_embed_vl.evals.types import EvalResult

logger = logging.getLogger(__name__)

_DOCCI_FAST_LIMIT = 500
_DOCCI_CACHE_DIR = Path("data/eval_cache/docci")


def run_docci(
    encoder: DeltaEmbedEncoder,
    *,
    batch_size: int,
    fast: bool = False,
) -> list[EvalResult]:
    rows = _load_docci_rows()
    if fast:
        rows = rows[:_DOCCI_FAST_LIMIT]

    images = []
    descriptions: list[str] = []
    skipped = 0

    for row in rows:
        image_url = row.get("image_url")
        description = row.get("description")
        if not isinstance(image_url, str) or not image_url or not description:
            skipped += 1
            continue

        try:
            images.append(load_remote_image(image_url, cache_dir=_DOCCI_CACHE_DIR))
        except Exception as exc:
            skipped += 1
            logger.warning("Skipping DOCCI image %s: %s", row.get("image_key"), exc)
            continue
        descriptions.append(description)

    logger.info(
        "DOCCI rows=%d usable_pairs=%d skipped=%d",
        len(rows),
        len(descriptions),
        skipped,
    )
    if not images or not descriptions:
        raise ValueError("DOCCI evaluation produced no usable image/text pairs.")

    image_embeddings = encoder.encode_images(images, batch_size=batch_size)
    description_embeddings = encoder.encode_texts(descriptions, batch_size=batch_size)
    one_to_one = [{index} for index in range(len(images))]
    text_to_image = mean_recall_at_k(
        description_embeddings,
        image_embeddings,
        one_to_one,
        k=1,
    )
    image_to_text = mean_recall_at_k(
        image_embeddings,
        description_embeddings,
        one_to_one,
        k=1,
    )
    return [
        EvalResult(
            eval_type="Text -> Image",
            benchmark="Docci",
            score=text_to_image,
            metric="recall@1",
        ),
        EvalResult(
            eval_type="Image -> Text",
            benchmark="Docci",
            score=image_to_text,
            metric="recall@1",
        ),
    ]


@lru_cache(maxsize=1)
def _load_docci_rows() -> list[dict[str, Any]]:
    parquet_path = hf_hub_download(
        repo_id="google/docci",
        repo_type="dataset",
        revision="refs/convert/parquet",
        filename="default/train/0000.parquet",
    )
    dataset = load_dataset(
        "parquet",
        data_files={"train": [parquet_path]},
        split="train",
    )
    return [dataset[index] for index in range(len(dataset))]
