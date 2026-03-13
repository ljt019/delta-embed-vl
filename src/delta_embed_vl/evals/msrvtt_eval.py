from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Audio, Video, disable_progress_bars, load_dataset

from delta_embed_vl.evals.encoder import DeltaEmbedEncoder
from delta_embed_vl.evals.retrieval import mean_ndcg_at_k
from delta_embed_vl.evals.types import EvalResult

logger = logging.getLogger(__name__)

_MSRVTT_FAST_LIMIT = 100
_MSRVTT_CACHE_DIR = Path("data/eval_cache/msrvtt")


def run_msrvtt(
    encoder: DeltaEmbedEncoder,
    *,
    batch_size: int,
    fast: bool = False,
) -> list[EvalResult]:
    rows = _load_msrvtt_rows()
    if fast:
        rows = rows[:_MSRVTT_FAST_LIMIT]

    video_paths: list[Path] = []
    captions: list[str] = []
    for row in rows:
        video_paths.append(_materialize_video(row["video"]))
        captions.append(str(row["caption"]))

    logger.info("MSR-VTT rows=%d", len(rows))
    if not video_paths or not captions:
        raise ValueError("MSR-VTT evaluation produced no usable video/text pairs.")

    video_embeddings = encoder.encode_videos(video_paths, batch_size=batch_size)
    caption_embeddings = encoder.encode_texts(captions, batch_size=batch_size)
    one_to_one = [{index} for index in range(len(video_paths))]
    text_to_video = mean_ndcg_at_k(
        caption_embeddings,
        video_embeddings,
        one_to_one,
        k=10,
    )
    return [
        EvalResult(
            eval_type="Text -> Video",
            benchmark="MSR-VTT",
            score=text_to_video,
            metric="ndcg@10",
        )
    ]


def _load_msrvtt_rows() -> list[dict[str, Any]]:
    disable_progress_bars()
    dataset = load_dataset("mteb/MSR-VTT", split="test")
    dataset = dataset.cast_column("video", Video(decode=False)).cast_column(
        "audio",
        Audio(decode=False),
    )
    return [dataset[index] for index in range(len(dataset))]


def _materialize_video(video_item: dict[str, Any]) -> Path:
    _MSRVTT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = Path(str(video_item["path"])).name
    cache_path = _MSRVTT_CACHE_DIR / filename
    if cache_path.exists():
        return cache_path

    video_bytes = video_item.get("bytes")
    if not isinstance(video_bytes, bytes):
        raise ValueError(f"MSR-VTT video payload missing bytes for {filename}.")

    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    tmp_path.write_bytes(video_bytes)
    tmp_path.replace(cache_path)
    return cache_path
