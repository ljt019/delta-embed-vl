from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from collections import deque
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
)
from datasets import Image as HFImage
from datasets.arrow_writer import ArrowWriter
from PIL import Image
from qwen_vl_utils import smart_resize

from delta_embed_vl import cfg
from delta_embed_vl.data.download import load_raw_cauldron, load_raw_wikipedia
from delta_embed_vl.data.sources import iter_source_samples, normalization_source_names
from delta_embed_vl.data.teacher import TeacherEmbedder, load_teacher
from delta_embed_vl.model.tokenization import DEFAULT_EMBED_INSTRUCTION, EmbeddingInput

logger = logging.getLogger(__name__)
_DATASET_DIR = Path("data/dataset")
_NORMALIZED_DIR = Path("data/normalized")

_NORMALIZED_FEATURES = Features(
    {
        "text": Value("string"),
        "image": HFImage(),
        "instruction": Value("string"),
        "source": Value("string"),
        "role": Value("string"),
    }
)

_DATASET_FEATURES = Features(
    {
        "text": Value("string"),
        "image": HFImage(),
        "instruction": Value("string"),
        "source": Value("string"),
        "role": Value("string"),
        "teacher_embedding": Sequence(Value("float32")),
    }
)

_NORMALIZED_WRITE_BATCH_SIZE = 512
_MIN_RAW_ROWS_PER_NORMALIZATION_TASK = 512
_REBUCKET_WINDOW_BATCHES = 4


@dataclass(frozen=True)
class NormalizationTask:
    source: str
    shard_index: int = 0
    num_shards: int = 1


def _rows_per_second(rows: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return rows / elapsed_s


def _format_timing_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _log_detailed_timing(
    enabled: bool,
    *,
    stage: str,
    elapsed_s: float,
    **fields: object,
) -> None:
    if not enabled:
        return
    parts = [f"TIMING stage={stage}", f"elapsed_s={elapsed_s:.3f}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_format_timing_value(value)}")
    logger.info(" ".join(parts))


def load_training_dataset() -> Dataset:
    """Load the canonical dataset from disk, or auto-pull from Hub if missing."""
    if _is_saved_dataset(_DATASET_DIR):
        logger.info("Loading dataset from disk: %s", _DATASET_DIR)
        return Dataset.load_from_disk(str(_DATASET_DIR))

    hub_id = cfg["data"]["id"]
    logger.info("Dataset not found locally, downloading from Hub: %s", hub_id)
    dataset = load_dataset(hub_id, split="train")
    _save_dataset(dataset, _DATASET_DIR)
    logger.info("Downloaded and cached dataset: rows=%d", len(dataset))
    return dataset


def build_dataset(
    *,
    limit: int | None = None,
    limit_all: bool = False,
    max_length: int = cfg["max_length"],
    teacher_batch_size: int = cfg["data"]["batch_size"],
    attention: str | None = None,
    push_to_hub: bool = False,
    no_stream: bool = False,
    rebuild_normalized: bool = False,
    detailed_timings: bool = False,
) -> None:
    if teacher_batch_size < 1:
        raise ValueError("teacher_batch_size must be at least 1.")
    build_started = time.perf_counter()

    normalized_ds = _load_or_build_normalized(
        limit=limit,
        limit_all=limit_all,
        max_length=max_length,
        no_stream=no_stream,
        rebuild_normalized=rebuild_normalized,
        detailed_timings=detailed_timings,
    )

    logger.info("Embedding normalized samples with teacher")
    dataset, temp_dirs = _embed_normalized(
        normalized_ds=normalized_ds,
        teacher_batch_size=teacher_batch_size,
        attention=attention,
        detailed_timings=detailed_timings,
    )
    save_started = time.perf_counter()
    try:
        _save_dataset(dataset, _DATASET_DIR)
        logger.info("Dataset saved: rows=%d path=%s", len(dataset), _DATASET_DIR)
        _log_detailed_timing(
            detailed_timings,
            stage="dataset_save",
            elapsed_s=time.perf_counter() - save_started,
            rows=len(dataset),
            path=_DATASET_DIR,
        )
    finally:
        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)

    if push_to_hub:
        hub_id = cfg["data"]["id"]
        logger.info("Pushing dataset to Hub: %s", hub_id)
        push_started = time.perf_counter()
        dataset.push_to_hub(hub_id, split="train")
        _log_detailed_timing(
            detailed_timings,
            stage="dataset_push",
            elapsed_s=time.perf_counter() - push_started,
            rows=len(dataset),
            hub_id=hub_id,
        )

    total_elapsed = time.perf_counter() - build_started
    _log_detailed_timing(
        detailed_timings,
        stage="prepare_total",
        elapsed_s=total_elapsed,
        rows=len(dataset),
        rows_per_s=_rows_per_second(len(dataset), total_elapsed),
        limit="all" if limit is None else limit,
        no_stream=no_stream,
        rebuild_normalized=rebuild_normalized,
        push_to_hub=push_to_hub,
    )


_NORMALIZED_META = _NORMALIZED_DIR / "_meta.json"


def _normalized_cache_valid(limit: int | None, limit_all: bool) -> bool:
    if not _is_saved_dataset(_NORMALIZED_DIR) or not _NORMALIZED_META.exists():
        return False
    meta = json.loads(_NORMALIZED_META.read_text())
    cached_all = meta.get("limit_all", False)
    cached_limit = meta.get("limit")

    if limit_all:
        return cached_all
    if cached_all:
        return True
    if cached_limit is None:
        return False
    return limit is not None and cached_limit >= limit


def _write_normalized_meta(limit: int | None, limit_all: bool) -> None:
    _NORMALIZED_META.write_text(json.dumps({"limit": limit, "limit_all": limit_all}))


def _load_or_build_normalized(
    *,
    limit: int | None,
    limit_all: bool,
    max_length: int,
    no_stream: bool = False,
    rebuild_normalized: bool = False,
    detailed_timings: bool = False,
) -> Dataset:
    total_started = time.perf_counter()
    if rebuild_normalized:
        if _is_saved_dataset(_NORMALIZED_DIR):
            logger.info("Rebuilding normalized cache by request")
    elif _normalized_cache_valid(limit, limit_all):
        dataset = Dataset.load_from_disk(str(_NORMALIZED_DIR))
        logger.info("Normalized cache hit: %s", _NORMALIZED_DIR)
        _log_detailed_timing(
            detailed_timings,
            stage="normalize_cache_hit",
            elapsed_s=time.perf_counter() - total_started,
            rows=len(dataset),
            path=_NORMALIZED_DIR,
        )
        return dataset
    elif _is_saved_dataset(_NORMALIZED_DIR):
        logger.info("Normalized cache stale, rebuilding")

    resolved_limit = None if limit_all else limit
    tasks = _plan_normalization_tasks(limit=resolved_limit)
    cpu_workers = _detect_cpu_workers(len(tasks))
    available_cpus = _detect_available_cpu_count()
    logger.info(
        "Normalizing samples across %d CPU worker(s) (detected=%d, tasks=%d)",
        cpu_workers,
        available_cpus,
        len(tasks),
    )
    prime_started = time.perf_counter()
    _prime_sharded_raw_caches(tasks, limit=resolved_limit, no_stream=no_stream)
    sharded_sources = len({task.source for task in tasks if task.num_shards > 1})
    _log_detailed_timing(
        detailed_timings,
        stage="normalize_prime_raw",
        elapsed_s=time.perf_counter() - prime_started,
        sharded_sources=sharded_sources,
        tasks=len(tasks),
    )

    temp_root = _NORMALIZED_DIR.with_name("normalized.build")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        generate_started = time.perf_counter()
        task_results: dict[int, tuple[str, int, float]] = {}
        if cpu_workers == 1:
            for idx, task in enumerate(tasks):
                task_results[idx] = _normalize_task_to_arrow(
                    task,
                    limit=resolved_limit,
                    max_length=max_length,
                    no_stream=no_stream,
                    temp_root=str(temp_root),
                )
                _, rows, elapsed_s = task_results[idx]
                logger.info(
                    "Normalized %s: rows=%d%s (%d/%d)",
                    _normalization_task_label(task),
                    rows,
                    (
                        f" elapsed_s={elapsed_s:.3f} "
                        f"rows_per_s={_rows_per_second(rows, elapsed_s):.1f}"
                        if detailed_timings
                        else ""
                    ),
                    idx + 1,
                    len(tasks),
                )
        else:
            with ProcessPoolExecutor(max_workers=cpu_workers) as pool:
                futures = {
                    pool.submit(
                        _normalize_task_to_arrow,
                        task,
                        limit=resolved_limit,
                        max_length=max_length,
                        no_stream=no_stream,
                        temp_root=str(temp_root),
                    ): idx
                    for idx, task in enumerate(tasks)
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    idx = futures[future]
                    task_results[idx] = future.result()
                    _, rows, elapsed_s = task_results[idx]
                    logger.info(
                        "Normalized %s: rows=%d%s (%d/%d)",
                        _normalization_task_label(tasks[idx]),
                        rows,
                        (
                            f" elapsed_s={elapsed_s:.3f} "
                            f"rows_per_s={_rows_per_second(rows, elapsed_s):.1f}"
                            if detailed_timings
                            else ""
                        ),
                        completed,
                        len(tasks),
                    )

        ordered_results = [task_results[i] for i in range(len(tasks))]
        rows_written = sum(rows for _, rows, _ in ordered_results)
        logger.info("Normalized %d samples", rows_written)
        generate_elapsed = time.perf_counter() - generate_started
        _log_detailed_timing(
            detailed_timings,
            stage="normalize_generate",
            elapsed_s=generate_elapsed,
            rows=rows_written,
            rows_per_s=_rows_per_second(rows_written, generate_elapsed),
            tasks=len(tasks),
            workers=cpu_workers,
        )

        merge_started = time.perf_counter()
        shard_datasets = [
            Dataset.from_file(str(Path(temp_dir) / "data.arrow"))
            for temp_dir, rows, _ in ordered_results
            if rows > 0
        ]
        if not shard_datasets:
            dataset = Dataset.from_dict(
                {name: [] for name in _NORMALIZED_FEATURES},
                features=_NORMALIZED_FEATURES,
            )
        elif len(shard_datasets) == 1:
            dataset = shard_datasets[0]
        else:
            dataset = concatenate_datasets(shard_datasets)
        merge_elapsed = time.perf_counter() - merge_started
        _log_detailed_timing(
            detailed_timings,
            stage="normalize_merge",
            elapsed_s=merge_elapsed,
            rows=rows_written,
            shards=len(shard_datasets),
        )

        save_started = time.perf_counter()
        _save_dataset(dataset, _NORMALIZED_DIR)
        _write_normalized_meta(limit, limit_all)
        save_elapsed = time.perf_counter() - save_started
        _log_detailed_timing(
            detailed_timings,
            stage="normalize_save",
            elapsed_s=save_elapsed,
            rows=rows_written,
            path=_NORMALIZED_DIR,
        )
        total_elapsed = time.perf_counter() - total_started
        _log_detailed_timing(
            detailed_timings,
            stage="normalize_total",
            elapsed_s=total_elapsed,
            rows=rows_written,
            rows_per_s=_rows_per_second(rows_written, total_elapsed),
            limit="all" if limit_all else limit,
        )
        return dataset
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _detect_available_cpu_count() -> int:
    cpu_counter = getattr(os, "process_cpu_count", os.cpu_count)
    return max(1, cpu_counter() or 1)


def _detect_cpu_workers(num_tasks: int) -> int:
    return max(1, min(_detect_available_cpu_count(), num_tasks))


def _source_max_shards(*, limit: int | None) -> int:
    if limit is None:
        return _detect_available_cpu_count()
    return max(1, math.ceil(limit / _MIN_RAW_ROWS_PER_NORMALIZATION_TASK))


def _plan_normalization_tasks(*, limit: int | None) -> list[NormalizationTask]:
    sources = normalization_source_names()
    target_tasks = max(1, _detect_available_cpu_count())
    max_shards_per_source = _source_max_shards(limit=limit)
    shard_counts = {source: 1 for source in sources}
    remaining_tasks = max(0, target_tasks - len(sources))

    while remaining_tasks > 0:
        advanced = False
        for source in sources:
            if shard_counts[source] >= max_shards_per_source:
                continue
            shard_counts[source] += 1
            remaining_tasks -= 1
            advanced = True
            if remaining_tasks == 0:
                break
        if not advanced:
            break

    return [
        NormalizationTask(
            source=source,
            shard_index=shard_index,
            num_shards=shard_counts[source],
        )
        for source in sources
        for shard_index in range(shard_counts[source])
    ]


def _prime_sharded_raw_caches(
    tasks: list[NormalizationTask],
    *,
    limit: int | None,
    no_stream: bool,
) -> None:
    sharded_sources = sorted({task.source for task in tasks if task.num_shards > 1})
    if not sharded_sources:
        return

    logger.info("Priming raw cache for %d sharded source(s)", len(sharded_sources))
    max_workers = min(
        len(sharded_sources), max(1, min(_detect_available_cpu_count(), 16))
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _prime_raw_source,
                source,
                limit=limit,
                no_stream=no_stream,
            ): source
            for source in sharded_sources
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                future.result()
            except Exception:
                logger.exception("Failed to prime raw cache for %s", source)
                raise


def _prime_raw_source(source: str, *, limit: int | None, no_stream: bool) -> None:
    if source == "wikipedia":
        load_raw_wikipedia(limit=limit, no_stream=no_stream)
        return
    if source.startswith("cauldron/"):
        load_raw_cauldron(
            source.removeprefix("cauldron/"),
            limit=limit,
            no_stream=no_stream,
        )
        return
    supported_sources = ", ".join(normalization_source_names())
    raise ValueError(
        f"Unknown normalization source: {source}. Supported: {supported_sources}"
    )


def _normalization_task_label(task: NormalizationTask) -> str:
    if task.num_shards == 1:
        return task.source
    return f"{task.source}[{task.shard_index + 1}/{task.num_shards}]"


def _normalize_task_to_arrow(
    task: NormalizationTask,
    *,
    limit: int | None,
    max_length: int,
    no_stream: bool,
    temp_root: str,
) -> tuple[str, int, float]:
    task_started = time.perf_counter()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    source_slug = task.source.replace("/", "-")
    temp_dir = Path(temp_root) / (
        f"normalized-{source_slug}-{task.shard_index}-of-{task.num_shards}"
    )
    temp_dir.mkdir(parents=True, exist_ok=True)
    arrow_path = temp_dir / "data.arrow"
    writer = ArrowWriter(path=str(arrow_path), features=_NORMALIZED_FEATURES)
    rows_written = 0
    buffer: list[dict[str, object]] = []

    for sample in iter_source_samples(
        task.source,
        limit=limit,
        student_max_length=max_length,
        no_stream=no_stream,
        shard_index=task.shard_index,
        num_shards=task.num_shards,
    ):
        buffer.append(
            {
                "text": sample.text,
                "image": sample.image,
                "instruction": sample.instruction,
                "source": sample.source,
                "role": sample.role,
            }
        )
        if len(buffer) < _NORMALIZED_WRITE_BATCH_SIZE:
            continue
        writer.write_batch(_flush_buffer(buffer))
        rows_written += len(buffer)
        buffer = []

    if buffer:
        writer.write_batch(_flush_buffer(buffer))
        rows_written += len(buffer)

    writer.finalize()
    return str(temp_dir), rows_written, time.perf_counter() - task_started


def _flush_buffer(buffer: list[dict[str, object]]) -> dict[str, list[object]]:
    return {
        "text": [r["text"] for r in buffer],
        "image": [r["image"] for r in buffer],
        "instruction": [r["instruction"] for r in buffer],
        "source": [r["source"] for r in buffer],
        "role": [r["role"] for r in buffer],
    }


def _detect_devices() -> list[str]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for data preparation. Refusing to fall back to CPU."
        )
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def _load_embedding_batch(
    normalized_ds: Dataset,
    indices: list[int],
) -> dict[str, list[object]]:
    if not indices:
        return {name: [] for name in _NORMALIZED_FEATURES}

    start = indices[0]
    stop = indices[-1] + 1
    if indices == list(range(start, stop)):
        batch = normalized_ds[start:stop]
    else:
        batch = normalized_ds[indices]

    return {name: list(batch[name]) for name in _NORMALIZED_FEATURES}


def _resolve_rebucket_vision_params(
    processor: object,
) -> tuple[int, int | None, int | None]:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise AttributeError("Teacher processor does not expose an image_processor.")

    patch_size = int(getattr(image_processor, "patch_size", 14))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    min_pixels = getattr(image_processor, "min_pixels", None)
    max_pixels = getattr(image_processor, "max_pixels", None)
    return (
        patch_size * merge_size,
        None if min_pixels is None else int(min_pixels),
        None if max_pixels is None else int(max_pixels),
    )


def _estimate_sample_tokens(
    text: str | None,
    image: Image.Image | None,
    instruction: str | None,
    *,
    image_factor: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> int:
    """Estimate token cost using Qwen's resize helper and processor config."""
    image_tokens = 0
    if image is not None:
        w, h = image.size
        resized_height, resized_width = smart_resize(
            h,
            w,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image_tokens = (resized_height * resized_width) // (image_factor * image_factor)
    text_len = len(instruction or DEFAULT_EMBED_INSTRUCTION) + len(text or "")
    return image_tokens + text_len // 4


def _rebucket_window(
    window_rows: dict[str, list[object]],
    batch_size: int,
    *,
    image_factor: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> list[list[int]]:
    """Sort samples in a window by estimated token cost, return groups of indices."""
    texts = window_rows["text"]
    images = window_rows["image"]
    instructions = window_rows["instruction"]
    n = len(texts)
    if n == 0:
        return []
    order = sorted(
        range(n),
        key=lambda i: _estimate_sample_tokens(
            texts[i],
            images[i],
            instructions[i],
            image_factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        ),
        reverse=True,
    )
    return [order[s : s + batch_size] for s in range(0, len(order), batch_size)]


def _embed_shard(
    teacher: TeacherEmbedder,
    normalized_ds: Dataset,
    indices: list[int],
    batch_size: int,
    shard_id: int,
) -> tuple[Path, int, float, float]:
    """Embed a contiguous shard on one device, return (arrow_dir, rows, peak_gpu_pct, elapsed_s)."""
    shard_started = time.perf_counter()
    temp_dir = _DATASET_DIR.with_name(f"dataset.embed-shard-{shard_id}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    arrow_path = temp_dir / "data.arrow"
    writer = ArrowWriter(path=str(arrow_path), features=_DATASET_FEATURES)
    rows_written = 0
    device = str(teacher.device)
    gpu_available = device.startswith("cuda") and torch.cuda.is_available()
    peak_pct = 0.0

    _MAX_PENDING_WRITES = 7
    window_size = batch_size * _REBUCKET_WINDOW_BATCHES
    image_factor, min_pixels, max_pixels = _resolve_rebucket_vision_params(
        teacher.processor
    )

    with ThreadPoolExecutor(max_workers=1) as writer_pool:
        pending_writes: deque[Future] = deque()

        for win_start in range(0, len(indices), window_size):
            win_indices = indices[win_start : win_start + window_size]
            window_rows = _load_embedding_batch(normalized_ds, win_indices)
            groups = _rebucket_window(
                window_rows,
                batch_size,
                image_factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

            win_texts = window_rows["text"]
            win_images = window_rows["image"]
            win_instructions = window_rows["instruction"]

            window_embeddings: torch.Tensor | None = None
            for positions in groups:
                inputs = [
                    EmbeddingInput(
                        text=str(win_texts[i]) if win_texts[i] else None,
                        image=cast(Image.Image | None, win_images[i]),
                        instruction=str(
                            win_instructions[i] or DEFAULT_EMBED_INSTRUCTION
                        ),
                    )
                    for i in positions
                ]
                embeddings = teacher.embed(inputs)

                if window_embeddings is None:
                    window_embeddings = torch.empty(
                        (len(win_indices), embeddings.shape[1]),
                        device=embeddings.device,
                        dtype=embeddings.dtype,
                    )
                window_embeddings[positions] = embeddings

                if gpu_available:
                    free, total = torch.cuda.mem_get_info(device)
                    pct = (total - free) / total * 100
                    peak_pct = max(peak_pct, pct)

            if window_embeddings is None:
                continue

            for write_start in range(0, len(win_indices), batch_size):
                write_end = min(write_start + batch_size, len(win_indices))
                batch_payload = {
                    name: window_rows[name][write_start:write_end]
                    for name in window_rows
                }
                batch_payload["teacher_embedding"] = (
                    window_embeddings[write_start:write_end]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
                if len(pending_writes) >= _MAX_PENDING_WRITES:
                    pending_writes.popleft().result()
                pending_writes.append(
                    writer_pool.submit(writer.write_batch, batch_payload)
                )
                rows_written += write_end - write_start

        while pending_writes:
            pending_writes.popleft().result()

    writer.finalize()
    return temp_dir, rows_written, peak_pct, time.perf_counter() - shard_started


def _embed_normalized(
    *,
    normalized_ds: Dataset,
    teacher_batch_size: int,
    attention: str | None,
    detailed_timings: bool = False,
) -> tuple[Dataset, list[Path]]:
    embed_started = time.perf_counter()
    devices = _detect_devices()
    n = len(normalized_ds)

    if n == 0:
        return (
            Dataset.from_dict(
                {name: [] for name in _DATASET_FEATURES},
                features=_DATASET_FEATURES,
            ),
            [],
        )

    logger.info(
        "Loading teacher on %d device(s): %s",
        len(devices),
        ", ".join(devices),
    )
    teacher_load_started = time.perf_counter()
    teachers = [load_teacher(device=d, attn_implementation=attention) for d in devices]
    teacher_load_elapsed = time.perf_counter() - teacher_load_started
    _log_detailed_timing(
        detailed_timings,
        stage="embed_teacher_load",
        elapsed_s=teacher_load_elapsed,
        devices=len(devices),
    )

    num_shards = len(devices)
    shard_indices = [list(range(i, n, num_shards)) for i in range(num_shards)]

    temp_dirs: list[Path] = []
    compute_started = time.perf_counter()
    if num_shards == 1:
        temp_dir, rows, peak, elapsed_s = _embed_shard(
            teachers[0], normalized_ds, shard_indices[0], teacher_batch_size, 0
        )
        temp_dirs.append(temp_dir)
        shard_results = [(temp_dir, rows, peak, elapsed_s)]
    else:
        with ThreadPoolExecutor(max_workers=num_shards) as pool:
            futures = {
                pool.submit(
                    _embed_shard,
                    teachers[i],
                    normalized_ds,
                    shard_indices[i],
                    teacher_batch_size,
                    i,
                ): i
                for i in range(num_shards)
            }
            shard_results_unordered: dict[int, tuple[Path, int, float, float]] = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                shard_results_unordered[idx] = fut.result()
            shard_results = [shard_results_unordered[i] for i in range(num_shards)]

        temp_dirs = [r[0] for r in shard_results]
    compute_elapsed = time.perf_counter() - compute_started

    total_rows = sum(r[1] for r in shard_results)
    gpu_available = devices[0].startswith("cuda")
    if gpu_available:
        peak_pct = max(r[2] for r in shard_results)
        if peak_pct < 80:
            logger.warning(
                "GPU utilization peaked at %.0f%% — increase data.batch_size "
                "in config.toml for faster prep",
                peak_pct,
            )
        else:
            logger.info("GPU utilization peaked at %.0f%%", peak_pct)

    logger.info("Embedded %d rows across %d device(s)", total_rows, num_shards)
    for shard_id, (temp_dir, rows, peak_pct, elapsed_s) in enumerate(shard_results):
        _log_detailed_timing(
            detailed_timings,
            stage="embed_shard",
            elapsed_s=elapsed_s,
            shard=shard_id,
            device=devices[shard_id],
            rows=rows,
            rows_per_s=_rows_per_second(rows, elapsed_s),
            peak_gpu_pct=peak_pct if gpu_available else None,
        )
    _log_detailed_timing(
        detailed_timings,
        stage="embed_compute",
        elapsed_s=compute_elapsed,
        rows=total_rows,
        rows_per_s=_rows_per_second(total_rows, compute_elapsed),
        devices=num_shards,
        batch_size=teacher_batch_size,
    )

    merge_started = time.perf_counter()
    if num_shards == 1:
        arrow_path = temp_dirs[0] / "data.arrow"
        dataset = Dataset.from_file(str(arrow_path))
        merge_elapsed = time.perf_counter() - merge_started
        _log_detailed_timing(
            detailed_timings,
            stage="embed_merge",
            elapsed_s=merge_elapsed,
            rows=total_rows,
            shards=1,
        )
        total_elapsed = time.perf_counter() - embed_started
        _log_detailed_timing(
            detailed_timings,
            stage="embed_total",
            elapsed_s=total_elapsed,
            rows=total_rows,
            rows_per_s=_rows_per_second(total_rows, total_elapsed),
            devices=num_shards,
        )
        return dataset, temp_dirs

    shard_datasets = [Dataset.from_file(str(d / "data.arrow")) for d in temp_dirs]
    dataset = concatenate_datasets(shard_datasets)
    merge_elapsed = time.perf_counter() - merge_started
    _log_detailed_timing(
        detailed_timings,
        stage="embed_merge",
        elapsed_s=merge_elapsed,
        rows=total_rows,
        shards=len(shard_datasets),
    )
    total_elapsed = time.perf_counter() - embed_started
    _log_detailed_timing(
        detailed_timings,
        stage="embed_total",
        elapsed_s=total_elapsed,
        rows=total_rows,
        rows_per_s=_rows_per_second(total_rows, total_elapsed),
        devices=num_shards,
    )
    return dataset, temp_dirs


def _is_saved_dataset(path: Path) -> bool:
    return (path / "dataset_info.json").exists() or (path / "state.json").exists()


def _save_dataset(dataset: Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    if temp_path.exists():
        shutil.rmtree(temp_path)
    dataset.save_to_disk(str(temp_path))
    if output_path.exists():
        shutil.rmtree(output_path)
    temp_path.replace(output_path)
