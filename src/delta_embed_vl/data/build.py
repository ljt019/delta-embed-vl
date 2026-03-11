from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

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

from delta_embed_vl import cfg
from delta_embed_vl.data.download import load_raw_wikipedia
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
_MIN_RAW_ROWS_PER_WIKIPEDIA_TASK = 512


@dataclass(frozen=True)
class NormalizationTask:
    source: str
    shard_index: int = 0
    num_shards: int = 1


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
) -> None:
    if teacher_batch_size < 1:
        raise ValueError("teacher_batch_size must be at least 1.")

    normalized_ds = _load_or_build_normalized(
        limit=limit,
        limit_all=limit_all,
        max_length=max_length,
        no_stream=no_stream,
    )

    logger.info("Embedding normalized samples with teacher")
    dataset, temp_dirs = _embed_normalized(
        normalized_ds=normalized_ds,
        teacher_batch_size=teacher_batch_size,
        attention=attention,
    )
    try:
        _save_dataset(dataset, _DATASET_DIR)
        logger.info("Dataset saved: rows=%d path=%s", len(dataset), _DATASET_DIR)
    finally:
        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)

    if push_to_hub:
        hub_id = cfg["data"]["id"]
        logger.info("Pushing dataset to Hub: %s", hub_id)
        dataset.push_to_hub(hub_id, split="train")


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
) -> Dataset:
    if _normalized_cache_valid(limit, limit_all):
        logger.info("Normalized cache hit: %s", _NORMALIZED_DIR)
        return Dataset.load_from_disk(str(_NORMALIZED_DIR))

    if _is_saved_dataset(_NORMALIZED_DIR):
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

    wikipedia_shards = max(
        (task.num_shards for task in tasks if task.source == "wikipedia"),
        default=1,
    )
    if wikipedia_shards > 1:
        logger.info("Priming wikipedia raw cache for %d shard(s)", wikipedia_shards)
        load_raw_wikipedia(limit=resolved_limit, no_stream=no_stream)

    temp_root = _NORMALIZED_DIR.with_name("normalized.build")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        task_results: dict[int, tuple[str, int]] = {}
        if cpu_workers == 1:
            for idx, task in enumerate(tasks):
                task_results[idx] = _normalize_task_to_arrow(
                    task,
                    limit=resolved_limit,
                    max_length=max_length,
                    no_stream=no_stream,
                    temp_root=str(temp_root),
                )
                logger.info(
                    "Normalized %s: rows=%d (%d/%d)",
                    _normalization_task_label(task),
                    task_results[idx][1],
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
                    logger.info(
                        "Normalized %s: rows=%d (%d/%d)",
                        _normalization_task_label(tasks[idx]),
                        task_results[idx][1],
                        completed,
                        len(tasks),
                    )

        ordered_results = [task_results[i] for i in range(len(tasks))]
        rows_written = sum(rows for _, rows in ordered_results)
        logger.info("Normalized %d samples", rows_written)

        shard_datasets = [
            Dataset.from_file(str(Path(temp_dir) / "data.arrow"))
            for temp_dir, rows in ordered_results
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

        _save_dataset(dataset, _NORMALIZED_DIR)
        _write_normalized_meta(limit, limit_all)
        return dataset
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _detect_available_cpu_count() -> int:
    cpu_counter = getattr(os, "process_cpu_count", os.cpu_count)
    return max(1, cpu_counter() or 1)


def _detect_cpu_workers(num_tasks: int) -> int:
    return max(1, min(_detect_available_cpu_count(), num_tasks))


def _plan_normalization_tasks(*, limit: int | None) -> list[NormalizationTask]:
    sources = normalization_source_names()
    non_wikipedia_sources = [source for source in sources if source != "wikipedia"]
    max_wikipedia_shards = max(
        1, _detect_available_cpu_count() - len(non_wikipedia_sources)
    )
    if limit is None:
        wikipedia_shards = max_wikipedia_shards
    else:
        wikipedia_shards = min(
            max_wikipedia_shards,
            max(1, math.ceil(limit / _MIN_RAW_ROWS_PER_WIKIPEDIA_TASK)),
        )

    return [
        *[
            NormalizationTask(
                source="wikipedia",
                shard_index=shard_index,
                num_shards=wikipedia_shards,
            )
            for shard_index in range(wikipedia_shards)
        ],
        *[NormalizationTask(source=source) for source in non_wikipedia_sources],
    ]


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
) -> tuple[str, int]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    source_slug = task.source.replace("/", "-")
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"normalized-{source_slug}-{task.shard_index}-",
            dir=temp_root,
        )
    )
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
    return str(temp_dir), rows_written


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
        return ["cpu"]
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def _embed_shard(
    teacher: TeacherEmbedder,
    normalized_ds: Dataset,
    indices: list[int],
    batch_size: int,
    shard_id: int,
) -> tuple[Path, int, float]:
    """Embed a contiguous shard on one device, return (arrow_dir, rows, peak_gpu_pct)."""
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

    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        batch_rows = [normalized_ds[i] for i in chunk]
        inputs = [
            EmbeddingInput(
                text=row.get("text") or None,
                image=row.get("image"),
                instruction=row.get("instruction") or DEFAULT_EMBED_INSTRUCTION,
            )
            for row in batch_rows
        ]
        embeddings = teacher.embed(inputs)

        if gpu_available:
            free, total = torch.cuda.mem_get_info(device)
            pct = (total - free) / total * 100
            peak_pct = max(peak_pct, pct)

        embeddings_np = embeddings.detach().cpu().float().numpy()
        writer.write_batch(
            {
                "text": [r["text"] for r in batch_rows],
                "image": [r["image"] for r in batch_rows],
                "instruction": [r["instruction"] for r in batch_rows],
                "source": [r["source"] for r in batch_rows],
                "role": [r["role"] for r in batch_rows],
                "teacher_embedding": [emb.tolist() for emb in embeddings_np],
            }
        )
        rows_written += len(chunk)

    writer.finalize()
    return temp_dir, rows_written, peak_pct


def _embed_normalized(
    *,
    normalized_ds: Dataset,
    teacher_batch_size: int,
    attention: str | None,
) -> tuple[Dataset, list[Path]]:
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
    teachers = [load_teacher(device=d, attn_implementation=attention) for d in devices]

    num_shards = len(devices)
    shard_indices = [list(range(i, n, num_shards)) for i in range(num_shards)]

    temp_dirs: list[Path] = []
    if num_shards == 1:
        temp_dir, rows, peak = _embed_shard(
            teachers[0], normalized_ds, shard_indices[0], teacher_batch_size, 0
        )
        temp_dirs.append(temp_dir)
        shard_results = [(temp_dir, rows, peak)]
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
            shard_results_unordered: dict[int, tuple[Path, int, float]] = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                shard_results_unordered[idx] = fut.result()
            shard_results = [shard_results_unordered[i] for i in range(num_shards)]

        temp_dirs = [r[0] for r in shard_results]

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

    if num_shards == 1:
        arrow_path = temp_dirs[0] / "data.arrow"
        return Dataset.from_file(str(arrow_path)), temp_dirs

    shard_datasets = [Dataset.from_file(str(d / "data.arrow")) for d in temp_dirs]
    return concatenate_datasets(shard_datasets), temp_dirs


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
