from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from PIL import Image

from delta_embed_vl import cfg
from delta_embed_vl.data.sources import load_all_samples
from delta_embed_vl.data.teacher import TeacherEmbedder, load_teacher
from delta_embed_vl.model.tokenization import DEFAULT_EMBED_INSTRUCTION, EmbeddingInput

logger = logging.getLogger(__name__)
_DATASET_DIR = Path("data/dataset")

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
) -> None:
    if teacher_batch_size < 1:
        raise ValueError("teacher_batch_size must be at least 1.")

    logger.info("Normalizing samples")
    normalized_ds = _build_normalized(
        limit=limit,
        limit_all=limit_all,
        max_length=max_length,
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


def _build_normalized(
    *,
    limit: int | None,
    limit_all: bool,
    max_length: int,
) -> Dataset:
    temp_dir = _DATASET_DIR.with_name("dataset.normalize-build")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    arrow_path = temp_dir / "data.arrow"
    writer = ArrowWriter(path=str(arrow_path), features=_NORMALIZED_FEATURES)
    rows_written = 0

    for batch in _normalize_batches(
        limit=limit,
        limit_all=limit_all,
        max_length=max_length,
    ):
        writer.write_batch(batch)
        rows_written += len(batch["source"])
    writer.finalize()
    logger.info("Normalized %d samples", rows_written)

    if rows_written == 0:
        dataset = Dataset.from_dict(
            {name: [] for name in _NORMALIZED_FEATURES},
            features=_NORMALIZED_FEATURES,
        )
    else:
        dataset = Dataset.from_file(str(arrow_path))

    shutil.rmtree(temp_dir, ignore_errors=True)
    return dataset


def _normalize_batches(
    *,
    limit: int | None,
    limit_all: bool,
    max_length: int,
) -> Iterator[dict[str, list[object]]]:
    buffer: list[dict[str, object]] = []
    batch_size = 512
    for sample in load_all_samples(
        limit=limit,
        limit_all=limit_all,
        student_max_length=max_length,
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
        if len(buffer) < batch_size:
            continue
        yield _flush_buffer(buffer)
        buffer = []
    if buffer:
        yield _flush_buffer(buffer)


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
