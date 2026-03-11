from __future__ import annotations

import io
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from datasets.arrow_writer import ArrowWriter
from PIL import Image

from delta_embed_vl.data.sources import load_all_samples
from delta_embed_vl.data.teacher import load_teacher
from delta_embed_vl.model.tokenization import DEFAULT_EMBED_INSTRUCTION, EmbeddingInput
from delta_embed_vl.settings import Settings

logger = logging.getLogger(__name__)
_SETTINGS = Settings()
_DATASET_DIR = _SETTINGS.data_dir / "dataset"

_NORMALIZED_FEATURES = Features(
    {
        "text": Value("string"),
        "image_bytes": Value("binary"),
        "instruction": Value("string"),
        "source": Value("string"),
        "role": Value("string"),
    }
)

_DATASET_FEATURES = Features(
    {
        "text": Value("string"),
        "image_bytes": Value("binary"),
        "instruction": Value("string"),
        "source": Value("string"),
        "role": Value("string"),
        "teacher_embedding": Sequence(Value("float32")),
    }
)


def encode_image(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    return buffer.getvalue()


def decode_image(data: bytes | bytearray | memoryview | None) -> Image.Image | None:
    if data is None:
        return None
    if isinstance(data, memoryview):
        data = data.tobytes()
    elif isinstance(data, bytearray):
        data = bytes(data)
    with Image.open(io.BytesIO(data)) as loaded:
        return loaded.convert("RGB")


def dataset_path() -> Path:
    return _DATASET_DIR


def load_training_dataset() -> Dataset:
    """Load the canonical dataset from disk, or auto-pull from Hub if missing."""
    if _is_saved_dataset(_DATASET_DIR):
        logger.info("Loading dataset from disk: %s", _DATASET_DIR)
        return Dataset.load_from_disk(str(_DATASET_DIR))

    hub_id = _SETTINGS.hub_dataset
    logger.info("Dataset not found locally, downloading from Hub: %s", hub_id)
    dataset = load_dataset(hub_id, split="train")
    _save_dataset(dataset, _DATASET_DIR)
    logger.info("Downloaded and cached dataset: rows=%d", len(dataset))
    return dataset


def build_dataset(
    *,
    limit: int | None = None,
    limit_all: bool = False,
    max_length: int = _SETTINGS.student_max_length,
    teacher_device: str = "cuda:0",
    teacher_batch_size: int = 8,
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
    dataset, temp_build_dir = _embed_normalized(
        normalized_ds=normalized_ds,
        teacher_device=teacher_device,
        teacher_batch_size=teacher_batch_size,
        attention=attention,
    )
    try:
        _save_dataset(dataset, _DATASET_DIR)
        logger.info("Dataset saved: rows=%d path=%s", len(dataset), _DATASET_DIR)
    finally:
        shutil.rmtree(temp_build_dir, ignore_errors=True)

    if push_to_hub:
        hub_id = _SETTINGS.hub_dataset
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
                "image_bytes": None
                if sample.image is None
                else encode_image(sample.image),
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
        "image_bytes": [r["image_bytes"] for r in buffer],
        "instruction": [r["instruction"] for r in buffer],
        "source": [r["source"] for r in buffer],
        "role": [r["role"] for r in buffer],
    }


def _embed_normalized(
    *,
    normalized_ds: Dataset,
    teacher_device: str,
    teacher_batch_size: int,
    attention: str | None,
) -> tuple[Dataset, Path]:
    logger.info("Loading teacher for embedding")
    teacher = load_teacher(
        device=teacher_device,
        attn_implementation=attention,
    )

    temp_build_dir = _DATASET_DIR.with_name("dataset.embed-build")
    if temp_build_dir.exists():
        shutil.rmtree(temp_build_dir)
    temp_build_dir.mkdir(parents=True, exist_ok=True)
    arrow_path = temp_build_dir / "data.arrow"
    writer = ArrowWriter(path=str(arrow_path), features=_DATASET_FEATURES)
    rows_written = 0

    n = len(normalized_ds)
    gpu_available = teacher_device.startswith("cuda") and torch.cuda.is_available()
    peak_pct = 0.0

    for start in range(0, n, teacher_batch_size):
        stop = min(start + teacher_batch_size, n)
        batch_rows = [normalized_ds[i] for i in range(start, stop)]
        inputs = [
            EmbeddingInput(
                text=row.get("text") or None,
                image=decode_image(row.get("image_bytes")),
                instruction=row.get("instruction") or DEFAULT_EMBED_INSTRUCTION,
            )
            for row in batch_rows
        ]
        embeddings = teacher.embed(inputs)

        if gpu_available:
            free, total = torch.cuda.mem_get_info(teacher_device)
            pct = (total - free) / total * 100
            peak_pct = max(peak_pct, pct)

        embeddings_np = embeddings.detach().cpu().float().numpy()
        batch = {
            "text": [r["text"] for r in batch_rows],
            "image_bytes": [r["image_bytes"] for r in batch_rows],
            "instruction": [r["instruction"] for r in batch_rows],
            "source": [r["source"] for r in batch_rows],
            "role": [r["role"] for r in batch_rows],
            "teacher_embedding": [emb.tolist() for emb in embeddings_np],
        }
        writer.write_batch(batch)
        rows_written += len(batch_rows)

    if gpu_available:
        if peak_pct < 50:
            logger.warning(
                "GPU utilization peaked at %.0f%% — you can likely increase "
                "--teacher-batch-size for faster prep",
                peak_pct,
            )
        else:
            logger.info("GPU utilization peaked at %.0f%%", peak_pct)

    writer.finalize()

    if rows_written == 0:
        return (
            Dataset.from_dict(
                {name: [] for name in _DATASET_FEATURES},
                features=_DATASET_FEATURES,
            ),
            temp_build_dir,
        )
    return Dataset.from_file(str(arrow_path)), temp_build_dir


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
