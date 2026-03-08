import argparse
import shutil
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, Image, Sequence, concatenate_datasets, load_dataset
from huggingface_hub.utils import logging as hf_logging

from delta_embed_vl.data.dataloader import create_multimodal_dataloader
from delta_embed_vl.data.dataset import normalize_row
from delta_embed_vl.data.download import (
    CAULDRON_CONFIGS,
    CAULDRON_ID,
    WIKIPEDIA_CONFIG,
    WIKIPEDIA_ID,
)
from delta_embed_vl.data.prepare import prepare_dataset
from delta_embed_vl.settings import Settings

hf_logging.set_verbosity_error()

_DEFAULT_TEST_ROWS_PER_DATASET = 25
_TEST_BATCH_SIZE = 4
_TEST_MAX_LENGTH = 4096
_RAW_SUBSET_CACHE_DIR = Settings().data_dir / "check_dataloader_cache"


def _stream_subset_cache_dir(
    dataset_id: str,
    config: str,
) -> Path:
    safe_dataset_id = dataset_id.replace("/", "--")
    safe_config = config.replace("/", "--")
    return _RAW_SUBSET_CACHE_DIR / safe_dataset_id / safe_config


def _is_saved_dataset(path: Path) -> bool:
    return (path / "dataset_info.json").exists() or (path / "state.json").exists()


def _save_subset_cache(cache_dir: Path, dataset: Dataset) -> None:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    dataset.save_to_disk(str(temp_dir))
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    temp_dir.replace(cache_dir)


def _load_streaming_subset(
    dataset_id: str,
    config: str,
    *,
    rows: int,
) -> Dataset:
    cache_dir = _stream_subset_cache_dir(dataset_id, config)
    cached_dataset: Dataset | None = None
    cached_rows = 0

    if _is_saved_dataset(cache_dir):
        cached_dataset = Dataset.load_from_disk(str(cache_dir))
        cached_rows = len(cached_dataset)

    if cached_dataset is not None:
        if cached_rows >= rows:
            print(
                "raw_subset_cache_hit",
                {
                    "dataset_id": dataset_id,
                    "config": config,
                    "requested_rows": rows,
                    "cached_rows": cached_rows,
                },
            )
            if cached_rows == rows:
                return cached_dataset
            return cached_dataset.select(range(rows))

        print(
            "raw_subset_cache_extend",
            {
                "dataset_id": dataset_id,
                "config": config,
                "requested_rows": rows,
                "cached_rows": cached_rows,
                "missing_rows": rows - cached_rows,
            },
        )
    else:
        print(
            "raw_subset_cache_miss",
            {
                "dataset_id": dataset_id,
                "config": config,
                "requested_rows": rows,
            },
        )

    stream = load_dataset(
        dataset_id,
        config,
        split="train",
        streaming=True,
    )
    if dataset_id == CAULDRON_ID:
        stream = stream.cast_column(
            "images",
            cast(Any, Sequence(Image(decode=False))),
        )

    if cached_rows > 0:
        stream = stream.skip(cached_rows)

    missing_rows = rows - cached_rows
    fetched_rows = list(stream.take(missing_rows))
    if not fetched_rows and cached_dataset is None:
        raise ValueError(
            f"Failed to load any rows for dataset_id={dataset_id} config={config}."
        )

    if cached_dataset is None:
        dataset = Dataset.from_list(fetched_rows)
    elif not fetched_rows:
        dataset = cached_dataset
    else:
        dataset = concatenate_datasets(
            [
                cached_dataset,
                Dataset.from_list(fetched_rows),
            ]
        )

    _save_subset_cache(cache_dir, dataset)
    print(
        "raw_subset_cache_saved",
        {
            "dataset_id": dataset_id,
            "config": config,
            "cached_rows": len(dataset),
            "requested_rows": rows,
            "path": str(cache_dir),
        },
    )
    if len(dataset) == rows:
        return dataset
    return dataset.select(range(rows))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=_DEFAULT_TEST_ROWS_PER_DATASET,
        help=(
            "Rows to load per dataset. Uses and grows the local raw subset cache "
            f"from a default of {_DEFAULT_TEST_ROWS_PER_DATASET}."
        ),
    )
    return parser.parse_args()


def _resolve_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError("--limit must be at least 1.")
    return limit


def get_test_data(*, rows: int) -> list[tuple[str, Dataset]]:
    datasets = [
        (
            "wikipedia",
            _load_streaming_subset(
                WIKIPEDIA_ID,
                WIKIPEDIA_CONFIG,
                rows=rows,
            ),
        )
    ]
    datasets.extend(
        (
            f"cauldron/{config}",
            _load_streaming_subset(
                CAULDRON_ID,
                config,
                rows=rows,
            ),
        )
        for config in CAULDRON_CONFIGS
    )
    return datasets


def _summarize_tensor_batch(batch: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        key: tuple(value.shape)
        for key, value in batch.items()
        if hasattr(value, "shape")
    }


def main():
    args = _parse_args()
    limit = _resolve_limit(args.limit)

    print(
        "check_dataloader_config",
        {
            "requested_limit": args.limit,
            "effective_limit": limit,
            "default_limit": _DEFAULT_TEST_ROWS_PER_DATASET,
            "batch_size": _TEST_BATCH_SIZE,
            "max_length": _TEST_MAX_LENGTH,
        },
    )

    raw_test_data = get_test_data(rows=limit)
    print("dataset_sizes", [len(dataset) for _, dataset in raw_test_data])
    prepared_test_data = [
        (
            source,
            prepare_dataset(
                source,
                raw_dataset,
                student_max_length=_TEST_MAX_LENGTH,
            ),
        )
        for source, raw_dataset in raw_test_data
    ]
    raw_datasets_by_source = {
        source: raw_dataset for source, raw_dataset in raw_test_data
    }
    total_normalized_samples = 0

    for name, dataset in prepared_test_data:
        total_samples = 0
        zero_sample_rows = 0
        query_samples = 0
        document_samples = 0
        image_only_documents = 0
        max_samples_per_row = 0

        raw_dataset = raw_datasets_by_source.get(name)
        for row_idx, row in enumerate(dataset):
            sample = normalize_row(
                row,
                raw_dataset=raw_dataset if name.startswith("cauldron/") else None,
                row_index=row_idx,
            )
            sample_count = 1
            total_samples += sample_count
            total_normalized_samples += sample_count
            max_samples_per_row = max(max_samples_per_row, sample_count)

            if sample.role == "query":
                query_samples += 1
            else:
                document_samples += 1
                if sample.text is None and sample.image is not None:
                    image_only_documents += 1

        print(
            name,
            {
                "prepared_rows": len(dataset),
                "total_samples": total_samples,
                "zero_sample_rows": zero_sample_rows,
                "query_samples": query_samples,
                "document_samples": document_samples,
                "image_only_documents": image_only_documents,
                "max_samples_per_row": max_samples_per_row,
            },
        )

    loader = create_multimodal_dataloader(
        datasets=prepared_test_data,
        batch_size=_TEST_BATCH_SIZE,
        shuffle=False,
        max_length=_TEST_MAX_LENGTH,
        raw_datasets_by_source=raw_datasets_by_source,
    )

    batch_count = 0
    processed_samples = 0
    first_batch_shapes: dict[str, tuple[int, ...]] | None = None

    try:
        for batch_count, batch in enumerate(loader, start=1):
            if first_batch_shapes is None:
                first_batch_shapes = _summarize_tensor_batch(batch)

            processed_samples += int(batch["input_ids"].shape[0])
    except Exception:
        print(
            "loader_failed",
            {
                "completed_batches": batch_count,
                "processed_samples": processed_samples,
                "expected_samples": total_normalized_samples,
            },
        )
        raise

    print(
        "loader_summary",
        {
            "source_datasets": len(prepared_test_data),
            "rows_per_dataset": limit,
            "total_rows": sum(len(dataset) for _, dataset in prepared_test_data),
            "total_normalized_samples": total_normalized_samples,
            "batch_size": _TEST_BATCH_SIZE,
            "max_length": _TEST_MAX_LENGTH,
            "batch_count": batch_count,
            "processed_samples": processed_samples,
            "first_batch_shapes": first_batch_shapes,
        },
    )


if __name__ == "__main__":
    main()
