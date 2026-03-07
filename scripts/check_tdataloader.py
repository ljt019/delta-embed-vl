from typing import Any, cast

from datasets import Dataset, Image, Sequence, load_dataset
from huggingface_hub.utils import logging as hf_logging

from delta_embed_vl.data.download import (
    CAULDRON_CONFIGS,
    CAULDRON_ID,
    WIKIPEDIA_CONFIG,
    WIKIPEDIA_ID,
)
from delta_embed_vl.data.tdataloader import create_multimodal_dataloader
from delta_embed_vl.data.tdataset import normalize_row

hf_logging.set_verbosity_error()

_TEST_ROWS_PER_DATASET = 25
_TEST_BATCH_SIZE = 4


def _load_streaming_subset(
    dataset_id: str,
    config: str,
    *,
    rows: int,
) -> Dataset:
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
    return Dataset.from_list(list(stream.take(rows)))


def get_test_data() -> list[tuple[str, Dataset]]:
    datasets = [
        (
            "wikipedia",
            _load_streaming_subset(
                WIKIPEDIA_ID,
                WIKIPEDIA_CONFIG,
                rows=_TEST_ROWS_PER_DATASET,
            ),
        )
    ]
    datasets.extend(
        (
            f"cauldron/{config}",
            _load_streaming_subset(
                CAULDRON_ID,
                config,
                rows=_TEST_ROWS_PER_DATASET,
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
    test_data = get_test_data()
    print("dataset_sizes", [len(dataset) for _, dataset in test_data])
    total_normalized_samples = 0

    for name, dataset in test_data:
        total_samples = 0
        zero_sample_rows = 0
        query_samples = 0
        document_samples = 0
        image_only_documents = 0
        max_samples_per_row = 0

        for row in dataset:
            samples = normalize_row(row, source=name)
            sample_count = len(samples)
            total_samples += sample_count
            total_normalized_samples += sample_count
            max_samples_per_row = max(max_samples_per_row, sample_count)

            if sample_count == 0:
                zero_sample_rows += 1

            for sample in samples:
                if sample.role == "query":
                    query_samples += 1
                else:
                    document_samples += 1
                    if sample.text is None and sample.image is not None:
                        image_only_documents += 1

        print(
            name,
            {
                "rows": len(dataset),
                "total_samples": total_samples,
                "zero_sample_rows": zero_sample_rows,
                "query_samples": query_samples,
                "document_samples": document_samples,
                "image_only_documents": image_only_documents,
                "max_samples_per_row": max_samples_per_row,
            },
        )

    loader = create_multimodal_dataloader(
        datasets=test_data,
        batch_size=_TEST_BATCH_SIZE,
        shuffle=False,
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
            "source_datasets": len(test_data),
            "rows_per_dataset": _TEST_ROWS_PER_DATASET,
            "total_rows": sum(len(dataset) for _, dataset in test_data),
            "total_normalized_samples": total_normalized_samples,
            "batch_size": _TEST_BATCH_SIZE,
            "batch_count": batch_count,
            "processed_samples": processed_samples,
            "first_batch_shapes": first_batch_shapes,
        },
    )


if __name__ == "__main__":
    main()
