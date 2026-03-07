from typing import Any, cast

from datasets import Dataset, Image, Sequence, load_dataset
from huggingface_hub.utils import logging as hf_logging

from delta_embed_vl.data.download import (
    CAULDRON_CONFIGS,
    CAULDRON_ID,
    WIKIPEDIA_CONFIG,
    WIKIPEDIA_ID,
)
from delta_embed_vl.data.tdataloader import normalize_row

hf_logging.set_verbosity_error()

_TEST_ROWS_PER_DATASET = 25


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


def main():
    test_data = get_test_data()
    print("dataset_sizes", [len(dataset) for _, dataset in test_data])
    # loader = create_multimodal_dataloader(
    #    datasets=test_data,
    #    batch_size=4,
    #    shuffle=True,
    # )

    # print(f"source datasets: {len(test_data)}")
    # print(f"total rows: {len(loader.dataset)}")  # typ: ignore[arg-type]
    # print(f"batch_size: {loader.batch_size}")

    for name, dataset in test_data:
        row = dataset[0]
        samples = normalize_row(row, source=name)
        print(name, f"normalized_samples={len(samples)}")
        for sample in samples[:4]:
            print(
                {
                    "source": sample.source,
                    "role": sample.role,
                    "text_preview": (
                        None if sample.text is None else sample.text[:100]
                    ),
                    "has_image": sample.image is not None,
                }
            )
        if len(samples) > 4:
            print({"truncated_samples": len(samples) - 4})


if __name__ == "__main__":
    main()
