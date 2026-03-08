import shutil
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, Image, Sequence, concatenate_datasets, load_dataset

from delta_embed_vl.settings import Settings

### Public

WIKIPEDIA_ID = "wikimedia/wikipedia"
CAULDRON_ID = "HuggingFaceM4/the_cauldron"

WIKIPEDIA_CONFIG = "20231101.en"
CAULDRON_CONFIGS = [
    # General VQA
    "vqav2",  # 83k images, open-ended questions on COCO photos | CC BY 4.0
    "cocoqa",  # 46k images, auto-generated QA from COCO captions | CC BY 4.0
    "visual7w",  # 14k images, grounded QA with spatial reasoning | CC BY 4.0
    "aokvqa",  # 17k images, QA requiring outside knowledge | CC BY 4.0
    "tallyqa",  # 99k images, counting questions | Apache 2.0
    "vqarad",  # 313 images, radiology/medical visual QA | CC0 1.0
    "vsr",  # 2k images, visual spatial relationship verification | CC BY 4.0
    # Captioning
    "localized_narratives",  # 507k images, detailed region-level descriptions | CC BY 4.0
    "screen2words",  # 16k images, mobile UI screen summaries | MIT
    # OCR / document understanding
    "rendered_text",  # 999k images, synthetic rendered text recognition | Apache 2.0
    "docvqa",  # 10k images, QA on scanned documents | CC BY 4.0
    "textcaps",  # 22k images, captions requiring reading text in images | CC BY 4.0
    "textvqa",  # 22k images, QA requiring reading text in images | CC BY 4.0
    "st_vqa",  # 17k images, scene text visual QA | CC BY 4.0
    "ocrvqa",  # 166k images, QA on book covers via OCR | CC BY 4.0
    "visualmrc",  # 3k images, machine reading comprehension on screenshots | CC BY-SA 4.0
    "infographic_vqa",  # 2k images, QA on infographics | CC BY 4.0
    # Chart / figure understanding
    "chart2text",  # 27k images, chart-to-description generation | MIT
    "dvqa",  # 200k images, QA on bar charts | CC BY 4.0
    "vistext",  # 7k images, data visualization descriptions | CC BY 4.0
    "plotqa",  # 157k images, QA on scientific plots | CC BY 4.0
    "figureqa",  # 100k images, yes/no QA on synthetic figures | MIT
    "mapqa",  # 37k images, QA on choropleth maps | Apache 2.0
    # Table understanding
    "tat_qa",  # 2k images, QA over financial tables + text | MIT
    "hitab",  # 2.5k images, hierarchical table QA | MIT
    "multihiertt",  # 8k images, multi-step reasoning on financial tables | MIT
    "finqa",  # 5k images, numerical reasoning on financial reports | MIT
    "robut_wikisql",  # 75k images, SQL-grounded table QA | MIT
    "robut_sqa",  # 9k images, sequential table QA | MIT
    "robut_wtq",  # 38k images, web table questions | MIT
    # Academic / science
    "ai2d",  # 3k images, science diagram QA | Apache 2.0
]


def download_data(*, limit: int | None = None) -> None:
    """Download all raw datasets. Pass limit=N to only fetch first N rows per dataset."""
    load_raw_wikipedia(limit=limit)
    for config in CAULDRON_CONFIGS:
        load_raw_cauldron(config, limit=limit)


def load_raw_wikipedia(*, limit: int | None = None) -> Dataset:
    return _load_raw_data("wikipedia", limit=limit)


def load_raw_cauldron(config: str, *, limit: int | None = None) -> Dataset:
    """Load one raw Cauldron config by name, e.g. `vqav2`."""
    if config not in CAULDRON_CONFIGS:
        supported_configs = ", ".join(CAULDRON_CONFIGS)
        raise ValueError(
            f"Unknown Cauldron config: {config}. Supported: {supported_configs}"
        )
    return _load_raw_data(f"cauldron/{config}", limit=limit)


### Private

_RAW_DATA_DIR = Settings().data_dir / "raw"
_RAW_SUBSET_CACHE_DIR = Settings().data_dir / "raw_subsets"

_DATASET_REGISTRY = {
    "wikipedia": ("wikimedia/wikipedia", WIKIPEDIA_CONFIG, "train"),
    **{
        f"cauldron/{config}": ("HuggingFaceM4/the_cauldron", config, "train")
        for config in CAULDRON_CONFIGS
    },
}


def _load_raw_data(name: str, *, limit: int | None = None) -> Dataset:
    """Load a raw dataset. If limit is set, stream and cache only the first N rows."""
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        return _load_raw_subset(name, rows=limit)
    return _load_full_raw_data(name)


def _get_dataset_entry(name: str) -> tuple[str, str, str]:
    if name not in _DATASET_REGISTRY:
        supported_names = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset name: {name}. Supported: {supported_names}")
    return _DATASET_REGISTRY[name]


def _load_full_raw_data(name: str) -> Dataset:
    dataset_id, config, split = _get_dataset_entry(name)
    dataset = load_dataset(
        dataset_id,
        config,
        split=split,
        cache_dir=str(_RAW_DATA_DIR / name),
    )
    return _cast_cauldron_images(name, dataset)


def _load_raw_subset(name: str, *, rows: int) -> Dataset:
    dataset_id, config, _ = _get_dataset_entry(name)
    cache_dir = _subset_cache_dir(name)
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
                    "name": name,
                    "requested_rows": rows,
                    "cached_rows": cached_rows,
                    "path": str(cache_dir),
                },
            )
            if cached_rows == rows:
                return cached_dataset
            return cached_dataset.select(range(rows))
        print(
            "raw_subset_cache_extend",
            {
                "name": name,
                "requested_rows": rows,
                "cached_rows": cached_rows,
                "missing_rows": rows - cached_rows,
                "path": str(cache_dir),
            },
        )
    else:
        print(
            "raw_subset_cache_miss",
            {
                "name": name,
                "requested_rows": rows,
                "path": str(cache_dir),
            },
        )

    stream = load_dataset(
        dataset_id,
        config,
        split="train",
        streaming=True,
    )
    if name.startswith("cauldron/"):
        stream = stream.cast_column(
            "images",
            cast(Any, Sequence(Image(decode=False))),
        )

    if cached_rows > 0:
        stream = stream.skip(cached_rows)

    fetched_rows = list(stream.take(rows - cached_rows))
    if not fetched_rows and cached_dataset is None:
        raise ValueError(f"Failed to load any rows for dataset_id={dataset_id} config={config}.")

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
            "name": name,
            "requested_rows": rows,
            "cached_rows": len(dataset),
            "path": str(cache_dir),
        },
    )
    if len(dataset) == rows:
        return dataset
    return dataset.select(range(rows))


def _cast_cauldron_images(name: str, dataset: Dataset) -> Dataset:
    if not name.startswith("cauldron/"):
        return dataset
    return dataset.cast_column("images", Sequence(Image(decode=False)))


def _subset_cache_dir(name: str) -> Path:
    return _RAW_SUBSET_CACHE_DIR / name


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
