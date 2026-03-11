import json
import logging
import shutil
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, Image, Sequence, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)

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


def load_raw_wikipedia(*, limit: int | None = None, no_stream: bool = False) -> Dataset:
    return _load_raw_data("wikipedia", limit=limit, no_stream=no_stream)


def load_raw_cauldron(
    config: str, *, limit: int | None = None, no_stream: bool = False
) -> Dataset:
    """Load one raw Cauldron config by name, e.g. `vqav2`."""
    if config not in CAULDRON_CONFIGS:
        supported_configs = ", ".join(CAULDRON_CONFIGS)
        raise ValueError(
            f"Unknown Cauldron config: {config}. Supported: {supported_configs}"
        )
    return _load_raw_data(f"cauldron/{config}", limit=limit, no_stream=no_stream)


### Private

_RAW_DATA_DIR = Path("data/raw")
_CACHE_META_FILENAME = "_cache_meta.json"

_DATASET_REGISTRY = {
    "wikipedia": ("wikimedia/wikipedia", WIKIPEDIA_CONFIG, "train"),
    **{
        f"cauldron/{config}": ("HuggingFaceM4/the_cauldron", config, "train")
        for config in CAULDRON_CONFIGS
    },
}


def _load_raw_data(
    name: str, *, limit: int | None = None, no_stream: bool = False
) -> Dataset:
    """Load raw data from a single canonical local cache under data/raw/<source>."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1.")
    if no_stream:
        return _load_full_then_select(name, rows=limit)
    return _load_or_extend_raw_cache(name, rows=limit)


def _get_dataset_entry(name: str) -> tuple[str, str, str]:
    if name not in _DATASET_REGISTRY:
        supported_names = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset name: {name}. Supported: {supported_names}")
    return _DATASET_REGISTRY[name]


def _load_full_then_select(name: str, *, rows: int | None) -> Dataset:
    """Download the full dataset, cache it, then select requested rows."""
    cache_dir = _raw_cache_dir(name)
    if _is_saved_dataset(cache_dir) and _read_cache_complete(cache_dir):
        dataset = Dataset.load_from_disk(str(cache_dir))
    else:
        logger.debug("no_stream full download name=%s", name)
        dataset = _materialize_full_raw_data(name)
        _save_raw_cache(cache_dir, dataset, complete=True)
    if rows is not None and len(dataset) > rows:
        return dataset.select(range(rows))
    return dataset


def _load_or_extend_raw_cache(name: str, *, rows: int | None) -> Dataset:
    cache_dir = _raw_cache_dir(name)
    cached_dataset: Dataset | None = None
    cached_rows = 0
    cache_complete = False

    if _is_saved_dataset(cache_dir):
        cached_dataset = Dataset.load_from_disk(str(cache_dir))
        cached_rows = len(cached_dataset)
        cache_complete = _read_cache_complete(cache_dir)

    if rows is None:
        if cached_dataset is not None and cache_complete:
            logger.debug(
                "raw_cache_hit name=%s requested=all cached=%d path=%s",
                name,
                cached_rows,
                cache_dir,
            )
            return cached_dataset
        logger.debug(
            "raw_cache_materialize_full name=%s cached=%d path=%s",
            name,
            cached_rows,
            cache_dir,
        )
        dataset = _materialize_full_raw_data(name)
        _save_raw_cache(cache_dir, dataset, complete=True)
        return dataset

    if cached_dataset is not None and cached_rows >= rows:
        logger.debug(
            "raw_cache_hit name=%s requested=%d cached=%d path=%s",
            name,
            rows,
            cached_rows,
            cache_dir,
        )
        return (
            cached_dataset
            if cached_rows == rows
            else cached_dataset.select(range(rows))
        )

    if cached_dataset is not None and cache_complete:
        logger.debug(
            "raw_cache_hit_exhausted name=%s requested=%d cached=%d path=%s",
            name,
            rows,
            cached_rows,
            cache_dir,
        )
        return cached_dataset

    logger.debug(
        "%s name=%s requested=%d cached=%d path=%s",
        "raw_cache_extend" if cached_dataset is not None else "raw_cache_miss",
        name,
        rows,
        cached_rows,
        cache_dir,
    )

    dataset = _extend_raw_cache(
        name,
        cache_dir=cache_dir,
        cached_dataset=cached_dataset,
        cached_rows=cached_rows,
        rows=rows,
    )
    if len(dataset) <= rows:
        return dataset
    return dataset.select(range(rows))


def _materialize_full_raw_data(name: str) -> Dataset:
    dataset_id, config, split = _get_dataset_entry(name)
    dataset = load_dataset(
        dataset_id,
        config,
        split=split,
    )
    return _cast_cauldron_images(name, dataset)


def _extend_raw_cache(
    name: str,
    *,
    cache_dir: Path,
    cached_dataset: Dataset | None,
    cached_rows: int,
    rows: int,
) -> Dataset:
    dataset_id, config, _ = _get_dataset_entry(name)
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

    cache_complete = len(fetched_rows) < missing_rows
    _save_raw_cache(cache_dir, dataset, complete=cache_complete)
    logger.debug(
        "raw_cache_saved name=%s requested=%d cached=%d complete=%s path=%s",
        name,
        rows,
        len(dataset),
        cache_complete,
        cache_dir,
    )
    return dataset


def _cast_cauldron_images(name: str, dataset: Dataset) -> Dataset:
    if not name.startswith("cauldron/"):
        return dataset
    return dataset.cast_column("images", Sequence(Image(decode=False)))


def _raw_cache_dir(name: str) -> Path:
    return _RAW_DATA_DIR / name


def _is_saved_dataset(path: Path) -> bool:
    return (path / "dataset_info.json").exists() or (path / "state.json").exists()


def _save_raw_cache(cache_dir: Path, dataset: Dataset, *, complete: bool) -> None:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    dataset.save_to_disk(str(temp_dir))
    _write_cache_meta(temp_dir, complete=complete)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    temp_dir.replace(cache_dir)


def _read_cache_complete(path: Path) -> bool:
    meta_path = path / _CACHE_META_FILENAME
    if not meta_path.exists():
        return False
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return bool(metadata.get("complete", False))


def _write_cache_meta(path: Path, *, complete: bool) -> None:
    (path / _CACHE_META_FILENAME).write_text(
        json.dumps({"complete": complete}),
        encoding="utf-8",
    )
