from datasets import Dataset, load_dataset

from delta_embed_vl.settings import Settings

### Public

WIKIPEDIA_ID = "wikimedia/wikipedia"
CAULDRON_ID = "HuggingFaceM4/the_cauldron"

CAULDRON_CONFIGS = [
    # General VQA
    "vqav2",  # 83k images, open-ended questions on COCO photos | CC BY 4.0
    "cocoqa",  # 46k images, auto-generated QA from COCO captions | CC BY 4.0
    "visual7w",  # 14k images, grounded QA with spatial reasoning | CC BY 4.0
    "aokvqa",  # 17k images, QA requiring outside knowledge | CC BY 4.0
    "tallyqa",  # 99k images, counting questions | Apache 2.0
    "okvqa",  # 9k images, knowledge-based visual QA | CC BY 4.0
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

_DATASET_REGISTRY = {
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", "train"),
    **{
        f"cauldron/{config}": ("HuggingFaceM4/the_cauldron", config, "train")
        for config in CAULDRON_CONFIGS
    },
}


def _load_raw_data(name: str, *, limit: int | None = None) -> Dataset:
    """Load a raw dataset. If limit is set, stream and take only N rows."""
    if name not in _DATASET_REGISTRY:
        supported_names = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset name: {name}. Supported: {supported_names}")

    dataset_id, config, split = _DATASET_REGISTRY[name]

    if limit is not None:
        stream = load_dataset(dataset_id, config, split=split, streaming=True)
        return Dataset.from_list(list(stream.take(limit)))

    return load_dataset(
        dataset_id,
        config,
        split=split,
        cache_dir=str(_RAW_DATA_DIR / name),
    )
