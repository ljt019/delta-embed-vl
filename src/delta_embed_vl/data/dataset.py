from dataclasses import dataclass
from typing import Literal

from datasets import Dataset as HFDataset
from PIL import Image

from delta_embed_vl.data.download import load_raw_cauldron
from delta_embed_vl.data.prepare import load_prepared_cauldron_embedding_input


@dataclass(frozen=True)
class NormalizedSample:
    source: str
    role: Literal["query", "document"]
    text: str | None
    image: Image.Image | None


@dataclass(frozen=True)
class SampleRef:
    source: str
    row_idx: int


class MultimodalEmbeddingDataset:
    def __init__(
        self,
        datasets: list[tuple[str, HFDataset]],
        *,
        raw_datasets_by_source: dict[str, HFDataset] | None = None,
    ) -> None:
        self.datasets = {source: dataset for source, dataset in datasets}
        self.raw_datasets_by_source = dict(raw_datasets_by_source or {})
        self.sample_refs: list[SampleRef] = []

        for source, dataset in datasets:
            for row_idx in range(len(dataset)):
                self.sample_refs.append(
                    SampleRef(
                        source=source,
                        row_idx=row_idx,
                    )
                )

    def __len__(self) -> int:
        return len(self.sample_refs)

    def __getitem__(self, idx) -> NormalizedSample:
        sample_ref = self.sample_refs[idx]
        row = self.datasets[sample_ref.source][sample_ref.row_idx]
        return normalize_row(
            row,
            raw_dataset=self._get_raw_dataset(sample_ref.source),
            row_index=sample_ref.row_idx,
        )

    def _get_raw_dataset(self, source: str) -> HFDataset | None:
        if source == "wikipedia":
            return None

        raw_dataset = self.raw_datasets_by_source.get(source)
        if raw_dataset is not None:
            return raw_dataset

        if source.startswith("cauldron/"):
            config = source.split("/", maxsplit=1)[1]
            raw_dataset = load_raw_cauldron(config)
            self.raw_datasets_by_source[source] = raw_dataset
            return raw_dataset

        raise ValueError(f"Unsupported source: {source}")


def normalize_row(
    row: dict,
    *,
    raw_dataset: HFDataset | None = None,
    row_index: int | None = None,
) -> NormalizedSample:
    source = row["source"]
    role = row["role"]
    text = row.get("text") or None

    if source == "wikipedia":
        return NormalizedSample(
            source=source,
            role=role,
            text=text,
            image=None,
        )

    if source.startswith("cauldron/"):
        if raw_dataset is None:
            raise ValueError(f"{source} requires raw_dataset to resolve image.")

        config = source.split("/", maxsplit=1)[1]
        embedding_input = load_prepared_cauldron_embedding_input(
            raw_dataset,
            row,
            config=config,
            row_index=row_index,
        )
        return NormalizedSample(
            source=source,
            role=role,
            text=embedding_input.text,
            image=embedding_input.image,
        )

    raise ValueError(f"Unsupported source: {source}")
