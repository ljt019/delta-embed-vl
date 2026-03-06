from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, disable_progress_bars

from delta_embed_vl.data.download import CAULDRON_CONFIGS, load_raw_cauldron
from delta_embed_vl.data.preprocess import (
    load_cauldron_embedding_input,
    preprocess_cauldron_config,
    preprocess_wikipedia,
)
from delta_embed_vl.model.embedding_inputs import (
    EmbeddingInput,
    count_teacher_prompt_tokens,
    teacher_processed_token_lengths,
)
from delta_embed_vl.progress import ProgressBar
from delta_embed_vl.settings import Settings

logger = logging.getLogger(__name__)

_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_NOISY_LOGGERS = (
    "datasets",
    "fsspec",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "urllib3",
)
_DEFAULT_THRESHOLDS = (2048, 3072, 4096, 6144, 8192)
_DEFAULT_OUTPUT = Settings().data_dir / "reports" / "teacher_length_audit.json"


@dataclass(frozen=True)
class LengthRecord:
    dataset: str
    modality: str
    processed_tokens: int
    prompt_tokens: int
    source_index: int | None
    image_index: int | None


@dataclass(frozen=True)
class AuditError:
    dataset: str
    modality: str
    source_index: int | None
    image_index: int | None
    error: str


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT, force=True)
    disable_progress_bars()
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit teacher processor token lengths for text and multimodal samples."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=128,
        help="Raw sample cap per dataset/config. Default: 128.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        choices=CAULDRON_CONFIGS,
        default=CAULDRON_CONFIGS,
        help="Subset of Cauldron configs to audit. Default: all configs.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=list(_DEFAULT_THRESHOLDS),
        help="Token ceilings to report exceedances for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Processor batch size for the audit. Default: 8.",
    )
    parser.add_argument(
        "--skip-wikipedia",
        action="store_true",
        help="Skip auditing processed Wikipedia chunks.",
    )
    parser.add_argument(
        "--skip-cauldron",
        action="store_true",
        help="Skip auditing Cauldron configs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Where to save the JSON report. Default: {_DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()
    if args.skip_wikipedia and args.skip_cauldron:
        parser.error("Cannot skip both Wikipedia and Cauldron.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    return args


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    index = (len(values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(values[low])

    fraction = index - low
    return (values[low] * (1.0 - fraction)) + (values[high] * fraction)


def _distribution(values: list[int], *, thresholds: tuple[int, ...]) -> dict[str, Any]:
    empty_thresholds = {str(limit): {"count": 0, "pct": 0.0} for limit in thresholds}
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "threshold_exceedances": empty_thresholds,
        }

    ordered = sorted(values)
    count = len(ordered)
    exceedance_counts = {
        limit: sum(1 for value in ordered if value > limit) for limit in thresholds
    }
    threshold_exceedances = {
        str(limit): {
            "count": exceedance_counts[limit],
            "pct": round((exceedance_counts[limit] / count) * 100, 2),
        }
        for limit in thresholds
    }
    return {
        "count": count,
        "min": ordered[0],
        "mean": round(sum(ordered) / count, 2),
        "p50": round(_percentile(ordered, 0.50), 2),
        "p95": round(_percentile(ordered, 0.95), 2),
        "p99": round(_percentile(ordered, 0.99), 2),
        "max": ordered[-1],
        "threshold_exceedances": threshold_exceedances,
    }


def _summarize_records(
    records: list[LengthRecord], *, thresholds: tuple[int, ...]
) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "modalities": {},
            "processed_tokens": _distribution([], thresholds=thresholds),
            "prompt_tokens": _distribution([], thresholds=thresholds),
            "processor_overhead_tokens": _distribution([], thresholds=()),
        }

    processed_tokens = [record.processed_tokens for record in records]
    prompt_tokens = [record.prompt_tokens for record in records]
    processor_overhead_tokens = [
        record.processed_tokens - record.prompt_tokens for record in records
    ]
    return {
        "count": len(records),
        "modalities": dict(Counter(record.modality for record in records)),
        "processed_tokens": _distribution(processed_tokens, thresholds=thresholds),
        "prompt_tokens": _distribution(prompt_tokens, thresholds=thresholds),
        "processor_overhead_tokens": _distribution(
            processor_overhead_tokens,
            thresholds=(),
        ),
    }


def _group_summary(
    records: list[LengthRecord], *, thresholds: tuple[int, ...]
) -> dict[str, Any]:
    summary = _summarize_records(records, thresholds=thresholds)
    summary["by_modality"] = {
        modality: _summarize_records(
            [record for record in records if record.modality == modality],
            thresholds=thresholds,
        )
        for modality in sorted({record.modality for record in records})
    }
    return summary


def _serialize_record(record: LengthRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["processor_overhead_tokens"] = (
        record.processed_tokens - record.prompt_tokens
    )
    return payload


def _flush_batch(
    *,
    samples: list[EmbeddingInput],
    metadata: list[tuple[str, str, int | None, int | None]],
    records: list[LengthRecord],
    errors: list[AuditError],
) -> None:
    if not samples:
        return

    try:
        processed_lengths = teacher_processed_token_lengths(samples)
        prompt_lengths = [count_teacher_prompt_tokens(sample) for sample in samples]
    except Exception as exc:
        if len(samples) == 1:
            dataset, modality, source_index, image_index = metadata[0]
            errors.append(
                AuditError(
                    dataset=dataset,
                    modality=modality,
                    source_index=source_index,
                    image_index=image_index,
                    error=str(exc),
                )
            )
            return

        for sample, item in zip(samples, metadata, strict=False):
            _flush_batch(
                samples=[sample],
                metadata=[item],
                records=records,
                errors=errors,
            )
        return

    for item, processed_tokens, prompt_tokens in zip(
        metadata,
        processed_lengths,
        prompt_lengths,
        strict=False,
    ):
        dataset, modality, source_index, image_index = item
        records.append(
            LengthRecord(
                dataset=dataset,
                modality=modality,
                processed_tokens=processed_tokens,
                prompt_tokens=prompt_tokens,
                source_index=source_index,
                image_index=image_index,
            )
        )


def _audit_processed_dataset(
    *,
    label: str,
    dataset: Dataset,
    sample_builder: Callable[
        [dict[str, Any]],
        tuple[EmbeddingInput | None, tuple[str, str, int | None, int | None] | None],
    ],
    batch_size: int,
) -> tuple[list[LengthRecord], list[AuditError]]:
    records: list[LengthRecord] = []
    errors: list[AuditError] = []
    progress = ProgressBar(label=label, total=len(dataset), unit="samples")
    batch_samples: list[EmbeddingInput] = []
    batch_metadata: list[tuple[str, str, int | None, int | None]] = []
    processed_rows = 0

    for row in dataset:
        try:
            sample, metadata = sample_builder(row)
        except Exception as exc:
            errors.append(
                AuditError(
                    dataset=label,
                    modality=str(row.get("modality", "unknown")),
                    source_index=(
                        None
                        if row.get("source_index") is None
                        else int(row["source_index"])
                    ),
                    image_index=(
                        None
                        if row.get("image_index") is None
                        else int(row["image_index"])
                    ),
                    error=str(exc),
                )
            )
            processed_rows += 1
            progress.update(processed_rows, extra=f"errors={len(errors):,}")
            continue

        if sample is None or metadata is None:
            if metadata is not None:
                dataset_name, modality, source_index, image_index = metadata
                errors.append(
                    AuditError(
                        dataset=dataset_name,
                        modality=modality,
                        source_index=source_index,
                        image_index=image_index,
                        error="Could not reconstruct sample from processed dataset.",
                    )
                )
            processed_rows += 1
            progress.update(processed_rows, extra=f"errors={len(errors):,}")
            continue

        batch_samples.append(sample)
        batch_metadata.append(metadata)
        processed_rows += 1
        if len(batch_samples) < batch_size:
            progress.update(processed_rows, extra=f"errors={len(errors):,}")
            continue

        _flush_batch(
            samples=batch_samples,
            metadata=batch_metadata,
            records=records,
            errors=errors,
        )
        batch_samples = []
        batch_metadata = []
        progress.update(processed_rows, extra=f"errors={len(errors):,}")

    if batch_samples:
        _flush_batch(
            samples=batch_samples,
            metadata=batch_metadata,
            records=records,
            errors=errors,
        )

    progress.close(extra=f"errors={len(errors):,}")
    return records, errors


def _audit_wikipedia(
    *, limit: int | None, batch_size: int
) -> tuple[list[LengthRecord], list[AuditError]]:
    dataset = preprocess_wikipedia(limit=limit)
    return _audit_processed_dataset(
        label="audit/wikipedia",
        dataset=dataset,
        sample_builder=lambda row: (
            EmbeddingInput(text=row["text"] or None),
            ("wikipedia", row["modality"], None, None),
        ),
        batch_size=batch_size,
    )


def _audit_cauldron_config(
    config: str,
    *,
    limit: int | None,
    batch_size: int,
) -> tuple[list[LengthRecord], list[AuditError]]:
    dataset = preprocess_cauldron_config(config, limit=limit)
    raw = load_raw_cauldron(config, limit=limit)
    label = f"audit/cauldron/{config}"

    def build_sample(
        row: dict[str, Any],
    ) -> tuple[EmbeddingInput | None, tuple[str, str, int | None, int | None] | None]:
        source_index = int(row["source_index"])
        image_index = int(row["image_index"])
        modality = row["modality"]
        return (
            load_cauldron_embedding_input(raw, row, config=config),
            (
                f"cauldron/{config}",
                modality,
                source_index,
                None if image_index < 0 else image_index,
            ),
        )

    return _audit_processed_dataset(
        label=label,
        dataset=dataset,
        sample_builder=build_sample,
        batch_size=batch_size,
    )


def run_length_audit(
    *,
    limit: int | None = 128,
    configs: tuple[str, ...] = tuple(CAULDRON_CONFIGS),
    thresholds: tuple[int, ...] = _DEFAULT_THRESHOLDS,
    batch_size: int = 8,
    include_wikipedia: bool = True,
    include_cauldron: bool = True,
) -> dict[str, Any]:
    datasets: dict[str, dict[str, Any]] = {}
    all_records: list[LengthRecord] = []
    all_errors: list[AuditError] = []

    if include_wikipedia:
        wiki_records, wiki_errors = _audit_wikipedia(limit=limit, batch_size=batch_size)
        datasets["wikipedia"] = {
            "summary": _group_summary(wiki_records, thresholds=thresholds),
            "errors": [asdict(error) for error in wiki_errors],
        }
        all_records.extend(wiki_records)
        all_errors.extend(wiki_errors)

    if include_cauldron:
        for config in configs:
            label = f"cauldron/{config}"
            config_records, config_errors = _audit_cauldron_config(
                config,
                limit=limit,
                batch_size=batch_size,
            )
            datasets[label] = {
                "summary": _group_summary(config_records, thresholds=thresholds),
                "errors": [asdict(error) for error in config_errors],
            }
            all_records.extend(config_records)
            all_errors.extend(config_errors)

    multimodal_records = [
        record for record in all_records if record.modality in {"image", "text_image"}
    ]
    text_records = [record for record in all_records if record.modality == "text"]
    longest_records = sorted(
        all_records,
        key=lambda record: record.processed_tokens,
        reverse=True,
    )[:10]
    return {
        "settings": {
            "teacher_model": Settings().teacher_model,
            "teacher_max_model_len": Settings().teacher_max_model_len,
            "teacher_prompt_margin_tokens": Settings().teacher_prompt_margin_tokens,
        },
        "audit": {
            "limit": limit,
            "configs": list(configs),
            "thresholds": list(thresholds),
            "batch_size": batch_size,
            "include_wikipedia": include_wikipedia,
            "include_cauldron": include_cauldron,
        },
        "overall": _group_summary(all_records, thresholds=thresholds),
        "overall_text_only": _group_summary(text_records, thresholds=thresholds),
        "overall_multimodal": _group_summary(multimodal_records, thresholds=thresholds),
        "datasets": datasets,
        "top_longest_samples": [
            _serialize_record(record) for record in longest_records
        ],
        "error_count": len(all_errors),
    }


def length_audit_cli() -> None:
    _configure_logging()
    args = _parse_args()
    report = run_length_audit(
        limit=args.limit,
        configs=tuple(args.configs),
        thresholds=tuple(args.thresholds),
        batch_size=args.batch_size,
        include_wikipedia=not args.skip_wikipedia,
        include_cauldron=not args.skip_cauldron,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    multimodal_summary = report["overall_multimodal"]
    logger.info(
        json.dumps(
            {
                "event": "teacher_length_audit_complete",
                "report_path": str(args.output),
                "samples": report["overall"]["count"],
                "multimodal_samples": multimodal_summary["count"],
                "multimodal_p99": multimodal_summary["processed_tokens"]["p99"],
                "multimodal_max": multimodal_summary["processed_tokens"]["max"],
                "errors": report["error_count"],
            }
        )
    )


if __name__ == "__main__":
    length_audit_cli()
