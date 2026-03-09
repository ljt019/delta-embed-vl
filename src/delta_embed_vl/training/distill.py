import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import wandb
from datasets import Dataset
from torch.optim import AdamW
from transformers import Qwen3VLProcessor, get_cosine_schedule_with_warmup

from delta_embed_vl.data.download import CAULDRON_CONFIGS, load_raw_cauldron
from delta_embed_vl.data.prepare import (
    load_prepared_cauldron_embedding_input,
    prepare_cauldron,
    prepare_wikipedia,
)
from delta_embed_vl.model.embedding_inputs import (
    EmbeddingInput,
    build_student_batch,
    is_student_overlength_error,
)
from delta_embed_vl.model.pooling import last_token_pool, normalize
from delta_embed_vl.model.student import load_student, save_projection_head
from delta_embed_vl.model.teacher import TeacherEmbedder, load_teacher
from delta_embed_vl.settings import Settings
from delta_embed_vl.training.losses import cosine_distill_loss

logger = logging.getLogger(__name__)
_PROGRESS_LOG_INTERVAL_S = 30.0
_SETTINGS = Settings()


@dataclass(frozen=True)
class PreparedSource:
    name: str
    load_example: Callable[[int], EmbeddingInput]
    start: int
    stop: int


def _auto_teacher_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda:0"


def _auto_student_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if torch.cuda.device_count() > 1:
        return "cuda:1"
    return "cuda:0"


def _build_sources(
    *,
    limit: int | None = None,
    max_length: int = _SETTINGS.student_max_length,
) -> tuple[list[PreparedSource], int]:
    """Load processed data and resolve each global index back to an example."""
    wiki_ds = prepare_wikipedia(limit=limit)
    cauldron_datasets = prepare_cauldron(
        limit=limit,
        student_max_length=max_length,
    )

    sources: list[PreparedSource] = []
    next_start = 0

    def add_source(
        name: str,
        *,
        count: int,
        load_example: Callable[[int], EmbeddingInput],
    ) -> None:
        nonlocal next_start
        if count == 0:
            logger.info("Skipping empty %s", name)
            return
        sources.append(
            PreparedSource(
                name=name,
                load_example=load_example,
                start=next_start,
                stop=next_start + count,
            )
        )
        next_start += count

    def load_wikipedia_example(local_index: int) -> EmbeddingInput:
        row = wiki_ds[local_index]
        return EmbeddingInput(text=row["text"] or None)

    add_source(
        "wikipedia",
        count=len(wiki_ds),
        load_example=load_wikipedia_example,
    )

    def make_cauldron_loader(
        config: str, dataset: Dataset
    ) -> Callable[[int], EmbeddingInput]:
        raw_dataset: Dataset | None = None

        def load_example(local_index: int) -> EmbeddingInput:
            nonlocal raw_dataset
            if raw_dataset is None:
                raw_dataset = load_raw_cauldron(config, limit=limit)

            row = dataset[local_index]
            assert raw_dataset is not None
            return load_prepared_cauldron_embedding_input(
                raw_dataset,
                row,
                config=config,
                row_index=local_index,
            )

        return load_example

    for config, ds in zip(CAULDRON_CONFIGS, cauldron_datasets, strict=False):
        add_source(
            f"cauldron/{config}",
            count=len(ds),
            load_example=make_cauldron_loader(config, ds),
        )

    return sources, next_start


def _resolve_source(
    sources: list[PreparedSource], global_index: int
) -> tuple[PreparedSource, int]:
    for source in sources:
        if global_index < source.stop:
            return source, global_index - source.start
    raise IndexError(f"Global index {global_index} out of range")


def _collate(
    batch: list[EmbeddingInput],
    processor: Qwen3VLProcessor,
    max_length: int,
) -> dict[str, torch.Tensor]:
    encoded = build_student_batch(
        processor,
        batch,
        max_length=max_length,
    )
    return {key: value for key, value in encoded.items()}


def _summarize_origins(origins: list[str]) -> str:
    unique_origins = sorted(set(origins))
    if len(unique_origins) <= 3:
        return ", ".join(unique_origins)
    preview = ", ".join(unique_origins[:3])
    return f"{preview}, +{len(unique_origins) - 3} more"


def _collate_with_fallback(
    batch: list[EmbeddingInput],
    origins: list[str],
    processor: Qwen3VLProcessor,
    max_length: int,
) -> tuple[list[EmbeddingInput], dict[str, torch.Tensor] | None, list[str]]:
    try:
        return batch, _collate(batch, processor, max_length), []
    except ValueError as exc:
        if not is_student_overlength_error(exc):
            raise

    valid_samples: list[EmbeddingInput] = []
    skipped_origins: list[str] = []
    for sample, origin in zip(batch, origins, strict=False):
        try:
            _collate([sample], processor, max_length)
        except ValueError as sample_exc:
            if not is_student_overlength_error(sample_exc):
                raise
            skipped_origins.append(origin)
            continue
        valid_samples.append(sample)

    if not valid_samples:
        return [], None, skipped_origins
    return (
        valid_samples,
        _collate(valid_samples, processor, max_length),
        skipped_origins,
    )


def _embed_teacher_targets(
    teacher: TeacherEmbedder,
    batch: list[EmbeddingInput],
    *,
    teacher_batch_size: int,
    target_device: str,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(batch), teacher_batch_size):
        stop = start + teacher_batch_size
        embeddings = teacher.embed(batch[start:stop])
        chunks.append(embeddings.to(device=target_device, non_blocking=True))
    return torch.cat(chunks, dim=0)


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def _raise_if_nonfinite(
    tensor: torch.Tensor,
    *,
    name: str,
    batch_ordinal: int,
    batches_per_epoch: int,
    origins: list[str],
) -> None:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    if bool(finite_mask.all().item()):
        return

    nonfinite_count = detached.numel() - int(finite_mask.sum().item())
    message = (
        f"{name} became non-finite in training batch {batch_ordinal}/{batches_per_epoch}; "
        f"shape={tuple(detached.shape)} dtype={detached.dtype} device={detached.device} "
        f"nonfinite={nonfinite_count}/{detached.numel()}"
    )
    if origins:
        message += f" origins={_summarize_origins(origins)}"
    if bool(finite_mask.any().item()):
        finite_values = detached[finite_mask]
        message += (
            f" finite_min={float(finite_values.min().item()):.6g}"
            f" finite_max={float(finite_values.max().item()):.6g}"
        )
    raise FloatingPointError(message)


def train(
    *,
    limit: int | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = _SETTINGS.student_max_length,
    grad_accum_steps: int = 1,
    save_dir: str = "checkpoints",
    teacher_device: str | None = None,
    student_device: str | None = None,
    teacher_batch_size: int | None = None,
) -> None:
    """Run local teacher-student cosine distillation on multimodal data."""
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1.")

    resolved_teacher_device = teacher_device or _auto_teacher_device()
    resolved_student_device = student_device or _auto_student_device()
    resolved_teacher_batch_size = teacher_batch_size or batch_size
    if resolved_teacher_batch_size < 1:
        raise ValueError("teacher_batch_size must be at least 1.")

    logger.info(
        "Devices: teacher=%s student=%s",
        resolved_teacher_device,
        resolved_student_device,
    )

    logger.info("Loading training data")
    sources, n = _build_sources(limit=limit, max_length=max_length)
    if n == 0:
        raise ValueError("No training samples found.")
    logger.info("Training on %d samples for %d epochs", n, epochs)

    wandb_run = None
    if _SETTINGS.wandb_project is not None:
        wandb_run = wandb.init(
            project=_SETTINGS.wandb_project,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_length": max_length,
                "student_model": _SETTINGS.student_model,
                "teacher_model": _SETTINGS.teacher_model,
                "limit": limit,
                "grad_accum_steps": grad_accum_steps,
            },
        )

    try:
        logger.info("Loading frozen teacher model")
        teacher = load_teacher(device=resolved_teacher_device)

        logger.info("Loading student model")
        model, processor, projection_head = load_student(
            device=resolved_student_device,
            output_dim=teacher.output_dim,
        )
        model.train()
        projection_head.train()

        indices = np.arange(n, dtype=np.int64)
        batches_per_epoch = math.ceil(n / batch_size)
        total_steps = math.ceil((batches_per_epoch * epochs) / grad_accum_steps)
        warmup_steps = int(total_steps * warmup_ratio)

        optimizer = AdamW(
            list(model.parameters()) + list(projection_head.parameters()),
            lr=lr,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        global_step = 0
        optimizer.zero_grad(set_to_none=True)
        total_skipped_overlength_samples = 0
        total_skipped_overlength_batches = 0
        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            num_batches = 0
            epoch_start_s = time.monotonic()
            last_progress_log_s = epoch_start_s
            skipped_overlength_samples = 0
            skipped_overlength_batches = 0

            for batch_start in range(0, n, batch_size):
                batch_start_s = time.monotonic()
                batch_end = min(batch_start + batch_size, n)
                batch_ordinal = (batch_start // batch_size) + 1
                batch_indices = indices[batch_start:batch_end]
                batch_samples: list[EmbeddingInput] = []
                batch_origins: list[str] = []
                for global_index in batch_indices:
                    source, local_index = _resolve_source(sources, int(global_index))
                    batch_samples.append(source.load_example(local_index))
                    batch_origins.append(source.name)

                batch_samples, batch_data, skipped_origins = _collate_with_fallback(
                    batch_samples,
                    batch_origins,
                    processor,
                    max_length,
                )
                if skipped_origins:
                    skipped_count = len(skipped_origins)
                    skipped_overlength_samples += skipped_count
                    total_skipped_overlength_samples += skipped_count
                    if batch_data is None:
                        skipped_overlength_batches += 1
                        total_skipped_overlength_batches += 1
                        logger.warning(
                            "Skipping batch %d/%d after all %d samples exceeded student max_length=%d (%s)",
                            batch_ordinal,
                            batches_per_epoch,
                            skipped_count,
                            max_length,
                            _summarize_origins(skipped_origins),
                        )
                        continue
                    logger.warning(
                        "Recovered batch %d/%d by skipping %d student-overlength samples; kept=%d (%s)",
                        batch_ordinal,
                        batches_per_epoch,
                        skipped_count,
                        len(batch_samples),
                        _summarize_origins(skipped_origins),
                    )
                if batch_data is None:
                    continue
                teacher_target = _embed_teacher_targets(
                    teacher,
                    batch_samples,
                    teacher_batch_size=resolved_teacher_batch_size,
                    target_device=resolved_student_device,
                )
                _raise_if_nonfinite(
                    teacher_target,
                    name="teacher_target",
                    batch_ordinal=batch_ordinal,
                    batches_per_epoch=batches_per_epoch,
                    origins=batch_origins,
                )

                model_inputs = {
                    key: value.to(resolved_student_device)
                    for key, value in batch_data.items()
                }

                outputs = model(**model_inputs)
                _raise_if_nonfinite(
                    outputs.last_hidden_state,
                    name="student_last_hidden_state",
                    batch_ordinal=batch_ordinal,
                    batches_per_epoch=batches_per_epoch,
                    origins=batch_origins,
                )
                attention_mask = model_inputs["attention_mask"]
                pooled = last_token_pool(outputs.last_hidden_state, attention_mask)
                _raise_if_nonfinite(
                    pooled,
                    name="student_pooled",
                    batch_ordinal=batch_ordinal,
                    batches_per_epoch=batches_per_epoch,
                    origins=batch_origins,
                )
                projected = projection_head(pooled).float()
                _raise_if_nonfinite(
                    projected,
                    name="student_projected",
                    batch_ordinal=batch_ordinal,
                    batches_per_epoch=batches_per_epoch,
                    origins=batch_origins,
                )
                student_emb = normalize(projected)
                _raise_if_nonfinite(
                    student_emb,
                    name="student_embedding",
                    batch_ordinal=batch_ordinal,
                    batches_per_epoch=batches_per_epoch,
                    origins=batch_origins,
                )

                loss = (
                    cosine_distill_loss(student_emb, teacher_target) / grad_accum_steps
                )
                _raise_if_nonfinite(
                    loss.reshape(1),
                    name="distill_loss",
                    batch_ordinal=batch_ordinal,
                    batches_per_epoch=batches_per_epoch,
                    origins=batch_origins,
                )
                loss.backward()

                num_batches += 1
                should_step = (num_batches % grad_accum_steps == 0) or (batch_end == n)
                if should_step:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                epoch_loss += loss.item() * grad_accum_steps
                now_s = time.monotonic()
                batch_elapsed_s = now_s - batch_start_s

                if wandb_run is not None and should_step:
                    wandb.log(
                        {
                            "train/loss": loss.item() * grad_accum_steps,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/batch_time_s": batch_elapsed_s,
                        },
                        step=global_step,
                    )

                if (
                    num_batches == 1
                    or (now_s - last_progress_log_s) >= _PROGRESS_LOG_INTERVAL_S
                ):
                    elapsed_s = now_s - epoch_start_s
                    avg_batch_s = elapsed_s / max(num_batches, 1)
                    remaining_batches = max(batches_per_epoch - num_batches, 0)
                    eta_s = avg_batch_s * remaining_batches
                    logger.info(
                        "Epoch %d/%d  %d/%d batches (%.1f%%)  loss=%.6f avg=%.6f  batch=%.2fs  elapsed=%s  eta=%s",
                        epoch + 1,
                        epochs,
                        num_batches,
                        batches_per_epoch,
                        (100.0 * num_batches) / max(batches_per_epoch, 1),
                        loss.item() * grad_accum_steps,
                        epoch_loss / max(num_batches, 1),
                        batch_elapsed_s,
                        _format_duration(elapsed_s),
                        _format_duration(eta_s),
                    )
                    last_progress_log_s = now_s

            avg_loss = epoch_loss / max(num_batches, 1)
            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch/avg_loss": avg_loss,
                        "epoch/skipped_samples": skipped_overlength_samples,
                    },
                    step=global_step,
                )
            logger.info(
                "Epoch %d/%d  loss=%.6f  lr=%.2e  steps=%d  skipped_overlength_samples=%d  skipped_overlength_batches=%d",
                epoch + 1,
                epochs,
                avg_loss,
                scheduler.get_last_lr()[0],
                global_step,
                skipped_overlength_samples,
                skipped_overlength_batches,
            )

        out_path = Path(save_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_path))
        save_projection_head(projection_head, out_path)
        processor.save_pretrained(str(out_path))
        logger.info(
            "Training done: skipped_overlength_samples=%d skipped_overlength_batches=%d",
            total_skipped_overlength_samples,
            total_skipped_overlength_batches,
        )
        logger.info("Saved checkpoint to %s", out_path)
    finally:
        if wandb_run is not None:
            wandb.finish()
