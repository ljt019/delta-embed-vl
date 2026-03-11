import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from datasets import Dataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from delta_embed_vl import configure_logging, resolve_attention, set_seed
from delta_embed_vl.data.build import decode_image, load_training_dataset
from delta_embed_vl.model.pooling import last_token_pool, normalize
from delta_embed_vl.model.student import load_student, save_projection_head
from delta_embed_vl.model.tokenization import (
    DEFAULT_EMBED_INSTRUCTION,
    EmbeddingInput,
    build_student_batch,
    is_student_overlength_error,
)
from delta_embed_vl.settings import Settings

logger = logging.getLogger(__name__)
_PROGRESS_LOG_INTERVAL_S = 30.0
_WANDB_LOG_EVERY_N_STEPS = 20
_SETTINGS = Settings()
_PRE_FORWARD_LOG = Path("last_student_forward.jsonl")


def _auto_student_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if torch.cuda.device_count() > 1:
        return "cuda:1"
    return "cuda:0"


def _summarize_origins(origins: list[str]) -> str:
    unique_origins = sorted(set(origins))
    if len(unique_origins) <= 3:
        return ", ".join(unique_origins)
    preview = ", ".join(unique_origins[:3])
    return f"{preview}, +{len(unique_origins) - 3} more"


def _row_to_embedding_input(row: dict[str, Any]) -> EmbeddingInput:
    return EmbeddingInput(
        text=row.get("text") or None,
        image=decode_image(row.get("image_bytes")),
        instruction=row.get("instruction") or DEFAULT_EMBED_INSTRUCTION,
    )


def _collate_rows(
    rows: list[dict[str, Any]],
    *,
    processor,
    max_length: int,
) -> tuple[list[str], dict[str, torch.Tensor], torch.Tensor]:
    samples = [_row_to_embedding_input(row) for row in rows]
    origins = [str(row["source"]) for row in rows]
    try:
        encoded = build_student_batch(
            processor,
            samples,
            max_length=max_length,
        )
    except ValueError as exc:
        if not is_student_overlength_error(exc):
            raise
        raise ValueError(
            "Prepared dataset contains samples that exceed the current --max-length. "
            "Rebuild the dataset with `prepare-data --max-length` set to this value."
        ) from exc

    teacher_arrays = np.asarray(
        [row["teacher_embedding"] for row in rows],
        dtype=np.float32,
    )
    teacher_targets = torch.from_numpy(teacher_arrays)
    return origins, {key: value for key, value in encoded.items()}, teacher_targets


def _get_teacher_embedding_dim(dataset: Dataset) -> int:
    embedding = dataset[0]["teacher_embedding"]
    return len(embedding)


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


def _log_pre_forward(
    *,
    epoch: int,
    batch_ordinal: int,
    batches_per_epoch: int,
    origins: list[str],
    model_inputs: dict[str, torch.Tensor],
) -> None:
    record = {
        "epoch": epoch,
        "batch": f"{batch_ordinal}/{batches_per_epoch}",
        "origins": origins,
        "shapes": {key: list(value.shape) for key, value in model_inputs.items()},
    }
    with _PRE_FORWARD_LOG.open("w") as file:
        json.dump(record, file)
        file.write("\n")
        file.flush()
        sys.stdout.flush()


def cosine_distill_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
) -> torch.Tensor:
    student_norm = normalize(student_emb)
    teacher_norm = normalize(teacher_emb)
    return (1.0 - (student_norm * teacher_norm).sum(dim=-1)).mean()


def train_model(
    *,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = _SETTINGS.student_max_length,
    grad_accum_steps: int = 1,
    save_dir: str = "checkpoints",
    student_device: str | None = None,
    attention: str | None = None,
) -> None:
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1.")

    resolved_student_device = student_device or _auto_student_device()
    logger.info("Student device: %s", resolved_student_device)

    dataset = load_training_dataset()
    n = len(dataset)
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
                "hub_dataset": _SETTINGS.hub_dataset,
                "grad_accum_steps": grad_accum_steps,
            },
        )

    try:
        logger.info("Loading student model")
        model, processor, projection_head = load_student(
            device=resolved_student_device,
            output_dim=_get_teacher_embedding_dim(dataset),
            attn_implementation=attention,
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
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
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

                batch_rows = [
                    dataset[int(global_index)] for global_index in batch_indices
                ]
                batch_origins, batch_data, teacher_target = _collate_rows(
                    batch_rows,
                    processor=processor,
                    max_length=max_length,
                )
                teacher_target = teacher_target.to(
                    device=resolved_student_device,
                    non_blocking=True,
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
                _log_pre_forward(
                    epoch=epoch,
                    batch_ordinal=batch_ordinal,
                    batches_per_epoch=batches_per_epoch,
                    origins=batch_origins,
                    model_inputs=model_inputs,
                )

                outputs = model(**model_inputs)
                _raise_if_nonfinite(
                    outputs.last_hidden_state,
                    name="student_last_hidden_state",
                    batch_ordinal=batch_ordinal,
                    batches_per_epoch=batches_per_epoch,
                    origins=batch_origins,
                )
                pooled = last_token_pool(
                    outputs.last_hidden_state,
                    model_inputs["attention_mask"],
                )
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

                if (
                    wandb_run is not None
                    and should_step
                    and global_step % _WANDB_LOG_EVERY_N_STEPS == 0
                ):
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


def train_model_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=_SETTINGS.student_max_length)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=_SETTINGS.seed)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--attention",
        choices=("sdpa", "fa"),
        default=None,
        help="Force attention backend for teacher and student.",
    )
    parser.add_argument("--student-device", type=str, default=None)
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        grad_accum_steps=args.grad_accum_steps,
        save_dir=args.save_dir,
        student_device=args.student_device,
        attention=resolve_attention(args.attention),
    )


if __name__ == "__main__":
    train_model_cli()
