import json
import logging
import shutil
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

from delta_embed_vl import cfg, configure_logging, resolve_attention, set_seed
from delta_embed_vl.data.build import decode_image, load_training_dataset
from delta_embed_vl.model.pooling import last_token_pool, normalize
from delta_embed_vl.model.student import load_student, save_projection_head
from delta_embed_vl.model.tokenization import (
    DEFAULT_EMBED_INSTRUCTION,
    EmbeddingInput,
    build_student_batch,
    is_student_overlength_error,
)

logger = logging.getLogger(__name__)
_PROGRESS_LOG_INTERVAL_S = 30.0
_WANDB_LOG_EVERY_N_STEPS = cfg["wandb"]["interval"]
_PRE_FORWARD_LOG = Path("last_student_forward.jsonl")


def _auto_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
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
            "Prepared dataset contains samples that exceed max_length in config.toml. "
            "Update max_length and rebuild with `uv run prepare`."
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
    step: int,
    max_steps: int,
    origins: list[str],
) -> None:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    if bool(finite_mask.all().item()):
        return

    nonfinite_count = detached.numel() - int(finite_mask.sum().item())
    message = (
        f"{name} became non-finite at step {step}/{max_steps}; "
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
    step: int,
    max_steps: int,
    origins: list[str],
    model_inputs: dict[str, torch.Tensor],
) -> None:
    record = {
        "step": f"{step}/{max_steps}",
        "origins": origins,
        "shapes": {key: list(value.shape) for key, value in model_inputs.items()},
    }
    with _PRE_FORWARD_LOG.open("w") as file:
        json.dump(record, file)
        file.write("\n")
        file.flush()
        sys.stdout.flush()


def _save_checkpoint(
    *,
    model,
    projection_head,
    processor,
    save_dir: Path,
    step: int,
    keep: int,
) -> None:
    ckpt_path = save_dir / f"step-{step}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_path))
    save_projection_head(projection_head, ckpt_path)
    processor.save_pretrained(str(ckpt_path))
    logger.info("Saved checkpoint: %s", ckpt_path)

    if keep > 0:
        existing = sorted(save_dir.glob("step-*"), key=lambda p: p.stat().st_mtime)
        while len(existing) > keep:
            old = existing.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            logger.info("Removed old checkpoint: %s", old.name)


def cosine_distill_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
) -> torch.Tensor:
    student_norm = normalize(student_emb)
    teacher_norm = normalize(teacher_emb)
    return (1.0 - (student_norm * teacher_norm).sum(dim=-1)).mean()


def train_model(*, push_to_hub: bool = False) -> None:
    max_steps = cfg["train"]["max_steps"]
    batch_size = cfg["train"]["batch_size"]
    max_length = cfg["max_length"]
    attention = resolve_attention(cfg["attention"])
    save_dir = "checkpoints"

    resolved_student_device = _auto_device()
    logger.info("Student device: %s", resolved_student_device)

    dataset = load_training_dataset()
    n = len(dataset)
    if n == 0:
        raise ValueError("No training samples found.")
    logger.info("Training on %d samples for %d steps", n, max_steps)

    wandb_run = None
    if cfg["wandb"]["project"] is not None:
        wandb_run = wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"]["name"],
            config=cfg,
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

        lr = cfg["train"]["lr"]
        warmup_ratio = cfg["train"]["warmup_ratio"]
        grad_accum_steps = cfg["train"]["grad_accum_steps"]
        warmup_steps = int(max_steps * warmup_ratio)
        optimizer = AdamW(
            list(model.parameters()) + list(projection_head.parameters()),
            lr=lr,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )

        indices = np.arange(n, dtype=np.int64)
        cursor = n  # force shuffle on first iteration
        global_step = 0
        micro_step = 0
        running_loss = 0.0
        train_start_s = time.monotonic()
        last_progress_log_s = train_start_s
        optimizer.zero_grad(set_to_none=True)

        while global_step < max_steps:
            if cursor >= n:
                np.random.shuffle(indices)
                cursor = 0

            batch_start_s = time.monotonic()
            batch_end = min(cursor + batch_size, n)
            batch_indices = indices[cursor:batch_end]
            cursor = batch_end

            batch_rows = [dataset[int(idx)] for idx in batch_indices]
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
                step=global_step,
                max_steps=max_steps,
                origins=batch_origins,
            )

            model_inputs = {
                key: value.to(resolved_student_device)
                for key, value in batch_data.items()
            }
            _log_pre_forward(
                step=global_step,
                max_steps=max_steps,
                origins=batch_origins,
                model_inputs=model_inputs,
            )

            outputs = model(**model_inputs)
            _raise_if_nonfinite(
                outputs.last_hidden_state,
                name="student_last_hidden_state",
                step=global_step,
                max_steps=max_steps,
                origins=batch_origins,
            )
            pooled = last_token_pool(
                outputs.last_hidden_state,
                model_inputs["attention_mask"],
            )
            _raise_if_nonfinite(
                pooled,
                name="student_pooled",
                step=global_step,
                max_steps=max_steps,
                origins=batch_origins,
            )
            projected = projection_head(pooled).float()
            _raise_if_nonfinite(
                projected,
                name="student_projected",
                step=global_step,
                max_steps=max_steps,
                origins=batch_origins,
            )
            student_emb = normalize(projected)
            _raise_if_nonfinite(
                student_emb,
                name="student_embedding",
                step=global_step,
                max_steps=max_steps,
                origins=batch_origins,
            )

            loss = cosine_distill_loss(student_emb, teacher_target) / grad_accum_steps
            _raise_if_nonfinite(
                loss.reshape(1),
                name="distill_loss",
                step=global_step,
                max_steps=max_steps,
                origins=batch_origins,
            )
            loss.backward()

            micro_step += 1
            raw_loss = loss.item() * grad_accum_steps
            running_loss += raw_loss

            should_step = micro_step % grad_accum_steps == 0
            if should_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                now_s = time.monotonic()
                batch_elapsed_s = now_s - batch_start_s

                if (
                    wandb_run is not None
                    and global_step % _WANDB_LOG_EVERY_N_STEPS == 0
                ):
                    wandb.log(
                        {
                            "train/loss": raw_loss,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/batch_time_s": batch_elapsed_s,
                        },
                        step=global_step,
                    )

                ckpt_interval = cfg["ckpt"]["interval"]
                if ckpt_interval > 0 and global_step % ckpt_interval == 0:
                    _save_checkpoint(
                        model=model,
                        projection_head=projection_head,
                        processor=processor,
                        save_dir=Path(save_dir),
                        step=global_step,
                        keep=cfg["ckpt"]["keep"],
                    )

                if (
                    global_step == 1
                    or (now_s - last_progress_log_s) >= _PROGRESS_LOG_INTERVAL_S
                    or global_step == max_steps
                ):
                    elapsed_s = now_s - train_start_s
                    avg_step_s = elapsed_s / global_step
                    eta_s = avg_step_s * (max_steps - global_step)
                    avg_loss = running_loss / global_step
                    logger.info(
                        "Step %d/%d (%.1f%%)  loss=%.6f avg=%.6f  step=%.2fs  elapsed=%s  eta=%s",
                        global_step,
                        max_steps,
                        (100.0 * global_step) / max_steps,
                        raw_loss,
                        avg_loss,
                        batch_elapsed_s,
                        _format_duration(elapsed_s),
                        _format_duration(eta_s),
                    )
                    last_progress_log_s = now_s

        final_path = Path(save_dir) / cfg["name"]
        final_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_path))
        save_projection_head(projection_head, final_path)
        processor.save_pretrained(str(final_path))
        logger.info(
            "Training done: %d steps, avg_loss=%.6f",
            global_step,
            running_loss / max(global_step, 1),
        )
        logger.info("Saved final model to %s", final_path)

        if push_to_hub:
            hub_id = cfg["id"]
            logger.info("Pushing model to Hub: %s", hub_id)
            model.push_to_hub(hub_id)
            projection_head_path = final_path / "projection_head.pt"
            if projection_head_path.exists():
                from huggingface_hub import upload_file

                upload_file(
                    path_or_fileobj=str(projection_head_path),
                    path_in_repo="projection_head.pt",
                    repo_id=hub_id,
                )
            processor.push_to_hub(hub_id)
            logger.info("Pushed model to Hub: %s", hub_id)
    finally:
        if wandb_run is not None:
            wandb.finish()


def train_model_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help=f"Push trained model to Hub as {cfg['id']}",
    )
    args = parser.parse_args()

    configure_logging()
    set_seed(cfg["train"]["seed"])
    train_model(push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    train_model_cli()
