import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from torch.optim import AdamW
from transformers import Qwen3VLProcessor, get_cosine_schedule_with_warmup

from delta_embed_vl.data.download import CAULDRON_CONFIGS, load_raw_cauldron
from delta_embed_vl.data.preprocess import (
    load_cauldron_embedding_input,
    preprocess_cauldron,
    preprocess_wikipedia,
)
from delta_embed_vl.model.embedding_inputs import EmbeddingInput, build_student_batch
from delta_embed_vl.model.pooling import last_token_pool, normalize
from delta_embed_vl.model.student import load_student, save_projection_head
from delta_embed_vl.model.teacher import TeacherEmbedder, load_teacher
from delta_embed_vl.training.losses import cosine_distill_loss

logger = logging.getLogger(__name__)


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
    *, limit: int | None = None
) -> tuple[list[PreparedSource], int]:
    """Load processed data and resolve each global index back to an example."""
    wiki_ds = preprocess_wikipedia(limit=limit)
    cauldron_datasets = preprocess_cauldron(limit=limit)

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
            return load_cauldron_embedding_input(
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


def train(
    *,
    limit: int | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 8192,
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
    sources, n = _build_sources(limit=limit)
    if n == 0:
        raise ValueError("No training samples found.")
    logger.info("Training on %d samples for %d epochs", n, epochs)

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
    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        num_batches = 0

        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_indices = indices[batch_start:batch_end]
            batch_samples: list[EmbeddingInput] = []
            for global_index in batch_indices:
                source, local_index = _resolve_source(sources, int(global_index))
                batch_samples.append(source.load_example(local_index))

            batch_data = _collate(
                batch_samples,
                processor,
                max_length,
            )
            teacher_target = _embed_teacher_targets(
                teacher,
                batch_samples,
                teacher_batch_size=resolved_teacher_batch_size,
                target_device=resolved_student_device,
            )

            model_inputs = {
                key: value.to(resolved_student_device)
                for key, value in batch_data.items()
            }

            outputs = model(**model_inputs)
            attention_mask = model_inputs["attention_mask"]
            pooled = last_token_pool(outputs.last_hidden_state, attention_mask)
            student_emb = normalize(projection_head(pooled).float())

            loss = cosine_distill_loss(student_emb, teacher_target) / grad_accum_steps
            loss.backward()

            num_batches += 1
            should_step = (num_batches % grad_accum_steps == 0) or (batch_end == n)
            if should_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_loss += loss.item() * grad_accum_steps

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(
            "Epoch %d/%d  loss=%.6f  lr=%.2e  steps=%d",
            epoch + 1,
            epochs,
            avg_loss,
            scheduler.get_last_lr()[0],
            global_step,
        )

    out_path = Path(save_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    save_projection_head(projection_head, out_path)
    processor.save_pretrained(str(out_path))
    logger.info("Saved checkpoint to %s", out_path)
