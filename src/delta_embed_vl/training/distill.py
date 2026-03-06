import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from torch.optim import AdamW
from transformers import Qwen3VLProcessor, get_cosine_schedule_with_warmup

from delta_embed_vl.artifacts import versioned_name
from delta_embed_vl.data.download import CAULDRON_CONFIGS, load_raw_cauldron
from delta_embed_vl.data.media import coerce_image_to_rgb
from delta_embed_vl.data.preprocess import preprocess_cauldron, preprocess_wikipedia
from delta_embed_vl.model.embedding_inputs import EmbeddingInput, build_student_batch
from delta_embed_vl.model.pooling import last_token_pool, normalize
from delta_embed_vl.model.student import load_student, save_projection_head
from delta_embed_vl.settings import Settings
from delta_embed_vl.training.losses import cosine_distill_loss

logger = logging.getLogger(__name__)

_settings = Settings()
_EMBEDDINGS_DIR = _settings.data_dir / "embeddings"


@dataclass(frozen=True)
class PreparedSource:
    name: str
    embeddings: np.ndarray
    load_example: Callable[[int], EmbeddingInput]
    start: int
    stop: int


def _load_teacher_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher embeddings not found at {path}. Run `prepare-data` first."
        )
    return np.load(str(path), mmap_mode="r")


def _build_sources(
    *, limit: int | None = None
) -> tuple[list[PreparedSource], int, int]:
    """Load processed data and teacher embeddings without concatenating them in RAM."""
    wiki_ds = preprocess_wikipedia(limit=limit)
    wiki_emb = _load_teacher_embeddings(
        _EMBEDDINGS_DIR / f"{versioned_name('wikipedia', limit=limit)}.npy"
    )

    cauldron_datasets = preprocess_cauldron(limit=limit)

    sources: list[PreparedSource] = []
    next_start = 0
    teacher_dim = int(wiki_emb.shape[1])

    def add_source(
        name: str,
        *,
        count: int,
        emb: np.ndarray,
        load_example: Callable[[int], EmbeddingInput],
    ) -> None:
        nonlocal next_start, teacher_dim
        if count == 0 or emb.shape[0] == 0:
            logger.info("Skipping empty %s", name)
            return
        if count != emb.shape[0]:
            raise ValueError(
                f"Dataset/embedding mismatch for {name}: {count} vs {emb.shape[0]}"
            )
        if emb.shape[1] != teacher_dim:
            raise ValueError(
                f"Embedding dimension mismatch for {name}: expected {teacher_dim}, got {emb.shape[1]}"
            )
        sources.append(
            PreparedSource(
                name=name,
                embeddings=emb,
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
        emb=wiki_emb,
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
            text = row["text"] or None
            image_index = int(row["image_index"])
            if image_index < 0:
                return EmbeddingInput(text=text)

            source_index = int(row["source_index"])
            assert raw_dataset is not None
            image = coerce_image_to_rgb(
                raw_dataset[source_index]["images"][image_index]
            )
            if image is None:
                raise ValueError(
                    f"Could not resolve image for cauldron/{config} row {local_index}."
                )
            return EmbeddingInput(text=text, image=image)

        return load_example

    for config, ds in zip(CAULDRON_CONFIGS, cauldron_datasets, strict=False):
        emb_path = (
            _EMBEDDINGS_DIR / "cauldron" / f"{versioned_name(config, limit=limit)}.npy"
        )
        emb = _load_teacher_embeddings(emb_path)
        add_source(
            f"cauldron/{config}",
            count=len(ds),
            emb=emb,
            load_example=make_cauldron_loader(config, ds),
        )

    return sources, next_start, teacher_dim


def _resolve_source(
    sources: list[PreparedSource], global_index: int
) -> tuple[PreparedSource, int]:
    for source in sources:
        if global_index < source.stop:
            return source, global_index - source.start
    raise IndexError(f"Global index {global_index} out of range")


def _collate(
    batch: list[EmbeddingInput],
    teacher_targets: np.ndarray,
    processor: Qwen3VLProcessor,
    max_length: int,
) -> dict[str, torch.Tensor]:
    encoded = build_student_batch(
        processor,
        batch,
        max_length=max_length,
    )
    teacher_target = torch.from_numpy(teacher_targets).float()
    batch_data = {key: value for key, value in encoded.items()}
    batch_data["teacher_target"] = teacher_target
    return batch_data


def train(
    *,
    limit: int | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 512,
    grad_accum_steps: int = 1,
    save_dir: str = "checkpoints",
) -> None:
    """Run cosine distillation training on multimodal data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    logger.info("Loading training data + teacher embeddings")
    sources, n, teacher_dim = _build_sources(limit=limit)
    logger.info("Training on %d samples for %d epochs", n, epochs)

    logger.info("Loading student model")
    model, processor, projection_head = load_student(
        device=device, output_dim=teacher_dim
    )
    model.train()
    projection_head.train()

    indices = np.arange(n, dtype=np.int64)
    total_steps = (n * epochs) // (batch_size * grad_accum_steps)
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(
        list(model.parameters()) + list(projection_head.parameters()),
        lr=lr,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    global_step = 0
    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        num_batches = 0

        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_indices = indices[batch_start:batch_end]
            batch_samples: list[EmbeddingInput] = []
            teacher_targets: list[np.ndarray] = []
            for global_index in batch_indices:
                source, local_index = _resolve_source(sources, int(global_index))
                batch_samples.append(source.load_example(local_index))
                teacher_targets.append(
                    np.asarray(source.embeddings[local_index], dtype=np.float32)
                )

            batch_data = _collate(
                batch_samples,
                np.stack(teacher_targets),
                processor,
                max_length,
            )

            teacher_target = batch_data["teacher_target"].to(device)
            model_inputs = {
                key: value.to(device)
                for key, value in batch_data.items()
                if key != "teacher_target"
            }

            outputs = model(**model_inputs)
            attention_mask = model_inputs["attention_mask"]
            pooled = last_token_pool(outputs.last_hidden_state, attention_mask)
            student_emb = normalize(projection_head(pooled).float())

            loss = cosine_distill_loss(student_emb, teacher_target) / grad_accum_steps
            loss.backward()

            if (num_batches + 1) % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * grad_accum_steps
            num_batches += 1

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
