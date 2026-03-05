from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from torch.optim import AdamW
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from delta_embed.data.preprocess import preprocess_cauldron, preprocess_wikipedia
from delta_embed.model.pooling import last_token_pool, normalize
from delta_embed.model.student import load_student
from delta_embed.settings import Settings
from delta_embed.training.losses import cosine_distill_loss

logger = logging.getLogger(__name__)

_settings = Settings()
_EMBEDDINGS_DIR = _settings.data_dir / "embeddings"


def _load_teacher_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher embeddings not found at {path}. Run `prepare-data` first."
        )
    return np.load(str(path), mmap_mode="r")


def _build_dataset(*, limit: int | None = None) -> tuple[Dataset, np.ndarray]:
    """Load preprocessed data + teacher embeddings, return aligned pair."""
    suffix = f"_test{limit}" if limit else ""

    wiki_ds = preprocess_wikipedia(limit=limit)
    wiki_emb = _load_teacher_embeddings(_EMBEDDINGS_DIR / f"wikipedia{suffix}.npy")

    cauldron_datasets = preprocess_cauldron(limit=limit)
    cauldron_embs = []
    from delta_embed.data.download import CAULDRON_CONFIGS

    for config, ds in zip(CAULDRON_CONFIGS, cauldron_datasets):
        emb_path = _EMBEDDINGS_DIR / "cauldron" / f"{config}{suffix}.npy"
        cauldron_embs.append(_load_teacher_embeddings(emb_path))

    all_datasets = [wiki_ds] + cauldron_datasets
    all_embeddings = np.concatenate([wiki_emb] + cauldron_embs, axis=0)

    combined = concatenate_datasets(all_datasets)
    combined.set_format("python")

    assert len(combined) == len(all_embeddings), (
        f"Dataset/embedding mismatch: {len(combined)} vs {len(all_embeddings)}"
    )
    return combined, all_embeddings


def _collate(
    batch: list[dict],
    teacher_embeddings: np.ndarray,
    indices: list[int],
    processor: AutoProcessor,
    max_length: int,
) -> dict[str, torch.Tensor]:
    texts = [sample["text"] or "" for sample in batch]

    encoded = processor.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    teacher_target = torch.from_numpy(
        np.stack([teacher_embeddings[i] for i in indices])
    ).float()

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "teacher_target": teacher_target,
    }


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
    """Run cosine distillation training on text data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    logger.info("Loading student model")
    model, processor = load_student(device=device)
    model.train()

    logger.info("Loading training data + teacher embeddings")
    dataset, teacher_embeddings = _build_dataset(limit=limit)
    n = len(dataset)
    logger.info("Training on %d samples for %d epochs", n, epochs)

    indices = list(range(n))
    total_steps = (n * epochs) // (batch_size * grad_accum_steps)
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=lr)
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
            batch_samples = [dataset[int(i)] for i in batch_indices]

            batch_data = _collate(
                batch_samples, teacher_embeddings, batch_indices, processor, max_length
            )

            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            teacher_target = batch_data["teacher_target"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            student_emb = normalize(
                last_token_pool(outputs.last_hidden_state, attention_mask)
            )

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
    processor.save_pretrained(str(out_path))
    logger.info("Saved checkpoint to %s", out_path)
