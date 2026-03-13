from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from mteb import TaskMetadata
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType
from PIL import Image
from typing_extensions import Unpack

from delta_embed_vl import cfg
from delta_embed_vl.model.pooling import last_token_pool, normalize
from delta_embed_vl.model.student import (
    STUDENT_MODEL_ID,
    get_embedding_dim,
    load_student,
)
from delta_embed_vl.model.tokenization import (
    DEFAULT_EMBED_INSTRUCTION,
    EmbeddingInput,
    VideoInput,
    build_student_batch,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _as_mteb_model_name(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.exists() or "\\" in model_name or model_name.startswith("."):
        return f"local/{model_path.name}"
    if model_name.count("/") == 1 and not model_name.startswith("/"):
        return model_name
    return f"local/{model_path.name}"


def _raise_if_nonfinite(
    tensor: torch.Tensor,
    *,
    name: str,
    task_name: str,
    hf_subset: str,
    batch_start: int,
    batch_stop: int,
    max_length: int,
    batch_inputs: list[str],
) -> None:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    if bool(finite_mask.all().item()):
        return

    nonfinite_count = detached.numel() - int(finite_mask.sum().item())
    message = (
        f"{name} became non-finite during eval; task={task_name} subset={hf_subset} "
        f"batch={batch_start}:{batch_stop} max_length={max_length} "
        f"shape={tuple(detached.shape)} dtype={detached.dtype} device={detached.device} "
        f"nonfinite={nonfinite_count}/{detached.numel()}"
    )
    if batch_inputs:
        message += f" inputs={batch_inputs[:3]}"
    if bool(finite_mask.any().item()):
        finite_values = detached[finite_mask]
        message += (
            f" finite_min={float(finite_values.min().item()):.6g}"
            f" finite_max={float(finite_values.max().item()):.6g}"
        )
    raise FloatingPointError(message)


class DeltaEmbedEncoder:
    """Shared encoder for MTEB and our custom multimodal evals."""

    def __init__(
        self,
        model_name: str = STUDENT_MODEL_ID,
        revision: str | None = None,
        device: str | None = None,
        max_length: int = cfg["max_length"],
        attention: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.mteb_name = _as_mteb_model_name(model_name)
        self.revision = revision
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model, self.processor, self.projection_head = load_student(
            model_id=model_name,
            device=self._device,
            attn_implementation=attention,
        )
        self.model.eval()
        self.projection_head.eval()
        self.embed_dim = get_embedding_dim(self.model, self.projection_head)

    def encode_texts(
        self,
        texts: list[str],
        *,
        batch_size: int,
        instruction: str = DEFAULT_EMBED_INSTRUCTION,
    ) -> torch.Tensor:
        return self.encode_samples(
            [EmbeddingInput(text=text, instruction=instruction) for text in texts],
            batch_size=batch_size,
        )

    def encode_images(
        self,
        images: list[Image.Image],
        *,
        batch_size: int,
        instruction: str = DEFAULT_EMBED_INSTRUCTION,
    ) -> torch.Tensor:
        return self.encode_samples(
            [EmbeddingInput(image=image, instruction=instruction) for image in images],
            batch_size=batch_size,
        )

    def encode_videos(
        self,
        videos: list[VideoInput],
        *,
        batch_size: int,
        instruction: str = DEFAULT_EMBED_INSTRUCTION,
    ) -> torch.Tensor:
        return self.encode_samples(
            [EmbeddingInput(video=video, instruction=instruction) for video in videos],
            batch_size=batch_size,
        )

    def encode_samples(
        self,
        samples: list[EmbeddingInput],
        *,
        batch_size: int,
        task_name: str = "custom_eval",
        hf_subset: str = "custom",
    ) -> torch.Tensor:
        if not samples:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)

        all_embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                batch_samples = samples[start : start + batch_size]
                batch_stop = start + len(batch_samples)
                encoded = build_student_batch(
                    self.processor,
                    batch_samples,
                    max_length=self.max_length,
                ).to(self._device)
                outputs = self.model(**encoded)
                batch_inputs = [
                    self._describe_sample(sample) for sample in batch_samples
                ]
                _raise_if_nonfinite(
                    outputs.last_hidden_state,
                    name="eval_last_hidden_state",
                    task_name=task_name,
                    hf_subset=hf_subset,
                    batch_start=start,
                    batch_stop=batch_stop,
                    max_length=self.max_length,
                    batch_inputs=batch_inputs,
                )
                pooled = last_token_pool(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                )
                _raise_if_nonfinite(
                    pooled,
                    name="eval_pooled",
                    task_name=task_name,
                    hf_subset=hf_subset,
                    batch_start=start,
                    batch_stop=batch_stop,
                    max_length=self.max_length,
                    batch_inputs=batch_inputs,
                )
                projected = self.projection_head(pooled).float()
                _raise_if_nonfinite(
                    projected,
                    name="eval_projected",
                    task_name=task_name,
                    hf_subset=hf_subset,
                    batch_start=start,
                    batch_stop=batch_stop,
                    max_length=self.max_length,
                    batch_inputs=batch_inputs,
                )
                embeddings = normalize(projected)
                _raise_if_nonfinite(
                    embeddings,
                    name="eval_embedding",
                    task_name=task_name,
                    hf_subset=hf_subset,
                    batch_start=start,
                    batch_stop=batch_stop,
                    max_length=self.max_length,
                    batch_inputs=batch_inputs,
                )
                all_embeddings.append(embeddings.cpu().float())
        return torch.cat(all_embeddings, dim=0)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        batch_size = kwargs.get(
            "batch_size",
            cfg.get("eval", {}).get("batch_size", cfg["train"]["batch_size"]),
        )
        all_samples: list[EmbeddingInput] = []
        for batch in inputs:
            all_samples.extend(
                self._samples_from_batch(
                    batch,
                    task_metadata=task_metadata,
                    prompt_type=prompt_type,
                )
            )
        return self.encode_samples(
            all_samples,
            batch_size=batch_size,
            task_name=task_metadata.name,
            hf_subset=hf_subset,
        )

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        if not isinstance(embeddings1, torch.Tensor):
            embeddings1 = torch.tensor(embeddings1)
        if not isinstance(embeddings2, torch.Tensor):
            embeddings2 = torch.tensor(embeddings2)
        return embeddings1 @ embeddings2.T

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        if not isinstance(embeddings1, torch.Tensor):
            embeddings1 = torch.tensor(embeddings1)
        if not isinstance(embeddings2, torch.Tensor):
            embeddings2 = torch.tensor(embeddings2)
        return (embeddings1 * embeddings2).sum(dim=-1)

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return ModelMeta(
            loader=None,
            name=self.mteb_name,
            revision=self.revision,
            release_date=None,
            languages=["eng-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=self.max_length,
            embed_dim=self.embed_dim,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=False,
            training_datasets=None,
            modalities=["text", "image"],
        )

    def _samples_from_batch(
        self,
        batch: BatchedInput,
        *,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> list[EmbeddingInput]:
        batch_size = self._batch_size(batch)
        prompt_key = getattr(prompt_type, "value", prompt_type)
        task_prompt = None
        if prompt_key is not None and task_metadata.prompt is not None:
            task_prompt = task_metadata.prompt.get(prompt_key)

        samples: list[EmbeddingInput] = []
        for index in range(batch_size):
            samples.append(
                EmbeddingInput(
                    text=self._text_at(batch, index),
                    image=self._image_at(batch, index),
                    instruction=self._instruction_at(batch, index, task_prompt),
                )
            )
        return samples

    def _batch_size(self, batch: BatchedInput) -> int:
        for key in ("query", "text", "body", "title", "image"):
            values = batch.get(key)
            if values is not None:
                return len(values)
        raise ValueError(f"Unsupported MTEB batch keys: {list(batch.keys())}")

    def _instruction_at(
        self,
        batch: BatchedInput,
        index: int,
        task_prompt: str | None,
    ) -> str:
        instructions = batch.get("instruction")
        if instructions is not None and instructions[index]:
            return instructions[index]
        if task_prompt:
            return task_prompt
        return DEFAULT_EMBED_INSTRUCTION

    def _text_at(self, batch: BatchedInput, index: int) -> str | None:
        queries = batch.get("query")
        if queries is not None:
            return queries[index]
        texts = batch.get("text")
        if texts is not None:
            return texts[index]
        titles = batch.get("title")
        bodies = batch.get("body")
        if titles is None and bodies is None:
            return None

        title = titles[index] if titles is not None else ""
        body = bodies[index] if bodies is not None else ""
        combined = "\n".join(part for part in (title, body) if part)
        return combined or None

    def _image_at(self, batch: BatchedInput, index: int) -> Image.Image | None:
        images = batch.get("image")
        if images is None:
            return None

        image = images[index]
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, list):
            if not image:
                return None
            first_image = image[0]
            if isinstance(first_image, Image.Image):
                return first_image
        raise TypeError(f"Unsupported image payload type: {type(image)!r}")

    def _describe_sample(self, sample: EmbeddingInput) -> str:
        if sample.text:
            return sample.text[:80]
        if sample.image is not None:
            return "<image>"
        if sample.video is not None:
            return "<video>"
        return "<empty>"
