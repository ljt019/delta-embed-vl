from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mteb
import torch
from mteb import TaskMetadata
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.results import ModelResult
from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType
from typing_extensions import Unpack

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from delta_embed_vl import cfg
from delta_embed_vl.model.pooling import last_token_pool, normalize
from delta_embed_vl.model.student import (
    STUDENT_MODEL_ID,
    get_embedding_dim,
    load_student,
)
from delta_embed_vl.model.tokenization import EmbeddingInput, build_student_batch

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
    batch_texts: list[str],
) -> None:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    if bool(finite_mask.all().item()):
        return

    nonfinite_count = detached.numel() - int(finite_mask.sum().item())
    text_lengths = [len(text) for text in batch_texts]
    message = (
        f"{name} became non-finite during eval; task={task_name} subset={hf_subset} "
        f"batch={batch_start}:{batch_stop} max_length={max_length} "
        f"shape={tuple(detached.shape)} dtype={detached.dtype} device={detached.device} "
        f"nonfinite={nonfinite_count}/{detached.numel()}"
    )
    if text_lengths:
        message += f" text_len_min={min(text_lengths)} text_len_max={max(text_lengths)}"
    if bool(finite_mask.any().item()):
        finite_values = detached[finite_mask]
        message += (
            f" finite_min={float(finite_values.min().item()):.6g}"
            f" finite_max={float(finite_values.max().item()):.6g}"
        )
    raise FloatingPointError(message)


class DeltaEmbedEncoder:
    """MTEB-compatible encoder wrapping our student model."""

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
        all_texts: list[str] = [text for batch in inputs for text in batch["text"]]
        batch_size = kwargs.get("batch_size", cfg["train"]["batch_size"])
        all_embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i : i + batch_size]
                batch_stop = i + len(batch_texts)
                encoded = build_student_batch(
                    self.processor,
                    [EmbeddingInput(text=text) for text in batch_texts],
                    max_length=self.max_length,
                ).to(self._device)

                outputs = self.model(**encoded)
                _raise_if_nonfinite(
                    outputs.last_hidden_state,
                    name="eval_last_hidden_state",
                    task_name=task_metadata.name,
                    hf_subset=hf_subset,
                    batch_start=i,
                    batch_stop=batch_stop,
                    max_length=self.max_length,
                    batch_texts=batch_texts,
                )
                pooled = last_token_pool(
                    outputs.last_hidden_state, encoded["attention_mask"]
                )
                _raise_if_nonfinite(
                    pooled,
                    name="eval_pooled",
                    task_name=task_metadata.name,
                    hf_subset=hf_subset,
                    batch_start=i,
                    batch_stop=batch_stop,
                    max_length=self.max_length,
                    batch_texts=batch_texts,
                )
                projected = self.projection_head(pooled).float()
                _raise_if_nonfinite(
                    projected,
                    name="eval_projected",
                    task_name=task_metadata.name,
                    hf_subset=hf_subset,
                    batch_start=i,
                    batch_stop=batch_stop,
                    max_length=self.max_length,
                    batch_texts=batch_texts,
                )
                emb = normalize(projected)
                _raise_if_nonfinite(
                    emb,
                    name="eval_embedding",
                    task_name=task_metadata.name,
                    hf_subset=hf_subset,
                    batch_start=i,
                    batch_stop=batch_stop,
                    max_length=self.max_length,
                    batch_texts=batch_texts,
                )
                all_embeddings.append(emb.cpu().float())

        return torch.cat(all_embeddings, dim=0)

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
        )


def run_eval(
    model_path: str = STUDENT_MODEL_ID,
    tasks: list[str] | None = None,
    eval_batch_size: int = cfg["train"]["batch_size"],
    max_length: int = cfg["max_length"],
    device: str | None = None,
    attention: str | None = None,
) -> ModelResult:
    """Run MTEB evaluation on the given tasks."""
    if tasks is None:
        tasks = [
            "STS22.v2",
            "STSBenchmark",
            "NanoArguAnaRetrieval",
            "NanoNFCorpusRetrieval",
        ]

    encoder = DeltaEmbedEncoder(
        model_name=model_path,
        device=device,
        max_length=max_length,
        attention=attention,
    )
    mteb_tasks = mteb.get_tasks(tasks=tasks, languages=["eng"])
    result = mteb.evaluate(
        encoder,
        mteb_tasks,
        encode_kwargs={"batch_size": eval_batch_size},
    )

    for task_result in result.task_results:
        logger.info("%s: %s", task_result.task_name, task_result.get_score())

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_eval()
