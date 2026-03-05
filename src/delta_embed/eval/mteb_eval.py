from __future__ import annotations

import logging
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

from delta_embed.model.pooling import last_token_pool, normalize
from delta_embed.model.student import STUDENT_MODEL_ID, load_student

logger = logging.getLogger(__name__)


class DeltaEmbedEncoder:
    """MTEB-compatible encoder wrapping our student model."""

    def __init__(
        self,
        model_name: str = STUDENT_MODEL_ID,
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = load_student(
            model_id=model_name, device=self._device
        )
        self.model.eval()

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
        batch_size = kwargs.get("batch_size", 32)
        all_embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i : i + batch_size]
                encoded = self.processor.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self._device)

                outputs = self.model(**encoded)
                pooled = last_token_pool(
                    outputs.last_hidden_state, encoded["attention_mask"]
                )
                emb = normalize(pooled)
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
            name="delta-embed",
            revision=self.revision,
            release_date=None,
            languages=["eng-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=512,
            embed_dim=1024,
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
) -> ModelResult:
    """Run MTEB evaluation on the given tasks."""
    if tasks is None:
        tasks = [
            "STS22.v2",
            "STSBenchmark",
            "NanoArguAnaRetrieval",
            "NanoNFCorpusRetrieval",
        ]

    encoder = DeltaEmbedEncoder(model_name=model_path)
    mteb_tasks = mteb.get_tasks(tasks=tasks, languages=["eng"])
    result = mteb.evaluate(encoder, mteb_tasks)

    for task_result in result.task_results:
        logger.info("%s: %s", task_result.task_name, task_result.get_score())

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_eval()
