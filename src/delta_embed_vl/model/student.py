from __future__ import annotations

import logging

import torch
from transformers import AutoProcessor, Qwen3_5Model
from transformers.utils import is_flash_attn_2_available

from delta_embed_vl.model.pooling import last_token_pool, normalize

logger = logging.getLogger(__name__)
STUDENT_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"


def load_student(
    model_id: str = STUDENT_MODEL_ID,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Qwen3_5Model, AutoProcessor]:
    """Load the student backbone and processor."""
    model_kwargs: dict[str, object] = {"torch_dtype": dtype}
    if device.startswith("cuda"):
        attn_implementation = (
            "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        )
        model_kwargs["attn_implementation"] = attn_implementation
        logger.info("Loading student with %s attention", attn_implementation)

    model = Qwen3_5Model.from_pretrained(
        model_id,
        **model_kwargs,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


@torch.no_grad()
def embed(
    model: Qwen3_5Model,
    processor: AutoProcessor,
    texts: list[str],
    *,
    max_length: int = 512,
) -> torch.Tensor:
    """Encode texts into normalized embeddings. For inference / testing."""
    inputs = processor.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(model.device)

    outputs = model(**inputs)
    pooled = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return normalize(pooled)
