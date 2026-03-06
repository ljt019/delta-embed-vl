from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3_5Model
from transformers.utils import is_flash_attn_2_available

from delta_embed_vl.model.pooling import last_token_pool, normalize

logger = logging.getLogger(__name__)
STUDENT_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
PROJECTION_STATE_FILENAME = "projection_head.pt"
PROJECTION_CONFIG_FILENAME = "projection_config.json"


def get_backbone_hidden_size(model: Qwen3_5Model) -> int:
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "hidden_size"):
        return int(text_config.hidden_size)

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    raise AttributeError("Could not determine student hidden size from model config.")


def get_embedding_dim(model: Qwen3_5Model, projection_head: nn.Module) -> int:
    if isinstance(projection_head, nn.Linear):
        return projection_head.out_features
    return get_backbone_hidden_size(model)


def save_projection_head(projection_head: nn.Module, save_dir: str | Path) -> None:
    save_path = Path(save_dir)
    if not isinstance(projection_head, nn.Linear):
        return

    torch.save(
        projection_head.state_dict(),
        save_path / PROJECTION_STATE_FILENAME,
    )
    (save_path / PROJECTION_CONFIG_FILENAME).write_text(
        json.dumps({"output_dim": projection_head.out_features})
    )


def _load_projection_head(
    model_id: str,
    *,
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
    output_dim: int | None = None,
) -> nn.Module:
    model_path = Path(model_id)
    projection_state_path = model_path / PROJECTION_STATE_FILENAME
    projection_config_path = model_path / PROJECTION_CONFIG_FILENAME

    if projection_state_path.exists():
        projection_config = json.loads(projection_config_path.read_text())
        output_dim = int(projection_config["output_dim"])
        projection_head = nn.Linear(hidden_size, output_dim, bias=False)
        projection_head.load_state_dict(
            torch.load(projection_state_path, map_location=device)
        )
        return projection_head.to(device=device, dtype=dtype)

    if output_dim is None or output_dim == hidden_size:
        return nn.Identity().to(device)

    projection_head = nn.Linear(hidden_size, output_dim, bias=False)
    return projection_head.to(device=device, dtype=dtype)


def load_student(
    model_id: str = STUDENT_MODEL_ID,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    output_dim: int | None = None,
) -> tuple[Qwen3_5Model, PreTrainedTokenizerBase, nn.Module]:
    """Load the student backbone and tokenizer."""
    if device.startswith("cuda"):
        attn_implementation = (
            "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        )
        logger.info("Loading student with %s attention", attn_implementation)
        model = Qwen3_5Model.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
    else:
        model = Qwen3_5Model.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
    model = nn.Module.to(model, device=torch.device(device))
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_id))
    tokenizer.padding_side = "left"
    projection_head = _load_projection_head(
        model_id,
        hidden_size=get_backbone_hidden_size(model),
        device=device,
        dtype=dtype,
        output_dim=output_dim,
    )
    return model, tokenizer, projection_head


@torch.no_grad()
def embed(
    model: Qwen3_5Model,
    projection_head: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    max_length: int = 512,
) -> torch.Tensor:
    """Encode texts into normalized embeddings. For inference / testing."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(model.device)

    outputs = model(**inputs)
    pooled = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return normalize(projection_head(pooled).float())
