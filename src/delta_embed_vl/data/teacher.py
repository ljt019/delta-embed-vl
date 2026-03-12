from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLModel, Qwen3VLProcessor
from transformers.utils import is_flash_attn_2_available

from delta_embed_vl import cfg
from delta_embed_vl.model.pooling import last_token_pool, normalize
from delta_embed_vl.model.tokenization import (
    EmbeddingInput,
    get_processor_tokenizer,
)
from delta_embed_vl.model.tokenization import (
    build_teacher_batch as _build_teacher_batch_cpu,
)

logger = logging.getLogger(__name__)


@dataclass
class TeacherEmbedder:
    model: Qwen3VLModel
    processor: Qwen3VLProcessor
    device: torch.device
    output_dim: int

    @torch.inference_mode()
    def embed(self, samples: list[EmbeddingInput]) -> torch.Tensor:
        if not samples:
            return torch.empty((0, self.output_dim), device=self.device)

        if self.device.type == "cuda":
            inputs = _build_teacher_batch_on_device(
                self.processor,
                samples,
                device=self.device,
            )
        else:
            inputs = _build_teacher_batch_cpu(self.processor, samples)
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        pooled = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        return normalize(pooled.float())


def _build_teacher_batch_on_device(
    processor: Qwen3VLProcessor,
    samples: list[EmbeddingInput],
    *,
    device: torch.device,
):
    tokenizer = get_processor_tokenizer(processor)
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": sample.instruction}],
                },
                {
                    "role": "user",
                    "content": (
                        (
                            [{"type": "image", "image": sample.image}]
                            if sample.image is not None
                            else []
                        )
                        + (
                            [{"type": "text", "text": sample.text}]
                            if sample.text
                            else [{"type": "text", "text": ""}]
                        )
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for sample in samples
    ]
    image_inputs = [
        [] if sample.image is None else [sample.image] for sample in samples
    ]
    processor_kwargs: dict[str, object] = {
        "text": prompts,
        "return_tensors": "pt",
        "padding": True,
        "truncation": False,
        "device": device,
    }
    if any(image_inputs):
        processor_kwargs["images"] = image_inputs
    return processor(**processor_kwargs)


def _get_teacher_hidden_size(model: Qwen3VLModel) -> int:
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "hidden_size"):
        return int(text_config.hidden_size)

    raise AttributeError("Could not determine teacher hidden size from model config.")


def load_teacher(
    model_id: str = cfg["data"]["model_id"],
    *,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = None,
) -> TeacherEmbedder:
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    processor = cast(
        Qwen3VLProcessor,
        AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=True,
        ),
    )
    tokenizer = get_processor_tokenizer(processor)
    tokenizer.padding_side = "right"

    if device.startswith("cuda"):
        resolved_attn_implementation = attn_implementation or (
            "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        )
        logger.info("Loading teacher with %s attention", resolved_attn_implementation)
        model = Qwen3VLModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=resolved_attn_implementation,
            local_files_only=True,
        )
    else:
        model = Qwen3VLModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )

    model = nn.Module.to(model, device=torch.device(device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return TeacherEmbedder(
        model=model,
        processor=processor,
        device=torch.device(device),
        output_dim=_get_teacher_hidden_size(model),
    )
