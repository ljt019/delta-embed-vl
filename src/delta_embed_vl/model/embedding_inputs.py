from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen3VLProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import BatchEncoding

from delta_embed_vl.data.media import ImageLike, coerce_image_to_rgb
from delta_embed_vl.settings import Settings

_settings = Settings()

DEFAULT_EMBED_INSTRUCTION = "Represent the user's input."


@dataclass(frozen=True)
class EmbeddingInput:
    text: str | None = None
    image: ImageLike = None
    instruction: str = DEFAULT_EMBED_INSTRUCTION


@lru_cache
def get_teacher_processor() -> Qwen3VLProcessor:
    return cast(
        Qwen3VLProcessor,
        AutoProcessor.from_pretrained(
            _settings.teacher_model,
            trust_remote_code=True,
        ),
    )


def get_processor_tokenizer(processor: object) -> PreTrainedTokenizerBase:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError("Processor does not expose a tokenizer.")
    return cast(PreTrainedTokenizerBase, tokenizer)


@lru_cache
def get_teacher_tokenizer() -> PreTrainedTokenizerBase:
    processor = get_teacher_processor()
    return get_processor_tokenizer(processor)


def _resolve_input(
    sample: EmbeddingInput,
) -> tuple[str | None, Image.Image | None, str]:
    resolved_image = coerce_image_to_rgb(sample.image)
    if sample.image is not None and resolved_image is None:
        raise ValueError("Could not resolve image for embedding input.")
    return sample.text, resolved_image, sample.instruction


def _build_conversation(
    *, text: str | None, image: Image.Image | None, instruction: str
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    if text:
        user_content.append({"type": "text", "text": text})
    if not user_content:
        user_content.append({"type": "text", "text": ""})

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def build_embedding_conversation(sample: EmbeddingInput) -> list[dict[str, Any]]:
    text, image, instruction = _resolve_input(sample)
    return _build_conversation(text=text, image=image, instruction=instruction)


def _render_prompt(
    tokenizer: PreTrainedTokenizerBase,
    *,
    text: str | None,
    image: Image.Image | None,
    instruction: str,
    add_generation_prompt: bool,
) -> str:
    conversation = _build_conversation(
        text=text,
        image=image,
        instruction=instruction,
    )
    rendered = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return cast(str, rendered)


def render_teacher_prompt(
    sample: EmbeddingInput, *, add_generation_prompt: bool = True
) -> str:
    tokenizer = get_teacher_tokenizer()
    text, image, instruction = _resolve_input(sample)
    return _render_prompt(
        tokenizer,
        text=text,
        image=image,
        instruction=instruction,
        add_generation_prompt=add_generation_prompt,
    )


def count_teacher_prompt_tokens(
    sample: EmbeddingInput, *, add_generation_prompt: bool = True
) -> int:
    tokenizer = get_teacher_tokenizer()
    tokenized = cast(
        BatchEncoding,
        tokenizer.apply_chat_template(
            build_embedding_conversation(sample),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        ),
    )
    input_ids = cast(list[int], tokenized["input_ids"])
    return len(input_ids)


@lru_cache
def teacher_prompt_token_limit() -> int:
    return _settings.teacher_max_model_len - _settings.teacher_prompt_margin_tokens


@lru_cache
def teacher_text_token_budget() -> int:
    empty_prompt_tokens = count_teacher_prompt_tokens(EmbeddingInput(text=""))
    budget = teacher_prompt_token_limit() - empty_prompt_tokens
    if budget <= 0:
        raise ValueError("Teacher text token budget must be positive.")
    return budget


def _build_processor_batch(
    processor: Qwen3VLProcessor,
    samples: list[EmbeddingInput],
    *,
    padding: bool,
    truncation: bool,
    max_length: int | None = None,
) -> BatchFeature:
    tokenizer = get_teacher_tokenizer()
    resolved_inputs = [_resolve_input(sample) for sample in samples]
    prompts = [
        _render_prompt(
            tokenizer,
            text=text,
            image=image,
            instruction=instruction,
            add_generation_prompt=True,
        )
        for text, image, instruction in resolved_inputs
    ]

    image_inputs = [[] if image is None else [image] for _, image, _ in resolved_inputs]
    processor_kwargs: dict[str, Any] = {
        "text": prompts,
        "return_tensors": "pt",
        "padding": padding,
        "truncation": truncation,
    }
    if max_length is not None:
        processor_kwargs["max_length"] = max_length
    if any(image_inputs):
        processor_kwargs["images"] = image_inputs

    return processor(**processor_kwargs)


def teacher_processed_token_lengths(samples: list[EmbeddingInput]) -> list[int]:
    if not samples:
        return []

    features = _build_processor_batch(
        get_teacher_processor(),
        samples,
        padding=True,
        truncation=False,
    )
    attention_mask = features.get("attention_mask")
    if attention_mask is not None:
        rows = cast(list[list[int]], attention_mask.tolist())
        return [sum(row) for row in rows]

    input_ids = features["input_ids"]
    rows = cast(list[list[int]], input_ids.tolist())
    pad_token_id = get_teacher_tokenizer().pad_token_id
    if pad_token_id is None:
        return [len(row) for row in rows]
    return [sum(1 for token in row if token != pad_token_id) for row in rows]


def count_teacher_processed_tokens(sample: EmbeddingInput) -> int:
    lengths = teacher_processed_token_lengths([sample])
    if not lengths:
        raise ValueError("Expected at least one processed token length.")
    return lengths[0]


def build_student_batch(
    processor: Qwen3VLProcessor,
    samples: list[EmbeddingInput],
    *,
    max_length: int,
) -> BatchFeature:
    return _build_processor_batch(
        processor,
        samples,
        padding=True,
        truncation=True,
        max_length=max_length,
    )
