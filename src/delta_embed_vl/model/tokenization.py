from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen3VLProcessor
from transformers.feature_extraction_utils import BatchFeature

from delta_embed_vl import cfg

_STUDENT_MULTIMODAL_MISMATCH = "Mismatch in `image` token count"
_STUDENT_VIDEO_MULTIMODAL_MISMATCH = "Mismatch in `video` token count"
_STUDENT_OVERLENGTH_MESSAGE = (
    "Student batch exceeds the configured max_length after multimodal token expansion."
)
DEFAULT_EMBED_INSTRUCTION = "Represent the user's input."
VideoInput = str | Path | list[Image.Image]


@dataclass(frozen=True)
class EmbeddingInput:
    text: str | None = None
    image: Image.Image | None = None
    video: VideoInput | None = None
    instruction: str = DEFAULT_EMBED_INSTRUCTION


@lru_cache
def get_student_processor() -> Qwen3VLProcessor:
    processor = cast(
        Qwen3VLProcessor,
        AutoProcessor.from_pretrained(
            cfg["student_id"],
            trust_remote_code=True,
        ),
    )
    tokenizer = get_processor_tokenizer(processor)
    tokenizer.padding_side = "left"
    return processor


def get_processor_tokenizer(processor: object) -> PreTrainedTokenizerBase:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError("Processor does not expose a tokenizer.")
    return cast(PreTrainedTokenizerBase, tokenizer)


def _resolve_input(
    sample: EmbeddingInput,
) -> tuple[str | None, Image.Image | None, VideoInput | None, str]:
    return sample.text, sample.image, sample.video, sample.instruction


def _build_conversation(
    *,
    text: str | None,
    image: Image.Image | None,
    video: VideoInput | None,
    instruction: str,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    if video is not None:
        user_content.append({"type": "video", "video": video})
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


def _render_prompt(
    tokenizer: PreTrainedTokenizerBase,
    *,
    text: str | None,
    image: Image.Image | None,
    video: VideoInput | None,
    instruction: str,
    add_generation_prompt: bool,
) -> str:
    rendered = tokenizer.apply_chat_template(
        _build_conversation(
            text=text,
            image=image,
            video=video,
            instruction=instruction,
        ),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return cast(str, rendered)


def _build_processor_batch(
    processor: Qwen3VLProcessor,
    samples: list[EmbeddingInput],
    *,
    padding: bool,
    truncation: bool,
    max_length: int | None = None,
    device: object | None = None,
) -> BatchFeature:
    tokenizer = get_processor_tokenizer(processor)
    resolved_inputs = [_resolve_input(sample) for sample in samples]
    conversations = [
        _build_conversation(
            text=text,
            image=image,
            video=video,
            instruction=instruction,
        )
        for text, image, video, instruction in resolved_inputs
    ]
    prompts = [
        _render_prompt(
            tokenizer,
            text=text,
            image=image,
            video=video,
            instruction=instruction,
            add_generation_prompt=True,
        )
        for text, image, video, instruction in resolved_inputs
    ]
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        conversations,
        return_video_kwargs=True,
    )
    processor_kwargs: dict[str, Any] = {
        "text": prompts,
        "return_tensors": "pt",
        "padding": padding,
        "truncation": truncation,
    }
    if max_length is not None:
        processor_kwargs["max_length"] = max_length
    if device is not None:
        processor_kwargs["device"] = device
    if image_inputs is not None:
        processor_kwargs["images"] = image_inputs
    if video_inputs is not None:
        processor_kwargs["videos"] = video_inputs
    if video_kwargs is not None:
        processor_kwargs.update(video_kwargs)
    return processor(**processor_kwargs)


def is_student_overlength_error(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) and (
        _STUDENT_MULTIMODAL_MISMATCH in str(exc)
        or _STUDENT_VIDEO_MULTIMODAL_MISMATCH in str(exc)
        or _STUDENT_OVERLENGTH_MESSAGE in str(exc)
    )


def _student_batch_fit_flags(
    processor: Qwen3VLProcessor,
    samples: list[EmbeddingInput],
    *,
    max_length: int,
) -> list[bool]:
    if not samples:
        return []
    try:
        build_student_batch(processor, samples, max_length=max_length)
    except ValueError as exc:
        if not is_student_overlength_error(exc):
            raise
        if len(samples) == 1:
            return [False]
        midpoint = len(samples) // 2
        return _student_batch_fit_flags(
            processor,
            samples[:midpoint],
            max_length=max_length,
        ) + _student_batch_fit_flags(
            processor,
            samples[midpoint:],
            max_length=max_length,
        )
    return [True] * len(samples)


def student_batch_fit_flags(
    processor: Qwen3VLProcessor,
    samples: list[EmbeddingInput],
    *,
    max_length: int,
    batch_size: int = 8,
) -> list[bool]:
    fits: list[bool] = []
    for start in range(0, len(samples), batch_size):
        fits.extend(
            _student_batch_fit_flags(
                processor,
                samples[start : start + batch_size],
                max_length=max_length,
            )
        )
    return fits


def build_student_batch(
    processor: Qwen3VLProcessor,
    samples: list[EmbeddingInput],
    *,
    max_length: int,
) -> BatchFeature:
    try:
        return _build_processor_batch(
            processor,
            samples,
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    except ValueError as exc:
        if not is_student_overlength_error(exc):
            raise
        raise ValueError(
            f"{_STUDENT_OVERLENGTH_MESSAGE} "
            f"Increase max_length in config.toml (current: {max_length}; try 2048 or 4096)."
        ) from exc


def build_teacher_batch(
    processor: Qwen3VLProcessor,
    samples: list[EmbeddingInput],
    *,
    device: object | None = None,
) -> BatchFeature:
    return _build_processor_batch(
        processor,
        samples,
        padding=True,
        truncation=False,
        device=device,
    )
