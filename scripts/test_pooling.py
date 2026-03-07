# /// script
# dependencies = [
#   "pillow",
#   "torch-c-dlpack-ext",
#   "transformers",
#   "vllm",
# ]
# ///
from __future__ import annotations

import argparse
import asyncio
import importlib
import time
from typing import Any
from uuid import uuid4

from PIL import Image
from transformers import AutoProcessor

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_INSTRUCTION = "Represent the user's input."
DEFAULT_TEXT = "A red square used to validate multimodal async pooling."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe AsyncLLMEngine.encode() for multimodal pooling."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def _build_prompt(
    processor: Any,
    *,
    instruction: str,
    text: str | None = None,
    image: Image.Image | None = None,
) -> str:
    user_content: list[dict[str, Any]] = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    if text:
        user_content.append({"type": "text", "text": text})
    if not user_content:
        user_content.append({"type": "text", "text": ""})

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    rendered = processor.tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    return str(rendered)


async def _collect_embedding(
    engine: Any,
    *,
    prompt: object,
    request_id: str,
    pooling_params_cls: type[Any],
) -> list[float]:
    final_output = None
    async for output in engine.encode(
        prompt=prompt,
        pooling_params=pooling_params_cls(),
        request_id=request_id,
    ):
        final_output = output

    if final_output is None:
        raise RuntimeError(f"No pooling output returned for request {request_id}.")

    embedding = final_output.outputs.embedding
    return list(embedding)


async def _run_probe(args: argparse.Namespace) -> None:
    PoolingParams = importlib.import_module("vllm").PoolingParams
    AsyncEngineArgs = importlib.import_module("vllm.engine.arg_utils").AsyncEngineArgs
    AsyncLLMEngine = importlib.import_module(
        "vllm.engine.async_llm_engine"
    ).AsyncLLMEngine

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    image = Image.new("RGB", (args.image_size, args.image_size), color="red")

    text_prompt = _build_prompt(
        processor,
        instruction=DEFAULT_INSTRUCTION,
        text=args.text,
    )
    image_text_prompt = _build_prompt(
        processor,
        instruction=DEFAULT_INSTRUCTION,
        text=args.text,
        image=image,
    )

    engine_args = AsyncEngineArgs(
        model=args.model,
        runner="pooling",
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        data_parallel_size=args.data_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def one_multimodal(index: int) -> list[float]:
        payload = {
            "prompt": image_text_prompt.replace(args.text, f"{args.text} #{index}", 1),
            "multi_modal_data": {"image": image.copy()},
        }
        return await _collect_embedding(
            engine,
            prompt=payload,
            request_id=f"multimodal-{index}-{uuid4().hex}",
            pooling_params_cls=PoolingParams,
        )

    try:
        text_embedding = await _collect_embedding(
            engine,
            prompt=text_prompt,
            request_id=f"text-{uuid4().hex}",
            pooling_params_cls=PoolingParams,
        )
        multimodal_embedding = await _collect_embedding(
            engine,
            prompt={
                "prompt": image_text_prompt,
                "multi_modal_data": {"image": image.copy()},
            },
            request_id=f"image-text-{uuid4().hex}",
            pooling_params_cls=PoolingParams,
        )

        print("single_request_checks")
        print(
            {
                "text_dim": len(text_embedding),
                "text_head": text_embedding[:8],
                "multimodal_dim": len(multimodal_embedding),
                "multimodal_head": multimodal_embedding[:8],
            }
        )

        semaphore = asyncio.Semaphore(args.concurrency)

        async def worker(index: int) -> list[float]:
            async with semaphore:
                return await one_multimodal(index)

        started = time.perf_counter()
        embeddings = await asyncio.gather(
            *[worker(index) for index in range(args.num_requests)]
        )
        elapsed = time.perf_counter() - started

        print("concurrent_multimodal_check")
        print(
            {
                "requests": args.num_requests,
                "concurrency": args.concurrency,
                "elapsed_s": round(elapsed, 3),
                "emb_per_s": round(args.num_requests / elapsed, 1),
                "embedding_dim": len(embeddings[0]),
            }
        )
    finally:
        shutdown = getattr(engine, "shutdown_background_loop", None)
        if callable(shutdown):
            shutdown()


def main() -> None:
    args = _parse_args()
    asyncio.run(_run_probe(args))


if __name__ == "__main__":
    main()
