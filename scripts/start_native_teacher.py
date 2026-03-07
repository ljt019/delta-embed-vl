# /// script
# dependencies = [
#   "fastapi",
#   "httpx",
#   "pillow",
#   "torch-c-dlpack-ext",
#   "transformers",
#   "uvicorn",
#   "vllm",
# ]
# ///
from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PIL import Image

DEFAULT_EMBED_INSTRUCTION = "Represent the user's input."
DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-8B"


@dataclass(frozen=True)
class ServerConfig:
    model: str
    host: str
    port: int
    dtype: str
    max_model_len: int
    data_parallel_size: int
    gpu_memory_utilization: float


def emit(event: str, **payload: object) -> None:
    print(json.dumps({"event": event, **payload}), flush=True)


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(
        description="Thin native AsyncLLMEngine teacher wrapper for Qwen3-VL embeddings."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--data-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    args = parser.parse_args()
    return ServerConfig(
        model=args.model,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        data_parallel_size=args.data_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def load_fastapi_symbols() -> tuple[Any, Any, Any, Any]:
    fastapi_module = importlib.import_module("fastapi")
    responses_module = importlib.import_module("fastapi.responses")
    return (
        fastapi_module.FastAPI,
        fastapi_module.HTTPException,
        fastapi_module.Request,
        responses_module.PlainTextResponse,
    )


def load_vllm_symbols() -> tuple[Any, Any, Any]:
    vllm_module = importlib.import_module("vllm")
    async_engine_module = importlib.import_module("vllm.engine.async_llm_engine")
    return (
        async_engine_module.AsyncLLMEngine,
        vllm_module.AsyncEngineArgs,
        vllm_module.PoolingParams,
    )


def load_transformer_symbols() -> Any:
    transformers_module = importlib.import_module("transformers")
    return transformers_module.AutoProcessor


def parse_data_uri_image(url: str) -> Image.Image:
    header, encoded = url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Only base64 data URIs are supported.")
    image_bytes = base64.b64decode(encoded)
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.convert("RGB")


async def load_image(url: str, *, image_client: Any) -> Image.Image:
    if url.startswith("data:"):
        return parse_data_uri_image(url)

    if url.startswith("http://") or url.startswith("https://"):
        response = await image_client.get(url)
        response.raise_for_status()
        with Image.open(io.BytesIO(response.content)) as image:
            return image.convert("RGB")

    if url.startswith("file://"):
        path = Path(url[7:])
    else:
        path = Path(url)

    with Image.open(path) as image:
        return image.convert("RGB")


def extract_text_parts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]

    if not isinstance(content, list):
        return []

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return parts


async def build_engine_input(payload: dict[str, Any], app: Any) -> dict[str, object]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a non-empty 'messages' list.")

    instruction_parts: list[str] = []
    user_content: list[dict[str, Any]] = []
    images: list[Image.Image] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        content = message.get("content")

        if role == "system":
            instruction_parts.extend(extract_text_parts(content))
            continue

        if role != "user":
            continue

        if isinstance(content, str):
            user_content.append({"type": "text", "text": content})
            continue

        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            if item_type == "text":
                user_content.append({"type": "text", "text": str(item.get("text", ""))})
                continue

            if item_type != "image_url":
                continue

            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                raw_url = image_url.get("url")
            else:
                raw_url = image_url
            if not isinstance(raw_url, str):
                raise ValueError("Expected image_url.url to be a string.")

            image = await load_image(raw_url, image_client=app.state.image_client)
            images.append(image)
            user_content.append({"type": "image", "image": image})

    if not user_content:
        user_content.append({"type": "text", "text": ""})

    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "\n".join(
                        part for part in instruction_parts if part
                    ).strip()
                    or DEFAULT_EMBED_INSTRUCTION,
                }
            ],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    prompt = cast(
        str,
        app.state.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        ),
    )

    engine_input: dict[str, object] = {"prompt": prompt}
    if images:
        engine_input["multi_modal_data"] = {
            "image": images[0] if len(images) == 1 else images,
        }
    return engine_input


def extract_embedding(output: object) -> list[float]:
    outputs = getattr(output, "outputs", None)
    embedding = getattr(outputs, "embedding", None)
    if embedding is None:
        raise ValueError(
            f"Could not extract embedding from output type {type(output)!r}"
        )
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    if not isinstance(embedding, list):
        raise ValueError("Embedding output was not list-like.")
    return [float(value) for value in embedding]


def render_metrics(app: Any) -> str:
    counters = app.state.counters
    return "\n".join(
        [
            "# TYPE native_teacher_requests_total counter",
            f"native_teacher_requests_total {counters['requests_total']}",
            "# TYPE native_teacher_requests_succeeded_total counter",
            f"native_teacher_requests_succeeded_total {counters['requests_succeeded_total']}",
            "# TYPE native_teacher_requests_failed_total counter",
            f"native_teacher_requests_failed_total {counters['requests_failed_total']}",
            "# TYPE native_teacher_requests_in_flight gauge",
            f"native_teacher_requests_in_flight {counters['requests_in_flight']}",
            "",
        ]
    )


def create_app(config: ServerConfig) -> Any:
    FastAPI, HTTPException, Request, PlainTextResponse = load_fastapi_symbols()

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        AsyncLLMEngine, AsyncEngineArgs, PoolingParams = load_vllm_symbols()
        AutoProcessor = load_transformer_symbols()
        httpx_module = importlib.import_module("httpx")

        processor = AutoProcessor.from_pretrained(
            config.model,
            trust_remote_code=True,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Processor did not expose a tokenizer.")

        engine_args = AsyncEngineArgs(
            model=config.model,
            runner="pooling",
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            data_parallel_size=config.data_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        app.state.engine = engine
        app.state.pooling_params = PoolingParams()
        app.state.tokenizer = tokenizer
        app.state.image_client = httpx_module.AsyncClient(timeout=30.0)
        app.state.counters = {
            "requests_total": 0,
            "requests_succeeded_total": 0,
            "requests_failed_total": 0,
            "requests_in_flight": 0,
        }

        emit(
            "native_teacher_started",
            model=config.model,
            host=config.host,
            port=config.port,
            data_parallel_size=config.data_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
        )

        try:
            yield
        finally:
            await app.state.image_client.aclose()
            app.state.engine.shutdown_background_loop()

    app = FastAPI(title="native-teacher", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "native-teacher",
                }
            ],
        }

    @app.get("/metrics")
    async def metrics() -> Any:
        return PlainTextResponse(render_metrics(app), media_type="text/plain")

    @app.post("/v1/embeddings")
    async def embeddings(request: Any) -> dict[str, object]:
        app.state.counters["requests_total"] += 1
        app.state.counters["requests_in_flight"] += 1
        try:
            payload = cast(dict[str, Any], await request.json())
            requested_model = payload.get("model")
            if requested_model is not None and requested_model != config.model:
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested model {requested_model!r} does not match {config.model!r}.",
                )

            engine_input = await build_engine_input(payload, app)
            request_id = uuid.uuid4().hex
            final_output: object | None = None
            async for output in app.state.engine.encode(
                prompt=engine_input,
                pooling_params=app.state.pooling_params,
                request_id=request_id,
            ):
                final_output = output

            if final_output is None:
                raise RuntimeError("Async engine returned no output.")

            embedding = extract_embedding(final_output)
            app.state.counters["requests_succeeded_total"] += 1
            return {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": embedding,
                    }
                ],
                "model": config.model,
            }
        except HTTPException:
            app.state.counters["requests_failed_total"] += 1
            raise
        except Exception as exc:
            app.state.counters["requests_failed_total"] += 1
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            app.state.counters["requests_in_flight"] -= 1

    return app


def main() -> None:
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    config = parse_args()
    uvicorn = importlib.import_module("uvicorn")
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
