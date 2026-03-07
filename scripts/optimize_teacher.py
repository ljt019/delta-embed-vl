from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

from delta_embed_vl.data.media import coerce_image_to_rgb, image_to_data_uri
from delta_embed_vl.model.embedding_inputs import (
    DEFAULT_EMBED_INSTRUCTION,
    EmbeddingInput,
)
from delta_embed_vl.settings import Settings

settings = Settings()

RUNNING_RE = re.compile(r'^vllm:num_requests_running\{engine="(\d+)".*\} ([0-9.]+)$')
WAITING_RE = re.compile(r'^vllm:num_requests_waiting\{engine="(\d+)".*\} ([0-9.]+)$')
KV_RE = re.compile(r'^vllm:kv_cache_usage_perc\{engine="(\d+)".*\} ([0-9.]+)$')
NATIVE_IN_FLIGHT_RE = re.compile(r"^native_teacher_requests_in_flight ([0-9.]+)$")

DEFAULT_SWEEP_CONCURRENCY = [64, 80, 96, 112, 128]


@dataclass(frozen=True)
class Scenario:
    name: str
    text: str
    image_size: tuple[int, int] | None


SCENARIOS: dict[str, Scenario] = {
    "text_short": Scenario(
        name="text_short",
        text="Short embedding request for throughput probing.",
        image_size=None,
    ),
    "text_long": Scenario(
        name="text_long",
        text=(
            "Longer embedding request with dense instructional text, answer grounding, "
            "labels, legends, numeric values, and retrieval style phrasing. "
        )
        * 48,
        image_size=None,
    ),
    "image_64_text_short": Scenario(
        name="image_64_text_short",
        text="Short multimodal embedding request with a tiny synthetic image.",
        image_size=(64, 64),
    ),
    "image_224_text_short": Scenario(
        name="image_224_text_short",
        text="Representative multimodal request with moderate image size and short text.",
        image_size=(224, 224),
    ),
    "image_448_text_short": Scenario(
        name="image_448_text_short",
        text="Representative multimodal request with larger image size and short text.",
        image_size=(448, 448),
    ),
    "image_224_text_long": Scenario(
        name="image_224_text_long",
        text=(
            "Chart question with visual context, numeric reasoning, labels, legends, "
            "OCR-like snippets, and answer grounding. "
        )
        * 24,
        image_size=(224, 224),
    ),
}


def emit(event: str, **payload: object) -> None:
    print(json.dumps({"event": event, **payload}), flush=True)


def parse_concurrency_values(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one concurrency value is required.")
    return values


def teacher_urls(base_url: str) -> tuple[str, str]:
    stripped = base_url.rstrip("/")
    root = stripped.rsplit("/v1", 1)[0] if stripped.endswith("/v1") else stripped
    return f"{stripped}/embeddings", f"{root}/metrics"


def build_base_image(
    scenario: Scenario,
    *,
    image_path: Path | None,
) -> Image.Image | None:
    if image_path is not None:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    if scenario.image_size is None:
        return None

    width, height = scenario.image_size
    return Image.new("RGB", (width, height), color=(192, 48, 48))


def build_sample(
    scenario: Scenario,
    *,
    request_index: int,
    base_image: Image.Image | None,
) -> EmbeddingInput:
    text = f"{request_index}: {scenario.text}"
    image = base_image.copy() if base_image is not None else None
    return EmbeddingInput(text=text, image=image)


def build_payload(sample: EmbeddingInput) -> dict[str, object]:
    user_content: list[dict[str, object]] = []
    resolved_image = coerce_image_to_rgb(sample.image)
    if sample.image is not None and resolved_image is None:
        raise ValueError("Could not resolve image for teacher payload.")

    if resolved_image is not None:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_uri(resolved_image)},
            }
        )

    user_content.append({"type": "text", "text": sample.text or ""})

    return {
        "model": settings.teacher_model,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": sample.instruction or DEFAULT_EMBED_INSTRUCTION,
                    }
                ],
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ],
        "encoding_format": "float",
        "continue_final_message": True,
        "add_special_tokens": True,
    }


def parse_metrics(text: str) -> dict[str, float]:
    running: dict[int, float] = {}
    waiting: dict[int, float] = {}
    kv: dict[int, float] = {}
    native_in_flight: float | None = None
    for line in text.splitlines():
        running_match = RUNNING_RE.match(line)
        if running_match:
            running[int(running_match.group(1))] = float(running_match.group(2))
            continue

        waiting_match = WAITING_RE.match(line)
        if waiting_match:
            waiting[int(waiting_match.group(1))] = float(waiting_match.group(2))
            continue

        kv_match = KV_RE.match(line)
        if kv_match:
            kv[int(kv_match.group(1))] = float(kv_match.group(2))
            continue

        native_match = NATIVE_IN_FLIGHT_RE.match(line)
        if native_match:
            native_in_flight = float(native_match.group(1))

    if native_in_flight is not None and not running:
        return {
            "running_total": native_in_flight,
            "waiting_total": 0.0,
            "kv_avg": 0.0,
        }

    return {
        "running_total": sum(running.values()),
        "waiting_total": sum(waiting.values()),
        "kv_avg": statistics.mean(kv.values()) if kv else 0.0,
    }


def quantile_95(values: list[float]) -> float | None:
    if len(values) < 20:
        return None
    return statistics.quantiles(values, n=20)[18]


def benchmark_image_encode(
    scenario: Scenario,
    *,
    iterations: int,
    image_path: Path | None,
) -> None:
    base_image = build_base_image(scenario, image_path=image_path)
    if base_image is None:
        emit(
            "image_encode_skipped",
            scenario=scenario.name,
            reason="scenario_has_no_image",
        )
        return

    encoded_sizes: list[int] = []
    start = time.perf_counter()
    for _ in range(iterations):
        encoded = image_to_data_uri(base_image.copy())
        encoded_sizes.append(len(encoded))
    elapsed = time.perf_counter() - start
    emit(
        "image_encode_benchmark",
        scenario=scenario.name,
        iterations=iterations,
        elapsed_s=round(elapsed, 3),
        encodes_per_s=round(iterations / elapsed, 1),
        avg_encoded_chars=round(statistics.mean(encoded_sizes), 1),
    )


def benchmark_payload_build(
    scenario: Scenario,
    *,
    iterations: int,
    image_path: Path | None,
) -> None:
    base_image = build_base_image(scenario, image_path=image_path)
    json_sizes: list[int] = []
    start = time.perf_counter()
    for index in range(iterations):
        payload = build_payload(
            build_sample(scenario, request_index=index, base_image=base_image)
        )
        json_sizes.append(len(json.dumps(payload)))
    elapsed = time.perf_counter() - start
    emit(
        "payload_build_benchmark",
        scenario=scenario.name,
        iterations=iterations,
        elapsed_s=round(elapsed, 3),
        payloads_per_s=round(iterations / elapsed, 1),
        avg_json_chars=round(statistics.mean(json_sizes), 1),
    )


async def request_with_prebuilt_payload(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    payload: dict[str, object],
) -> dict[str, object]:
    async with semaphore:
        try:
            response = await client.post(url, json=payload)
            body = response.json()
        except Exception as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}: {str(exc)[:200]}"}

        if response.is_success and isinstance(body, dict) and "data" in body:
            return {"ok": True}
        return {"ok": False, "detail": f"{response.status_code}: {str(body)[:200]}"}


async def request_with_build_on_fly(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    *,
    scenario: Scenario,
    request_index: int,
    base_image: Image.Image | None,
) -> dict[str, object]:
    payload = build_payload(
        build_sample(scenario, request_index=request_index, base_image=base_image)
    )
    return await request_with_prebuilt_payload(client, semaphore, url, payload)


async def poll_metrics(
    metrics_url: str,
    *,
    stop_event: asyncio.Event,
    poll_interval_s: float,
    samples: list[dict[str, float]],
) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        while not stop_event.is_set():
            try:
                response = await client.get(metrics_url)
                samples.append(parse_metrics(response.text))
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval_s)
            except TimeoutError:
                pass


async def run_request_benchmark(
    *,
    mode: str,
    scenario: Scenario,
    requests: int,
    concurrency: int,
    base_url: str,
    image_path: Path | None,
    poll_metrics_enabled: bool,
    poll_interval_s: float,
    timeout_s: float,
) -> dict[str, object]:
    embeddings_url, metrics_url = teacher_urls(base_url)
    base_image = build_base_image(scenario, image_path=image_path)
    semaphore = asyncio.Semaphore(concurrency)
    stop_event = asyncio.Event()
    metric_samples: list[dict[str, float]] = []
    poll_task: asyncio.Task[None] | None = None

    limits = httpx.Limits(
        max_connections=max(concurrency * 2, 256),
        max_keepalive_connections=max(concurrency * 2, 256),
    )

    if poll_metrics_enabled:
        poll_task = asyncio.create_task(
            poll_metrics(
                metrics_url,
                stop_event=stop_event,
                poll_interval_s=poll_interval_s,
                samples=metric_samples,
            )
        )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s), limits=limits
    ) as client:
        start = time.perf_counter()
        if mode == "prebuilt":
            payloads = [
                build_payload(
                    build_sample(scenario, request_index=index, base_image=base_image)
                )
                for index in range(requests)
            ]
            results = await asyncio.gather(
                *[
                    request_with_prebuilt_payload(
                        client, semaphore, embeddings_url, payload
                    )
                    for payload in payloads
                ]
            )
        else:
            results = await asyncio.gather(
                *[
                    request_with_build_on_fly(
                        client,
                        semaphore,
                        embeddings_url,
                        scenario=scenario,
                        request_index=index,
                        base_image=base_image,
                    )
                    for index in range(requests)
                ]
            )
        elapsed = time.perf_counter() - start

    stop_event.set()
    if poll_task is not None:
        await poll_task

    ok = sum(1 for result in results if result["ok"] is True)
    failures = [result["detail"] for result in results if result["ok"] is not True]
    running_values = [sample["running_total"] for sample in metric_samples]
    waiting_values = [sample["waiting_total"] for sample in metric_samples]
    kv_values = [sample["kv_avg"] for sample in metric_samples]
    p95_running_total = quantile_95(running_values)

    benchmark: dict[str, object] = {
        "scenario": scenario.name,
        "mode": mode,
        "requests": requests,
        "concurrency": concurrency,
        "ok": ok,
        "fail": len(failures),
        "elapsed_s": round(elapsed, 3),
        "emb_per_s": round(ok / elapsed, 1),
        "first_error": failures[0] if failures else None,
        "metrics_url": metrics_url if poll_metrics_enabled else None,
        "metric_polls": len(metric_samples),
        "max_running_total": max(running_values, default=None),
        "avg_running_total": round(statistics.mean(running_values), 2)
        if running_values
        else None,
        "p95_running_total": round(p95_running_total, 2)
        if p95_running_total is not None
        else None,
        "max_waiting_total": max(waiting_values, default=None),
        "avg_waiting_total": round(statistics.mean(waiting_values), 2)
        if waiting_values
        else None,
        "max_kv_avg": round(max(kv_values, default=0.0), 4),
    }
    return benchmark


def benchmark_request_mode(
    scenario: Scenario,
    *,
    mode: str,
    requests: int,
    concurrency: int,
    repeats: int,
    base_url: str,
    image_path: Path | None,
    poll_metrics_enabled: bool,
    poll_interval_s: float,
    timeout_s: float,
) -> None:
    for repeat in range(repeats):
        result = asyncio.run(
            run_request_benchmark(
                mode=mode,
                scenario=scenario,
                requests=requests,
                concurrency=concurrency,
                base_url=base_url,
                image_path=image_path,
                poll_metrics_enabled=poll_metrics_enabled,
                poll_interval_s=poll_interval_s,
                timeout_s=timeout_s,
            )
        )
        emit("request_benchmark", repeat=repeat + 1, **result)


def benchmark_sweep(
    scenario: Scenario,
    *,
    mode: str,
    requests: int,
    concurrency_values: list[int],
    repeats: int,
    base_url: str,
    image_path: Path | None,
    poll_metrics_enabled: bool,
    poll_interval_s: float,
    timeout_s: float,
) -> None:
    for concurrency in concurrency_values:
        for repeat in range(repeats):
            result = asyncio.run(
                run_request_benchmark(
                    mode=mode,
                    scenario=scenario,
                    requests=requests,
                    concurrency=concurrency,
                    base_url=base_url,
                    image_path=image_path,
                    poll_metrics_enabled=poll_metrics_enabled,
                    poll_interval_s=poll_interval_s,
                    timeout_s=timeout_s,
                )
            )
            emit("sweep_result", repeat=repeat + 1, **result)


def benchmark_suite(
    scenario: Scenario,
    *,
    image_path: Path | None,
    base_url: str,
    timeout_s: float,
) -> None:
    emit(
        "suite_plan",
        scenario=scenario.name,
        steps=[
            "image_encode",
            "payload_build",
            "request_prebuilt",
            "request_build_on_fly",
            "sweep_prebuilt_with_metrics",
        ],
    )
    benchmark_image_encode(scenario, iterations=100, image_path=image_path)
    benchmark_payload_build(scenario, iterations=200, image_path=image_path)
    benchmark_request_mode(
        scenario,
        mode="prebuilt",
        requests=512,
        concurrency=96,
        repeats=1,
        base_url=base_url,
        image_path=image_path,
        poll_metrics_enabled=False,
        poll_interval_s=0.1,
        timeout_s=timeout_s,
    )
    benchmark_request_mode(
        scenario,
        mode="build",
        requests=512,
        concurrency=96,
        repeats=1,
        base_url=base_url,
        image_path=image_path,
        poll_metrics_enabled=False,
        poll_interval_s=0.1,
        timeout_s=timeout_s,
    )
    benchmark_sweep(
        scenario,
        mode="prebuilt",
        requests=1024,
        concurrency_values=DEFAULT_SWEEP_CONCURRENCY,
        repeats=1,
        base_url=base_url,
        image_path=image_path,
        poll_metrics_enabled=True,
        poll_interval_s=0.1,
        timeout_s=timeout_s,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated teacher optimization harness for client/server throughput tests."
    )
    parser.add_argument(
        "--base-url",
        default=settings.teacher_base_url,
        help="Teacher base URL, defaulting to settings.teacher_base_url.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        help="Optional real image path to use instead of synthetic generated images.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("image-encode", "payload-build"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
        subparser.add_argument("--iterations", type=int, default=200)

    for name in ("request-prebuilt", "request-build"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
        subparser.add_argument("--requests", type=int, default=512)
        subparser.add_argument("--concurrency", type=int, default=96)
        subparser.add_argument("--repeats", type=int, default=1)
        subparser.add_argument("--poll-metrics", action="store_true")
        subparser.add_argument("--poll-interval-s", type=float, default=0.1)
        subparser.add_argument("--timeout-s", type=float, default=120.0)

    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    sweep.add_argument("--mode", choices=("prebuilt", "build"), default="prebuilt")
    sweep.add_argument("--requests", type=int, default=1024)
    sweep.add_argument("--repeats", type=int, default=1)
    sweep.add_argument(
        "--concurrency-values",
        default="64,80,96,112,128",
        help="Comma-separated list, for example 64,80,96,112,128.",
    )
    sweep.add_argument("--poll-metrics", action="store_true")
    sweep.add_argument("--poll-interval-s", type=float, default=0.1)
    sweep.add_argument("--timeout-s", type=float, default=120.0)

    suite = subparsers.add_parser("suite")
    suite.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    suite.add_argument("--timeout-s", type=float, default=120.0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    scenario = SCENARIOS[getattr(args, "scenario")]

    if args.command == "image-encode":
        benchmark_image_encode(
            scenario,
            iterations=args.iterations,
            image_path=args.image_path,
        )
        return

    if args.command == "payload-build":
        benchmark_payload_build(
            scenario,
            iterations=args.iterations,
            image_path=args.image_path,
        )
        return

    if args.command in {"request-prebuilt", "request-build"}:
        benchmark_request_mode(
            scenario,
            mode="prebuilt" if args.command == "request-prebuilt" else "build",
            requests=args.requests,
            concurrency=args.concurrency,
            repeats=args.repeats,
            base_url=args.base_url,
            image_path=args.image_path,
            poll_metrics_enabled=args.poll_metrics,
            poll_interval_s=args.poll_interval_s,
            timeout_s=args.timeout_s,
        )
        return

    if args.command == "sweep":
        benchmark_sweep(
            scenario,
            mode=args.mode,
            requests=args.requests,
            concurrency_values=parse_concurrency_values(args.concurrency_values),
            repeats=args.repeats,
            base_url=args.base_url,
            image_path=args.image_path,
            poll_metrics_enabled=args.poll_metrics,
            poll_interval_s=args.poll_interval_s,
            timeout_s=args.timeout_s,
        )
        return

    benchmark_suite(
        scenario,
        image_path=args.image_path,
        base_url=args.base_url,
        timeout_s=args.timeout_s,
    )


if __name__ == "__main__":
    main()
