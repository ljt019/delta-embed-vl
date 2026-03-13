from __future__ import annotations

import logging
from statistics import mean

import mteb

from delta_embed_vl.evals.encoder import DeltaEmbedEncoder
from delta_embed_vl.evals.types import EvalResult

logger = logging.getLogger(__name__)

_ENGLISH_BENCHMARK = "MTEB(eng, v2)"
_FAST_ENGLISH_BENCHMARK = "NanoBEIR"
_CODE_BENCHMARK = "MTEB(Code, v1)"
_FAST_CODE_BENCHMARK = "CodeRAG"


def run_english(
    encoder: DeltaEmbedEncoder,
    *,
    batch_size: int,
    fast: bool = False,
) -> list[EvalResult]:
    benchmark_name = _FAST_ENGLISH_BENCHMARK if fast else _ENGLISH_BENCHMARK
    score = _run_benchmark(
        encoder, benchmark_name=benchmark_name, batch_size=batch_size
    )
    return [
        EvalResult(
            eval_type="Text -> Text",
            benchmark="MTEB (English)",
            score=score,
            metric="avg",
        )
    ]


def run_code(
    encoder: DeltaEmbedEncoder,
    *,
    batch_size: int,
    fast: bool = False,
) -> list[EvalResult]:
    benchmark_name = _FAST_CODE_BENCHMARK if fast else _CODE_BENCHMARK
    score = _run_benchmark(
        encoder, benchmark_name=benchmark_name, batch_size=batch_size
    )
    return [
        EvalResult(
            eval_type="Text -> Text",
            benchmark="MTEB (Code)",
            score=score,
            metric="avg",
        )
    ]


def _run_benchmark(
    encoder: DeltaEmbedEncoder,
    *,
    benchmark_name: str,
    batch_size: int,
) -> float:
    benchmark = mteb.get_benchmark(benchmark_name)
    result = mteb.evaluate(
        encoder,
        benchmark.tasks,
        encode_kwargs={"batch_size": batch_size},
        overwrite_strategy="always",
        show_progress_bar=False,
    )
    task_scores = [task_result.get_score() for task_result in result.task_results]
    benchmark_score = float(mean(task_scores))
    logger.info(
        "%s: %.6f across %d task(s)", benchmark_name, benchmark_score, len(task_scores)
    )
    return benchmark_score
