from __future__ import annotations

import logging
from statistics import mean

import mteb

from delta_embed_vl.evals.encoder import DeltaEmbedEncoder
from delta_embed_vl.evals.types import EvalResult

logger = logging.getLogger(__name__)

_VIDORE_BENCHMARK = "ViDoRe(v2)"


def run_vidore(
    encoder: DeltaEmbedEncoder,
    *,
    batch_size: int,
    fast: bool = False,
) -> list[EvalResult]:
    benchmark = mteb.get_benchmark(_VIDORE_BENCHMARK)
    tasks = benchmark.tasks[:1] if fast else benchmark.tasks
    result = mteb.evaluate(
        encoder,
        tasks,
        encode_kwargs={"batch_size": batch_size},
        overwrite_strategy="always",
        show_progress_bar=False,
    )
    task_scores = [task_result.get_score() for task_result in result.task_results]
    score = float(mean(task_scores))
    logger.info(
        "%s: %.6f across %d task(s)", _VIDORE_BENCHMARK, score, len(task_scores)
    )
    return [
        EvalResult(
            eval_type="Text -> Document",
            benchmark="ViDoRe v2",
            score=score,
            metric="ndcg@5",
        )
    ]
