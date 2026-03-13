from __future__ import annotations

from collections import defaultdict

from delta_embed_vl.evals.types import EvalResult

_TYPE_ORDER = [
    "Text -> Text",
    "Text -> Document",
    "Text -> Image",
    "Image -> Text",
    "Text -> Video",
]


def build_summary_table(results: list[EvalResult]) -> str:
    grouped: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        grouped[result.eval_type].append(result)

    ordered_results: list[tuple[str, list[EvalResult]]] = []
    for eval_type in _TYPE_ORDER:
        if grouped.get(eval_type):
            ordered_results.append((eval_type, grouped.pop(eval_type)))
    for eval_type in sorted(grouped):
        ordered_results.append((eval_type, grouped[eval_type]))

    benchmark_width = max(
        [len("Benchmark"), *[len(result.benchmark) for result in results]],
        default=len("Benchmark"),
    )
    type_width = max(
        [len("Type"), *[len(eval_type) for eval_type, _ in ordered_results]],
        default=len("Type"),
    )
    score_width = max(
        [len("Score"), *[len(_format_score(result.score)) for result in results]],
        default=len("Score"),
    )

    lines = [
        f"{'Type':<{type_width}}  {'Benchmark':<{benchmark_width}}  {'Score':>{score_width}}",
        "-" * (type_width + benchmark_width + score_width + 4),
    ]
    for eval_type, eval_results in ordered_results:
        for index, result in enumerate(eval_results):
            lines.append(
                f"{(eval_type if index == 0 else ''):<{type_width}}  "
                f"{result.benchmark:<{benchmark_width}}  "
                f"{_format_score(result.score):>{score_width}}"
            )
    return "\n".join(lines)


def _format_score(score: float) -> str:
    if score >= 100:
        return f"{score:.1f}"
    if score >= 10:
        return f"{score:.1f}"
    if score >= 1:
        return f"{score:.2f}"
    return f"{score:.4f}".rstrip("0").rstrip(".")
