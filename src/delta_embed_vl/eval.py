from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import wandb

from delta_embed_vl import cfg, configure_logging, resolve_attention, set_seed
from delta_embed_vl.evals.docci_eval import run_docci
from delta_embed_vl.evals.encoder import DeltaEmbedEncoder
from delta_embed_vl.evals.msrvtt_eval import run_msrvtt
from delta_embed_vl.evals.mteb_eval import run_code, run_english
from delta_embed_vl.evals.table import build_summary_table
from delta_embed_vl.evals.textcaps_eval import run_textcaps
from delta_embed_vl.evals.types import EvalResult
from delta_embed_vl.evals.vidore_eval import run_vidore

logger = logging.getLogger(__name__)

_SUITE_ORDER = [
    "mteb_english",
    "mteb_code",
    "vidore",
    "textcaps",
    "docci",
    "msrvtt",
]
_SUITE_ALIASES = {
    "mteb": ["mteb_english", "mteb_code"],
    "mteb-english": ["mteb_english"],
    "mteb-english-v2": ["mteb_english"],
    "mteb-code": ["mteb_code"],
    "mteb_code": ["mteb_code"],
    "vidore": ["vidore"],
    "textcaps": ["textcaps"],
    "docci": ["docci"],
    "msrvtt": ["msrvtt"],
    "msr-vtt": ["msrvtt"],
}


def eval_model(
    *,
    batch_size: int | None = None,
    fast: bool = False,
    suites: str | None = None,
) -> list[EvalResult]:
    model_path = Path("checkpoints") / cfg["name"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    resolved_batch_size = batch_size or cfg["eval"]["batch_size"]
    enabled_suites = _resolve_suites(suites)
    logger.info(
        "Running eval suites=%s batch_size=%d fast=%s",
        ",".join(enabled_suites),
        resolved_batch_size,
        fast,
    )

    encoder = DeltaEmbedEncoder(
        model_name=str(model_path),
        max_length=cfg["max_length"],
        attention=resolve_attention(cfg["attention"]),
    )
    runners = {
        "mteb_english": lambda: run_english(
            encoder,
            batch_size=resolved_batch_size,
            fast=fast,
        ),
        "mteb_code": lambda: run_code(
            encoder,
            batch_size=resolved_batch_size,
            fast=fast,
        ),
        "vidore": lambda: run_vidore(
            encoder,
            batch_size=resolved_batch_size,
            fast=fast,
        ),
        "textcaps": lambda: run_textcaps(
            encoder,
            batch_size=resolved_batch_size,
            fast=fast,
        ),
        "docci": lambda: run_docci(
            encoder,
            batch_size=resolved_batch_size,
            fast=fast,
        ),
        "msrvtt": lambda: run_msrvtt(
            encoder,
            batch_size=resolved_batch_size,
            fast=fast,
        ),
    }

    results: list[EvalResult] = []
    for suite in enabled_suites:
        results.extend(runners[suite]())

    table = build_summary_table(results)
    print(table)
    _save_results(
        results,
        output_dir=model_path,
        fast=fast,
        suites=enabled_suites,
        batch_size=resolved_batch_size,
    )
    _log_to_wandb(
        results,
        fast=fast,
        suites=enabled_suites,
        batch_size=resolved_batch_size,
    )
    return results


def _resolve_suites(suites: str | None) -> list[str]:
    if suites is None:
        return _SUITE_ORDER.copy()

    resolved: list[str] = []
    for raw_name in suites.split(","):
        name = raw_name.strip().lower()
        if not name:
            continue
        if name not in _SUITE_ALIASES:
            valid = ", ".join(sorted(_SUITE_ALIASES))
            raise ValueError(
                f"Unknown eval suite '{raw_name}'. Expected one of: {valid}"
            )
        for suite in _SUITE_ALIASES[name]:
            if suite not in resolved:
                resolved.append(suite)
    return resolved


def _save_results(
    results: list[EvalResult],
    *,
    output_dir: Path,
    fast: bool,
    suites: list[str],
    batch_size: int,
) -> None:
    output_path = output_dir / (
        "eval_results_fast.json" if fast else "eval_results.json"
    )
    payload = {
        "fast": fast,
        "batch_size": batch_size,
        "suites": suites,
        "results": [result.to_dict() for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved eval results to %s", output_path)


def _log_to_wandb(
    results: list[EvalResult],
    *,
    fast: bool,
    suites: list[str],
    batch_size: int,
) -> None:
    project = cfg["wandb"]["project"]
    if project is None:
        return

    wandb.init(
        project=project,
        job_type="eval",
        config={
            "model": cfg["name"],
            "fast": fast,
            "suites": suites,
            "batch_size": batch_size,
        },
    )
    try:
        metrics = {}
        for result in results:
            type_slug = _slugify(result.eval_type)
            benchmark_slug = _slugify(result.benchmark)
            metrics[f"eval/{type_slug}/{benchmark_slug}"] = result.score
        wandb.log(metrics)
    finally:
        wandb.finish()


def _slugify(value: str) -> str:
    slug = value.lower()
    for old, new in ((" -> ", "_to_"), (" ", "_"), ("(", ""), (")", "")):
        slug = slug.replace(old, new)
    return slug


def eval_model_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Eval batch size (default: eval.batch_size = {cfg['eval']['batch_size']})",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Run quick representative subsets instead of the full eval suites.",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default=None,
        help="Comma-separated suites: mteb, vidore, textcaps, docci, msrvtt.",
    )
    args = parser.parse_args()

    configure_logging()
    set_seed(cfg["train"]["seed"])
    eval_model(batch_size=args.batch_size, fast=args.fast, suites=args.suites)


if __name__ == "__main__":
    eval_model_cli()
