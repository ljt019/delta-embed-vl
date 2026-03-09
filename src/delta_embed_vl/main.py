import argparse
import logging
import random
from dataclasses import dataclass

import numpy as np
import torch
from datasets import disable_progress_bars

from delta_embed_vl.data.prepare import prepare_data as prepare_cached_data
from delta_embed_vl.eval.mteb_eval import run_eval
from delta_embed_vl.settings import Settings
from delta_embed_vl.training.distill import train

logger = logging.getLogger(__name__)
_SETTINGS = Settings()

_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_NOISY_LOGGERS = (
    "datasets",
    "fsspec",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "urllib3",
)


@dataclass(frozen=True)
class TrainRunArgs:
    limit: int | None
    epochs: int
    batch_size: int
    lr: float
    warmup_ratio: float
    max_length: int
    grad_accum_steps: int
    save_dir: str
    teacher_device: str | None
    student_device: str | None
    teacher_batch_size: int | None
    eval_batch_size: int
    seed: int
    attention: str | None


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT, force=True)
    disable_progress_bars()
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_limit_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only first N rows per dataset for test runs.",
    )


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=_SETTINGS.student_max_length)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=_SETTINGS.seed)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--attention",
        choices=("sdpa", "fa"),
        default=None,
        help="Force attention backend for teacher/student/eval.",
    )
    parser.add_argument("--teacher-device", type=str, default=None)
    parser.add_argument("--student-device", type=str, default=None)
    parser.add_argument("--teacher-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=16)


def _parse_limit() -> int | None:
    parser = argparse.ArgumentParser()
    _add_limit_arg(parser)
    return parser.parse_args().limit


def _parse_train_run_args(*, include_limit: bool) -> TrainRunArgs:
    parser = argparse.ArgumentParser()
    if include_limit:
        _add_limit_arg(parser)
    _add_train_args(parser)
    args = parser.parse_args()
    return TrainRunArgs(
        limit=getattr(args, "limit", None),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        grad_accum_steps=args.grad_accum_steps,
        save_dir=args.save_dir,
        teacher_device=args.teacher_device,
        student_device=args.student_device,
        teacher_batch_size=args.teacher_batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        attention=_resolve_attention(args.attention),
    )


def _resolve_attention(attention: str | None) -> str | None:
    if attention == "fa":
        return "flash_attention_2"
    return attention


def prepare_data(
    *,
    limit: int | None = None,
    max_length: int = _SETTINGS.student_max_length,
):
    """Preprocess datasets to Arrow caches."""
    logger.info("Preprocessing datasets")
    prepare_cached_data(limit=limit, student_max_length=max_length)

    logger.info("Data preparation complete")


def prepare_data_cli():
    _configure_logging()
    prepare_data(limit=_parse_limit())


def train_model(
    *,
    limit: int | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = _SETTINGS.student_max_length,
    grad_accum_steps: int = 1,
    save_dir: str = "checkpoints",
    teacher_device: str | None = None,
    student_device: str | None = None,
    teacher_batch_size: int | None = None,
    attention: str | None = None,
):
    """Train student via local teacher-student cosine distillation."""
    train(
        limit=limit,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        grad_accum_steps=grad_accum_steps,
        save_dir=save_dir,
        teacher_device=teacher_device,
        student_device=student_device,
        teacher_batch_size=teacher_batch_size,
        attention=attention,
    )


def train_model_cli():
    _configure_logging()
    args = _parse_train_run_args(include_limit=True)
    _set_seed(args.seed)
    train_model(
        limit=args.limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        grad_accum_steps=args.grad_accum_steps,
        save_dir=args.save_dir,
        teacher_device=args.teacher_device,
        student_device=args.student_device,
        teacher_batch_size=args.teacher_batch_size,
        attention=args.attention,
    )


def eval_model(
    *,
    model_path: str = "checkpoints",
    eval_batch_size: int = 16,
    max_length: int = _SETTINGS.student_max_length,
    student_device: str | None = None,
    attention: str | None = None,
):
    """Run MTEB eval on the latest checkpoint directory."""
    run_eval(
        model_path=model_path,
        eval_batch_size=eval_batch_size,
        max_length=max_length,
        device=student_device,
        attention=attention,
    )


def eval_model_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=_SETTINGS.seed)
    parser.add_argument("--student-device", type=str, default=None)
    parser.add_argument(
        "--attention",
        choices=("sdpa", "fa"),
        default=None,
        help="Force eval attention backend.",
    )
    parser.add_argument("--max-length", type=int, default=_SETTINGS.student_max_length)
    args = parser.parse_args()
    _configure_logging()
    _set_seed(args.seed)
    eval_model(
        model_path=args.model_path,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        student_device=args.student_device,
        attention=_resolve_attention(args.attention),
    )


def main():
    _configure_logging()
    args = _parse_train_run_args(include_limit=True)
    _set_seed(args.seed)
    prepare_data(limit=args.limit, max_length=args.max_length)
    train_model(
        limit=args.limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        grad_accum_steps=args.grad_accum_steps,
        save_dir=args.save_dir,
        teacher_device=args.teacher_device,
        student_device=args.student_device,
        teacher_batch_size=args.teacher_batch_size,
        attention=args.attention,
    )
    eval_model(
        model_path=args.save_dir,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        student_device=args.student_device,
        attention=args.attention,
    )


if __name__ == "__main__":
    main()
