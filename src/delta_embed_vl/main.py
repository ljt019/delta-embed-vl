import argparse
import logging
from dataclasses import dataclass

from datasets import disable_progress_bars

from delta_embed_vl.data.preprocess import preprocess_data
from delta_embed_vl.eval.mteb_eval import run_eval
from delta_embed_vl.training.distill import train

logger = logging.getLogger(__name__)

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


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT, force=True)
    disable_progress_bars()
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


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
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--teacher-device", type=str, default=None)
    parser.add_argument("--student-device", type=str, default=None)
    parser.add_argument("--teacher-batch-size", type=int, default=None)


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
    )


def prepare_data(*, limit: int | None = None):
    """Preprocess datasets to Arrow caches."""
    logger.info("Preprocessing datasets")
    preprocess_data(limit=limit)

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
    max_length: int = 512,
    grad_accum_steps: int = 1,
    save_dir: str = "checkpoints",
    teacher_device: str | None = None,
    student_device: str | None = None,
    teacher_batch_size: int | None = None,
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
    )


def train_model_cli():
    _configure_logging()
    args = _parse_train_run_args(include_limit=True)
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
    )


def eval_model(*, model_path: str = "checkpoints"):
    """Run MTEB eval on the latest checkpoint directory."""
    run_eval(model_path=model_path)


def eval_model_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints")
    args = parser.parse_args()
    _configure_logging()
    eval_model(model_path=args.model_path)


def main():
    _configure_logging()
    args = _parse_train_run_args(include_limit=True)
    prepare_data(limit=args.limit)
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
    )
    eval_model(model_path=args.save_dir)


if __name__ == "__main__":
    main()
