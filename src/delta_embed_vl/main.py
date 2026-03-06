import argparse
import logging
from dataclasses import dataclass

from delta_embed_vl.data.download import download_data
from delta_embed_vl.data.preprocess import preprocess_data
from delta_embed_vl.eval.mteb_eval import run_eval
from delta_embed_vl.teacher.generate import embed_all
from delta_embed_vl.training.distill import train

logger = logging.getLogger(__name__)

_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


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
    )


def prepare_data(*, limit: int | None = None):
    """Download raw data, preprocess to Arrow, generate teacher embeddings."""
    logger.info("Downloading raw datasets")
    download_data(limit=limit)

    logger.info("Preprocessing datasets")
    preprocess_data(limit=limit)

    logger.info("Generating teacher embeddings")
    embed_all(limit=limit)

    logger.info("Data preparation complete")


def prepare_data_cli():
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
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
):
    """Train student via cosine distillation on all available prepared data."""
    train(
        limit=limit,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        grad_accum_steps=grad_accum_steps,
        save_dir=save_dir,
    )


def train_model_cli():
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
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
    )


def eval_model(*, model_path: str = "checkpoints"):
    """Run MTEB eval on the latest checkpoint directory."""
    run_eval(model_path=model_path)


def eval_model_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    eval_model(model_path=args.model_path)


def main():
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
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
    )
    eval_model(model_path=args.save_dir)


if __name__ == "__main__":
    main()
