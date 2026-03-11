import argparse

from mteb.results import ModelResult

from delta_embed_vl import configure_logging, resolve_attention, set_seed
from delta_embed_vl.evals.mteb_eval import run_eval as run_mteb_eval
from delta_embed_vl.settings import Settings

_DEFAULT_EVAL_BATCH_SIZE = 16
_SETTINGS = Settings()


def eval_model(
    *,
    model_path: str = "checkpoints",
    eval_batch_size: int = _DEFAULT_EVAL_BATCH_SIZE,
    max_length: int = _SETTINGS.student_max_length,
    student_device: str | None = None,
    attention: str | None = None,
) -> ModelResult:
    return run_mteb_eval(
        model_path=model_path,
        eval_batch_size=eval_batch_size,
        max_length=max_length,
        device=student_device,
        attention=attention,
    )


def eval_model_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints")
    parser.add_argument("--eval-batch-size", type=int, default=_DEFAULT_EVAL_BATCH_SIZE)
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

    configure_logging()
    set_seed(args.seed)
    eval_model(
        model_path=args.model_path,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        student_device=args.student_device,
        attention=resolve_attention(args.attention),
    )


if __name__ == "__main__":
    eval_model_cli()
