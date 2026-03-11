import argparse
import logging

import torch

from delta_embed_vl import configure_logging, resolve_attention
from delta_embed_vl.data.build import build_dataset
from delta_embed_vl.settings import Settings

logger = logging.getLogger(__name__)
_SETTINGS = Settings()
_LIMIT_ALL = "all"


def _auto_teacher_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda:0"


def prepare_data(
    *,
    limit: int | None = None,
    limit_all: bool = False,
    max_length: int = _SETTINGS.student_max_length,
    teacher_device: str | None = None,
    teacher_batch_size: int = 8,
    attention: str | None = None,
    push_to_hub: bool = False,
) -> None:
    build_dataset(
        limit=limit,
        limit_all=limit_all,
        max_length=max_length,
        teacher_device=teacher_device or _auto_teacher_device(),
        teacher_batch_size=teacher_batch_size,
        attention=attention,
        push_to_hub=push_to_hub,
    )


def _parse_limit(value: str) -> int | str:
    if value.lower() == _LIMIT_ALL:
        return _LIMIT_ALL
    return int(value)


def prepare_data_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=_parse_limit,
        default=None,
        help="N rows per dataset, or 'all' for full Cauldron + balanced Wikipedia.",
    )
    parser.add_argument("--max-length", type=int, default=_SETTINGS.student_max_length)
    parser.add_argument("--teacher-device", type=str, default=None)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help=f"Push to Hub as {_SETTINGS.hub_dataset}",
    )
    parser.add_argument(
        "--attention",
        choices=("sdpa", "fa"),
        default=None,
        help="Force teacher attention backend during dataset build.",
    )
    args = parser.parse_args()

    limit_all = args.limit == _LIMIT_ALL
    limit = None if limit_all else args.limit

    configure_logging()
    prepare_data(
        limit=limit,
        limit_all=limit_all,
        max_length=args.max_length,
        teacher_device=args.teacher_device,
        teacher_batch_size=args.teacher_batch_size,
        attention=resolve_attention(args.attention),
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    prepare_data_cli()
