import argparse
import logging

from delta_embed_vl import cfg, configure_logging, resolve_attention
from delta_embed_vl.data.build import build_dataset

logger = logging.getLogger(__name__)


def prepare_data(*, limit: int | None = None, push_to_hub: bool = False) -> None:
    build_dataset(
        limit=limit,
        limit_all=limit is None,
        max_length=cfg["max_length"],
        teacher_batch_size=cfg["data"]["batch_size"],
        attention=resolve_attention(cfg["attention"]),
        push_to_hub=push_to_hub,
    )


def prepare_data_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows per source (default: all)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help=f"Push to Hub as {cfg['data']['id']}",
    )
    args = parser.parse_args()

    configure_logging()
    prepare_data(limit=args.limit, push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    prepare_data_cli()
