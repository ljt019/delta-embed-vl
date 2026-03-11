import argparse
import logging

from delta_embed_vl import cfg, configure_logging, resolve_attention
from delta_embed_vl.data.build import build_dataset

logger = logging.getLogger(__name__)


def prepare_data(
    *,
    limit: int | None = None,
    push_to_hub: bool = False,
    no_stream: bool = False,
    rebuild_normalized: bool = False,
    detailed_timings: bool = False,
) -> None:
    build_dataset(
        limit=limit,
        limit_all=limit is None,
        max_length=cfg["max_length"],
        teacher_batch_size=cfg["data"]["batch_size"],
        attention=resolve_attention(cfg["attention"]),
        push_to_hub=push_to_hub,
        no_stream=no_stream,
        rebuild_normalized=rebuild_normalized,
        detailed_timings=detailed_timings,
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
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Download full datasets then select, instead of streaming rows",
    )
    parser.add_argument(
        "--rebuild-normalized",
        action="store_true",
        default=False,
        help="Ignore the normalized cache and rebuild it before embedding",
    )
    parser.add_argument(
        "--detailed-timings",
        action="store_true",
        default=False,
        help="Print detailed timing logs for normalization and teacher embedding",
    )
    args = parser.parse_args()

    configure_logging()
    prepare_data(
        limit=args.limit,
        push_to_hub=args.push_to_hub,
        no_stream=args.no_stream,
        rebuild_normalized=args.rebuild_normalized,
        detailed_timings=args.detailed_timings,
    )


if __name__ == "__main__":
    prepare_data_cli()
