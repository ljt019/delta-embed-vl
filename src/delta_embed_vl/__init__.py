import logging
import random

import numpy as np
import torch
from datasets import disable_progress_bars

_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_NOISY_LOGGERS = (
    "datasets",
    "fsspec",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "urllib3",
)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT, force=True)
    disable_progress_bars()
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_attention(attention: str | None) -> str | None:
    if attention == "fa":
        return "flash_attention_2"
    return attention
