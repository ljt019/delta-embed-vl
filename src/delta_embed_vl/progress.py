from __future__ import annotations

import logging
import time


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m"
    return f"{minutes:02d}m{secs:02d}s"


class ProgressLogger:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        label: str,
        total: int,
        unit: str,
        every_items: int,
        min_seconds: float = 60.0,
    ) -> None:
        self.logger = logger
        self.label = label
        self.total = total
        self.unit = unit
        self.every_items = max(1, every_items)
        self.min_seconds = min_seconds
        self.started_at = time.monotonic()
        self.last_log_at = self.started_at
        self.last_value = 0
        self.next_items = self.every_items

    def maybe_log(
        self,
        current: int,
        *,
        extra: str | None = None,
        force: bool = False,
    ) -> None:
        current = min(current, self.total)
        now = time.monotonic()
        advanced = current > self.last_value
        timed_out = advanced and (now - self.last_log_at) >= self.min_seconds
        hit_item_threshold = current >= self.next_items
        finished = current == self.total

        if not (force or timed_out or hit_item_threshold or finished):
            return

        elapsed = max(now - self.started_at, 1e-6)
        rate = current / elapsed
        percent = (100.0 * current / self.total) if self.total else 100.0
        eta = None
        if rate > 0 and current < self.total:
            eta = (self.total - current) / rate

        message = (
            f"{self.label}: {current:,} / {self.total:,} {self.unit} "
            f"({percent:.1f}%, {rate:.1f}/s, elapsed {_format_duration(elapsed)}"
        )
        if eta is not None:
            message += f", eta {_format_duration(eta)}"
        if extra:
            message += f", {extra}"
        message += ")"

        self.logger.info(message)
        self.last_log_at = now
        self.last_value = current
        while self.next_items <= current:
            self.next_items += self.every_items
