from __future__ import annotations

from tqdm.auto import tqdm


class ProgressBar:
    def __init__(
        self,
        *,
        label: str,
        total: int,
        unit: str,
        leave: bool = False,
    ) -> None:
        self.total = total
        self.current = 0
        self.bar = tqdm(
            total=total,
            desc=label,
            unit=unit,
            dynamic_ncols=True,
            leave=leave,
        )

    def update(self, current: int, *, extra: str | None = None) -> None:
        current = min(current, self.total)
        delta = current - self.current
        if delta > 0:
            self.bar.update(delta)
            self.current = current
        if extra:
            self.bar.set_postfix_str(extra, refresh=False)

    def close(self, *, extra: str | None = None) -> None:
        if extra:
            self.bar.set_postfix_str(extra, refresh=False)
        if self.current < self.total:
            self.bar.update(self.total - self.current)
            self.current = self.total
        self.bar.close()
