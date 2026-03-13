from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvalResult:
    eval_type: str
    benchmark: str
    score: float
    metric: str

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)
