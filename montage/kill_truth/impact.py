"""Auxiliary impact-time estimation; never a source of kill truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ActivityEvidence:
    timestamp: float
    kind: str
    score: float
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "score": self.score,
            "detail": self.detail,
        }


def estimate_impact_time(
    confirmation_time: float,
    evidence: Iterable[Mapping[str, object] | ActivityEvidence],
    *,
    before_s: float = 0.6,
    after_s: float = 0.1,
    minimum_score: float = 0.75,
) -> float | None:
    """Choose an auxiliary impact timestamp, or return ``None``.

    This function has no path to creating or deleting an OwnKillEvent.  It is
    intentionally conservative because confirmation time is the only V6
    semantic fact available before later audiovisual review.
    """
    candidates: list[tuple[float, float]] = []
    for item in evidence:
        if isinstance(item, ActivityEvidence):
            timestamp = item.timestamp
            score = item.score
        else:
            try:
                timestamp = float(item["timestamp"])
                score = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
        if confirmation_time - before_s <= timestamp <= confirmation_time + after_s and score >= minimum_score:
            candidates.append((abs(timestamp - confirmation_time), timestamp))
    return min(candidates)[1] if candidates else None
