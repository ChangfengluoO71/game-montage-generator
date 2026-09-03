"""Conservative transition and music synchronization decisions."""

from __future__ import annotations

from typing import Mapping

from .models import Candidate, MusicAnalysis, VideoAnalysis


_PRIORITY = {
    "section_change": 6,
    "phrase": 5,
    "bar": 4,
    "strong_beat": 3,
    "beat": 2,
    "onset": 1,
}


def choose_sync_point(
    shot: Candidate,
    music: MusicAnalysis,
    target: float,
    tolerance: float = 0.20,
) -> tuple[float, str, float]:
    del shot
    points = [point for point in music.edit_points if abs(float(point["timestamp"]) - target) <= max(0.75, tolerance * 4.0)]
    if not points:
        return target, "none", 0.0
    selected = min(
        points,
        key=lambda point: (
            abs(float(point["timestamp"]) - target),
            -_PRIORITY.get(str(point.get("type")), 0),
        ),
    )
    timestamp = float(selected["timestamp"])
    return timestamp, str(selected.get("type", "beat")), timestamp - target


def choose_transition(
    previous: Candidate | None,
    current: Candidate,
    analyses: Mapping[str, VideoAnalysis],
) -> str:
    del current, analyses
    if previous is None:
        return "hard_cut"
    # Direction metadata is intentionally not guessed in V1. A hard cut is safer than a false match.
    return "hard_cut"
