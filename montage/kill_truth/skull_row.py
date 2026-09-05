"""Skull-row grouping and geometry validation."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .models import DetectedSkull, SkullRowState


def _group_by_y(detections: list[DetectedSkull], y_tolerance: float) -> list[list[DetectedSkull]]:
    groups: list[list[DetectedSkull]] = []
    for detection in sorted(detections, key=lambda item: item.center[1]):
        for group in groups:
            center_y = sum(item.center[1] for item in group) / len(group)
            if abs(detection.center[1] - center_y) <= y_tolerance:
                group.append(detection)
                break
        else:
            groups.append([detection])
    return groups


def _geometry_score(group: list[DetectedSkull], preferred_y: float | None) -> float:
    if not group:
        return 0.0
    heights = [max(1.0, float(item.bbox[3])) for item in group]
    widths = [max(1.0, float(item.bbox[2])) for item in group]
    centers_y = [float(item.center[1]) for item in group]
    mean_height = sum(heights) / len(heights)
    y_std = math.sqrt(sum((value - sum(centers_y) / len(centers_y)) ** 2 for value in centers_y) / len(centers_y))
    y_score = max(0.0, 1.0 - y_std / max(1.0, mean_height * 0.75))
    width_mean = sum(widths) / len(widths)
    width_cv = math.sqrt(sum((value - width_mean) ** 2 for value in widths) / len(widths)) / max(1.0, width_mean)
    size_score = max(0.0, 1.0 - min(1.0, width_cv * 2.0))
    ordered = sorted(group, key=lambda item: item.center[0])
    if len(ordered) == 1:
        spacing_score = 0.80
    else:
        gaps = [ordered[index + 1].center[0] - ordered[index].center[0] for index in range(len(ordered) - 1)]
        valid = [0.65 * width_mean <= gap <= 3.4 * width_mean for gap in gaps]
        gap_mean = sum(gaps) / len(gaps)
        gap_cv = math.sqrt(sum((gap - gap_mean) ** 2 for gap in gaps) / len(gaps)) / max(1.0, gap_mean)
        spacing_score = (sum(valid) / len(valid)) * max(0.0, 1.0 - min(1.0, gap_cv * 2.0))
    if preferred_y is None:
        preference_score = 0.75
    else:
        mean_y = sum(centers_y) / len(centers_y)
        preference_score = max(0.0, 1.0 - abs(mean_y - preferred_y) / max(8.0, mean_height * 5.0))
    return max(0.0, min(1.0, 0.38 * y_score + 0.20 * size_score + 0.27 * spacing_score + 0.15 * preference_score))


def build_skull_row(
    timestamp: float,
    detections: Iterable[DetectedSkull],
    *,
    panel_structure_score: float | None = None,
    panel_structure_threshold: float = 0.0,
    geometry_threshold: float = 0.62,
    y_tolerance: float = 12.0,
    preferred_y: float | None = None,
    source_id: str = "",
) -> SkullRowState:
    materialized = list(detections)
    if not materialized:
        return SkullRowState(
            timestamp=float(timestamp),
            panel_present=False,
            skull_count=0,
            normal_count=0,
            headshot_count=0,
            geometry_score=0.0,
            row_bbox=None,
            confidence=0.0,
            detections=(),
            panel_structure_score=0.0,
            source_id=source_id,
        )
    groups = _group_by_y(materialized, y_tolerance)
    scored = [(_geometry_score(group, preferred_y), group) for group in groups]
    score, best = max(scored, key=lambda item: (item[0], len(item[1])))
    best = sorted(best, key=lambda item: item.center[0])
    x1 = min(item.bbox[0] for item in best)
    y1 = min(item.bbox[1] for item in best)
    x2 = max(item.bbox[0] + item.bbox[2] for item in best)
    y2 = max(item.bbox[1] + item.bbox[3] for item in best)
    structure = max(0.0, min(1.0, float(panel_structure_score if panel_structure_score is not None else 0.75)))
    template_score = sum(max(0.0, min(1.0, item.template_score)) for item in best) / len(best)
    confidence = max(0.0, min(1.0, 0.50 * template_score + 0.35 * score + 0.15 * structure))
    valid = score >= geometry_threshold and structure >= panel_structure_threshold
    return SkullRowState(
        timestamp=float(timestamp),
        panel_present=valid,
        skull_count=len(best) if valid else 0,
        normal_count=sum(item.kind.upper() != "HEADSHOT" for item in best) if valid else 0,
        headshot_count=sum(item.kind.upper() == "HEADSHOT" for item in best) if valid else 0,
        geometry_score=score,
        row_bbox=(x1, y1, x2 - x1, y2 - y1),
        confidence=confidence if valid else 0.0,
        detections=tuple(best),
        panel_structure_score=structure,
        source_id=source_id,
    )
