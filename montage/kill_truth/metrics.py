"""Gold-set metrics for V6 event truth."""

from __future__ import annotations

from typing import Iterable


def match_event_times(
    detected_times: Iterable[float],
    truth_times: Iterable[float],
    *,
    tolerance: float = 0.5,
) -> dict[str, object]:
    detected = sorted(float(value) for value in detected_times)
    truth = sorted(float(value) for value in truth_times)
    used: set[int] = set()
    matches: list[tuple[float, float]] = []
    false_positives: list[float] = []
    for candidate in detected:
        options = [
            (abs(candidate - expected), index, expected)
            for index, expected in enumerate(truth)
            if index not in used and abs(candidate - expected) <= tolerance
        ]
        if not options:
            false_positives.append(candidate)
            continue
        _, index, expected = min(options)
        used.add(index)
        matches.append((candidate, expected))
    false_negatives = [expected for index, expected in enumerate(truth) if index not in used]
    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
