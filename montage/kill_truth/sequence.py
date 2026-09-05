"""Build KillSequence records from OwnKillEvent facts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .models import KillSequence, OwnKillEvent, sequence_type_for_count


def build_sequences(
    events: Iterable[OwnKillEvent],
    *,
    source_id: str,
    source_path: Path,
    sequence_gap_s: float = 6.0,
) -> list[KillSequence]:
    """Group chronologically ordered events, using panel identity when present.

    Events produced by the panel state machine normally already carry the
    same sequence id.  The gap is deliberately only a fallback for events
    imported from a dense-refinement pass that lacks a panel lifecycle record.
    """
    materialized = sorted(events, key=lambda item: item.confirmation_time)
    if not materialized:
        return []
    groups: list[list[OwnKillEvent]] = []
    for event in materialized:
        if not groups:
            groups.append([event])
            continue
        previous = groups[-1][-1]
        same_sequence = bool(event.sequence_id and previous.sequence_id and event.sequence_id == previous.sequence_id)
        close_in_time = event.confirmation_time - previous.confirmation_time <= sequence_gap_s
        if same_sequence or (not event.sequence_id and not previous.sequence_id and close_in_time):
            groups[-1].append(event)
        else:
            groups.append([event])
    sequences: list[KillSequence] = []
    for index, group in enumerate(groups, start=1):
        sequence_id = group[0].sequence_id or f"{source_id}-seq-derived-{index:03d}"
        impacts = [event.impact_time for event in group if event.impact_time is not None]
        kill_count = sum(event.kill_count_delta for event in group)
        sequences.append(
            KillSequence(
                sequence_id=sequence_id,
                source_id=source_id,
                source_path=Path(source_path),
                first_confirmation=group[0].confirmation_time,
                last_confirmation=group[-1].confirmation_time,
                first_impact=min(impacts) if impacts else None,
                last_impact=max(impacts) if impacts else None,
                kill_count=kill_count,
                headshot_count=sum(event.kill_count_delta for event in group if event.kill_type == "HEADSHOT"),
                span_seconds=max(0.0, group[-1].confirmation_time - group[0].confirmation_time),
                impact_span_seconds=(max(impacts) - min(impacts)) if len(impacts) >= 2 else (0.0 if impacts else None),
                sequence_type=sequence_type_for_count(kill_count),
                event_ids=tuple(event.event_id for event in group),
                confidence=min(event.confidence for event in group),
            )
        )
    return sequences
