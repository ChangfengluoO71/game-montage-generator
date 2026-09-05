"""Dense refinement contracts for coarse skull-count jumps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import OwnKillEvent, SkullRowState


@dataclass(frozen=True)
class RefinementRequest:
    source_id: str
    source_path: Path
    before_count: int
    after_count: int
    coarse_time: float
    window_start: float
    window_end: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_path": str(self.source_path),
            "before_count": self.before_count,
            "after_count": self.after_count,
            "coarse_time": self.coarse_time,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "reason": self.reason,
        }


def _event(
    request: Mapping[str, Any],
    timestamp: float,
    before: int,
    after: int,
    state: SkullRowState | None,
    *,
    simultaneous: bool,
) -> OwnKillEvent:
    delta = after - before
    evidence = {
        "panel_present": bool(state.panel_present) if state else True,
        "normal_skulls": int(state.normal_count) if state else 0,
        "headshot_skulls": int(state.headshot_count) if state else 0,
        "skull_row_geometry_score": float(state.geometry_score) if state else 0.0,
        "panel_structure_score": float(state.panel_structure_score) if state else 0.0,
        "dense_refinement_used": True,
    }
    headshot = bool(state and state.headshot_count > 0)
    return OwnKillEvent(
        event_id=f"{request['source_id']}-dense-{timestamp:.6f}-{after}",
        source_id=str(request["source_id"]),
        source_path=Path(request["source_path"]),
        sequence_id="",
        type="SIMULTANEOUS_MULTI_KILL" if simultaneous else "OWN_KILL",
        confirmation_time=float(timestamp),
        impact_time=None,
        sequence_index=0,
        skull_count_before=before,
        skull_count_after=after,
        kill_count_delta=delta,
        kill_type="HEADSHOT" if headshot else "UNKNOWN_KILL",
        confidence=float(state.confidence if state else 0.0),
        evidence=evidence,
        dense_refinement_used=True,
        refinement_reason=str(request.get("reason", "coarse_count_jump")),
    )


def resolve_count_jump(
    request: Mapping[str, Any] | RefinementRequest,
    dense_states: Iterable[SkullRowState],
) -> list[OwnKillEvent]:
    """Resolve a coarse ``before -> after`` jump without inventing counts.

    A dense sequence is accepted as evidence only when every intermediate
    transition is observed.  Otherwise one simultaneous event preserves the
    known delta and explicitly records the uncertainty.
    """
    if isinstance(request, RefinementRequest):
        data: Mapping[str, Any] = request.to_dict()
    else:
        data = request
    before = int(data["before_count"])
    after = int(data["after_count"])
    states = sorted((state for state in dense_states if state.panel_present), key=lambda item: item.timestamp)
    counts = [before]
    state_by_after: dict[int, SkullRowState] = {}
    for state in states:
        if state.skull_count >= counts[-1]:
            if state.skull_count != counts[-1]:
                counts.append(state.skull_count)
                state_by_after[state.skull_count] = state
    if after < before or not states:
        return []
    # The panel can lose one icon during fade/occlusion after it has already
    # shown the requested count.  The terminal dense frame is therefore not
    # the right acceptance criterion; use the first frame that reaches the
    # coarse target and keep later lower observations out of the truth path.
    target_states = [state for state in states if state.skull_count == after]
    if not target_states:
        return []
    target_timestamp = target_states[0].timestamp
    if counts[-1] == after and all(after_count - before_count == 1 for before_count, after_count in zip(counts, counts[1:])):
        events: list[OwnKillEvent] = []
        current = before
        for count in counts[1:]:
            state = state_by_after.get(count)
            events.append(
                _event(
                    data,
                    state.timestamp if state else float(data.get("coarse_time", 0.0)),
                    current,
                    count,
                    state,
                    simultaneous=False,
                )
            )
            current = count
        return events
    target_state = target_states[0]
    return [_event(data, target_timestamp, before, after, target_state, simultaneous=True)]
