"""Temporal skull-row state machine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import KillSequence, OwnKillEvent, SkullRowState, event_with_sequence, sequence_type_for_count
from .refinement import RefinementRequest, resolve_count_jump


@dataclass(frozen=True)
class StateMachineResult:
    new_events: tuple[OwnKillEvent, ...] = ()
    refinement_requests: tuple[RefinementRequest, ...] = ()
    sequence_ended: bool = False


class TemporalStateMachine:
    def __init__(
        self,
        source_id: str,
        source_path: Path,
        *,
        panel_disappear_s: float = 0.75,
        refinement_radius_s: float = 1.0,
    ) -> None:
        self.source_id = str(source_id)
        self.source_path = Path(source_path)
        self.panel_disappear_s = float(panel_disappear_s)
        self.refinement_radius_s = float(refinement_radius_s)
        self.events: list[OwnKillEvent] = []
        self.sequences: list[KillSequence] = []
        self.refinement_requests: list[RefinementRequest] = []
        self._current_count = 0
        self._last_state: SkullRowState | None = None
        self._last_panel_time: float | None = None
        self._sequence_id: str | None = None
        self._sequence_events: list[OwnKillEvent] = []
        self._sequence_counter = 0

    def _start_sequence(self) -> None:
        if self._sequence_id is not None:
            return
        self._sequence_counter += 1
        self._sequence_id = f"{self.source_id}-seq-{self._sequence_counter:03d}"
        self._sequence_events = []

    def _close_sequence(self) -> bool:
        if self._sequence_id is None or not self._sequence_events:
            self._sequence_id = None
            self._sequence_events = []
            self._current_count = 0
            return False
        events = self._sequence_events
        impacts = [event.impact_time for event in events if event.impact_time is not None]
        first = events[0].confirmation_time
        last = events[-1].confirmation_time
        impact_span = (max(impacts) - min(impacts)) if len(impacts) >= 2 else (0.0 if impacts else None)
        self.sequences.append(
            KillSequence(
                sequence_id=self._sequence_id,
                source_id=self.source_id,
                source_path=self.source_path,
                first_confirmation=first,
                last_confirmation=last,
                first_impact=min(impacts) if impacts else None,
                last_impact=max(impacts) if impacts else None,
                kill_count=sum(event.kill_count_delta for event in events),
                headshot_count=sum(event.kill_count_delta for event in events if event.kill_type == "HEADSHOT"),
                span_seconds=max(0.0, last - first),
                impact_span_seconds=impact_span,
                sequence_type=sequence_type_for_count(sum(event.kill_count_delta for event in events)),
                event_ids=tuple(event.event_id for event in events),
                confidence=min(event.confidence for event in events),
            )
        )
        self._sequence_id = None
        self._sequence_events = []
        self._current_count = 0
        return True

    def _make_event(self, state: SkullRowState, before: int, after: int, *, simultaneous: bool = False) -> OwnKillEvent:
        self._start_sequence()
        index = len(self._sequence_events) + 1
        event = OwnKillEvent(
            event_id=f"{self.source_id}-e-{len(self.events) + 1:05d}",
            source_id=self.source_id,
            source_path=self.source_path,
            sequence_id=self._sequence_id or "",
            type="SIMULTANEOUS_MULTI_KILL" if simultaneous else "OWN_KILL",
            confirmation_time=state.timestamp,
            impact_time=None,
            sequence_index=index,
            skull_count_before=before,
            skull_count_after=after,
            kill_count_delta=after - before,
            kill_type="HEADSHOT" if state.headshot_count > 0 else "UNKNOWN_KILL",
            confidence=state.confidence,
            evidence={
                "panel_present": state.panel_present,
                "normal_skulls": state.normal_count,
                "headshot_skulls": state.headshot_count,
                "row_bbox": list(state.row_bbox) if state.row_bbox else None,
                "skull_detections": [detection.to_dict() for detection in state.detections],
                "skull_row_geometry_score": state.geometry_score,
                "panel_structure_score": state.panel_structure_score,
                "temporal_state": "count_transition",
                "dense_refinement_used": False,
            },
        )
        return event

    def _record_event(self, event: OwnKillEvent) -> None:
        self.events.append(event)
        self._sequence_events.append(event)

    def consume(self, state: SkullRowState) -> StateMachineResult:
        if not state.panel_present or state.skull_count <= 0:
            ended = False
            if self._last_panel_time is not None and state.timestamp - self._last_panel_time >= self.panel_disappear_s:
                ended = self._close_sequence()
            self._last_state = state
            return StateMachineResult(sequence_ended=ended)

        self._start_sequence()
        self._last_panel_time = state.timestamp
        new_events: list[OwnKillEvent] = []
        requests: list[RefinementRequest] = []
        previous_count = self._current_count
        if previous_count == 0:
            if state.skull_count == 1:
                event = self._make_event(state, 0, 1)
                self._record_event(event)
                new_events.append(event)
            else:
                request = RefinementRequest(
                    source_id=self.source_id,
                    source_path=self.source_path,
                    before_count=0,
                    after_count=state.skull_count,
                    coarse_time=state.timestamp,
                    window_start=max(0.0, state.timestamp - self.refinement_radius_s),
                    window_end=state.timestamp + self.refinement_radius_s,
                    reason="initial_multi_count_requires_dense_refinement",
                )
                requests.append(request)
                self.refinement_requests.append(request)
        elif state.skull_count == previous_count:
            pass
        elif state.skull_count > previous_count:
            delta = state.skull_count - previous_count
            if delta == 1:
                event = self._make_event(state, previous_count, state.skull_count)
                self._record_event(event)
                new_events.append(event)
            else:
                request = RefinementRequest(
                    source_id=self.source_id,
                    source_path=self.source_path,
                    before_count=previous_count,
                    after_count=state.skull_count,
                    coarse_time=state.timestamp,
                    window_start=max(0.0, state.timestamp - self.refinement_radius_s),
                    window_end=state.timestamp + self.refinement_radius_s,
                    reason="coarse_count_jump",
                )
                requests.append(request)
                self.refinement_requests.append(request)
        else:
            # A lower count is a noisy/occluded observation of the same panel,
            # not evidence that the sequence ended.  Sequence boundaries are
            # defined by panel disappearance; closing here caused one real
            # panel to fragment into repeated 0->1 events whenever a skull
            # briefly failed to match.
            pass
        if state.skull_count >= previous_count:
            self._current_count = state.skull_count
        self._last_state = state
        return StateMachineResult(tuple(new_events), tuple(requests), False)

    def apply_refinement(self, request: RefinementRequest, states: list[SkullRowState]) -> list[OwnKillEvent]:
        events = resolve_count_jump(request, states)
        assigned_events: list[OwnKillEvent] = []
        for event in events:
            self._start_sequence()
            assigned = event_with_sequence(event, self._sequence_id or "", len(self._sequence_events) + 1)
            self.events.append(assigned)
            self._sequence_events.append(assigned)
            assigned_events.append(assigned)
        if states:
            self._current_count = max(state.skull_count for state in states if state.panel_present) if any(state.panel_present for state in states) else self._current_count
            self._last_panel_time = max(state.timestamp for state in states if state.panel_present) if any(state.panel_present for state in states) else self._last_panel_time
        return assigned_events

    def finish(self) -> list[KillSequence]:
        self._close_sequence()
        return list(self.sequences)
