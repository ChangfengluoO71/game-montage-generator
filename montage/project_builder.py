"""Build an editable timeline project from detected event windows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any

from .workflow import EditableProject, MontageWorkflow, TimelineClip


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass
class _Window:
    source_in: float
    source_out: float
    events: list[dict[str, Any]] = field(default_factory=list)


def _validate_event(event: Any, source_durations: dict[str, float], seen_ids: set[str], known_rule_ids: set[str]) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("events must be objects")
    for key in ("event_id", "source_id", "rule_id", "label"):
        if not isinstance(event.get(key), str) or not event[key].strip():
            raise ValueError(f"event {key} is required")
    event_id = event["event_id"]
    if event_id in seen_ids:
        raise ValueError(f"duplicate event id: {event_id}")
    seen_ids.add(event_id)
    source_id = event["source_id"]
    if source_id not in source_durations:
        raise ValueError(f"unknown source: {source_id}")
    if known_rule_ids and event["rule_id"] not in known_rule_ids:
        raise ValueError(f"unknown rule: {event['rule_id']}")
    timestamp = _finite(event.get("timestamp"), f"event {event_id} timestamp")
    duration = source_durations[source_id]
    if not 0 <= timestamp <= duration:
        raise ValueError(f"event {event_id} timestamp is outside source bounds")
    confidence = _finite(event.get("confidence"), f"event {event_id} confidence")
    if not 0 <= confidence <= 1:
        raise ValueError(f"event {event_id} confidence must be between 0 and 1")
    return deepcopy(event)


def _merge_source_windows(windows: list[_Window], duration: float, merge_gap: float, bridge: float) -> list[_Window]:
    merged: list[_Window] = []
    for window in windows:
        if not merged:
            merged.append(window)
            continue
        previous = merged[-1]
        gap = window.source_in - previous.source_out
        if gap <= merge_gap:
            previous.source_out = max(previous.source_out, window.source_out)
            previous.events.extend(window.events)
            continue
        extension = bridge / 2.0
        previous.source_out = min(duration, previous.source_out + extension)
        window.source_in = max(0.0, window.source_in - extension)
        merged.append(window)
    for window in merged:
        if window.source_out <= window.source_in:
            raise ValueError("event window produced a degenerate source range")
    return merged


def build_project(
    workflow: MontageWorkflow,
    events: list[dict[str, Any]],
    source_ledger: list[dict[str, Any]],
    *,
    project_id: str,
    music_source: str | None = None,
    render_settings: dict[str, Any] | None = None,
) -> EditableProject:
    """Create a bounded, source ordered, editable project from detected events."""
    if not isinstance(workflow, MontageWorkflow):
        raise ValueError("workflow must be a MontageWorkflow")
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("project_id is required")
    if not isinstance(events, list) or not events:
        raise ValueError("at least one event is required")
    if not isinstance(source_ledger, list):
        raise ValueError("source_ledger must be an array")
    if not isinstance(render_settings, (dict, type(None))):
        raise ValueError("render_settings must be an object")

    workflow_snapshot = deepcopy(workflow)
    workflow_snapshot.validate()
    ledger_snapshot = deepcopy(source_ledger)
    source_durations: dict[str, float] = {}
    source_order: list[str] = []
    for source in source_ledger:
        if not isinstance(source, dict) or not isinstance(source.get("source_id"), str) or not source["source_id"].strip():
            raise ValueError("source ledger entries require source_id")
        source_id = source["source_id"]
        if source_id in source_durations:
            raise ValueError(f"duplicate source id: {source_id}")
        duration = _finite(source.get("duration"), f"source {source_id} duration")
        if duration <= 0:
            raise ValueError(f"source {source_id} duration must be positive")
        source_durations[source_id] = duration
        source_order.append(source_id)

    workflow_snapshot.edit_rules.validate()
    seen_event_ids: set[str] = set()
    known_rule_ids = {rule.id for rule in workflow_snapshot.rules}
    by_source: dict[str, list[dict[str, Any]]] = {source_id: [] for source_id in source_order}
    for event in events:
        checked = _validate_event(event, source_durations, seen_event_ids, known_rule_ids)
        by_source[checked["source_id"]].append(checked)

    grouped_windows: list[tuple[str, _Window]] = []
    rules = workflow_snapshot.edit_rules
    for source_id in source_order:
        source_events = sorted(by_source[source_id], key=lambda item: (float(item["timestamp"]), item["event_id"]))
        windows = [
            _Window(
                max(0.0, float(event["timestamp"]) - rules.event_pre_seconds),
                min(source_durations[source_id], float(event["timestamp"]) + rules.event_post_seconds),
                [event],
            )
            for event in source_events
        ]
        for window in _merge_source_windows(windows, source_durations[source_id], rules.merge_gap_seconds, rules.long_gap_bridge_seconds):
            grouped_windows.append((source_id, window))

    clips: list[TimelineClip] = []
    timeline_position = 0.0
    for index, (source_id, window) in enumerate(grouped_windows, start=1):
        duration = window.source_out - window.source_in
        if duration <= 0:
            raise ValueError("event window produced a degenerate source range")
        event_ids = [event["event_id"] for event in window.events]
        labels = ", ".join(dict.fromkeys(event["label"] for event in window.events))
        rule_ids = ", ".join(dict.fromkeys(event["rule_id"] for event in window.events))
        timeline_out = timeline_position + duration
        clips.append(
            TimelineClip(
                clip_id=f"clip-{index:04d}",
                source_id=source_id,
                source_in=window.source_in,
                source_out=window.source_out,
                timeline_in=timeline_position,
                timeline_out=timeline_out,
                event_ids=event_ids,
                ai_reason=f"Detected {labels} ({rule_ids})",
                review_status="pending",
            )
        )
        timeline_position = timeline_out

    settings = deepcopy(render_settings) if render_settings is not None else {}
    settings.setdefault("fade_to_black_seconds", workflow_snapshot.edit_rules.fade_to_black_seconds)
    project = EditableProject(project_id, workflow_snapshot, clips, ledger_snapshot, deepcopy(music_source), settings)
    project.to_dict()
    return project
