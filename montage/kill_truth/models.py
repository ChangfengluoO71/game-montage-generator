"""Serializable V6 kill-truth data contracts."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve(strict=False))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return {field.name: _json_value(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True)
class DetectedSkull(Serializable):
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    kind: str
    template_score: float
    color_score: float = 0.0
    scale: float = 1.0
    # Separate silhouette evidence prevents square/X-shaped assist glyphs
    # from entering the skull row through grayscale correlation alone.
    shape_score: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectedSkull":
        return cls(
            bbox=tuple(int(value) for value in data["bbox"]),
            center=tuple(float(value) for value in data["center"]),
            kind=str(data.get("kind", "NORMAL")),
            template_score=float(data.get("template_score", 0.0)),
            color_score=float(data.get("color_score", 0.0)),
            scale=float(data.get("scale", 1.0)),
            shape_score=float(data.get("shape_score", 0.0)),
        )


@dataclass(frozen=True)
class SkullRowState(Serializable):
    timestamp: float
    panel_present: bool
    skull_count: int
    normal_count: int
    headshot_count: int
    geometry_score: float
    row_bbox: tuple[int, int, int, int] | None
    confidence: float
    detections: tuple[DetectedSkull, ...] = ()
    panel_structure_score: float = 0.0
    source_id: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SkullRowState":
        bbox = data.get("row_bbox")
        return cls(
            timestamp=float(data["timestamp"]),
            panel_present=bool(data.get("panel_present", False)),
            skull_count=int(data.get("skull_count", 0)),
            normal_count=int(data.get("normal_count", 0)),
            headshot_count=int(data.get("headshot_count", 0)),
            geometry_score=float(data.get("geometry_score", 0.0)),
            row_bbox=tuple(int(value) for value in bbox) if bbox else None,
            confidence=float(data.get("confidence", 0.0)),
            detections=tuple(DetectedSkull.from_dict(item) for item in data.get("detections", [])),
            panel_structure_score=float(data.get("panel_structure_score", 0.0)),
            source_id=str(data.get("source_id", "")),
        )


def sequence_type_for_count(count: int) -> str:
    if count <= 1:
        return "SINGLE_KILL"
    if count == 2:
        return "DOUBLE_KILL"
    if count == 3:
        return "TRIPLE_KILL"
    if count == 4:
        return "QUAD_KILL"
    return "MULTI_KILL_5_PLUS"


@dataclass(frozen=True)
class OwnKillEvent(Serializable):
    event_id: str
    source_id: str
    source_path: Path
    sequence_id: str
    type: str
    confirmation_time: float
    impact_time: float | None
    sequence_index: int
    skull_count_before: int
    skull_count_after: int
    kill_count_delta: int
    kill_type: str
    confidence: float
    evidence: Mapping[str, Any]
    dense_refinement_used: bool = False
    refinement_reason: str = ""

    def __post_init__(self) -> None:
        if self.type not in {"OWN_KILL", "SIMULTANEOUS_MULTI_KILL"}:
            raise ValueError(f"unsupported V6 event type: {self.type}")
        if self.kill_count_delta < 1:
            raise ValueError("kill_count_delta must be positive")
        if self.skull_count_after < self.skull_count_before:
            raise ValueError("skull count cannot decrease for a kill event")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("OwnKillEvent confidence must be within [0, 1]")
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "evidence", dict(self.evidence))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OwnKillEvent":
        return cls(
            event_id=str(data["event_id"]),
            source_id=str(data["source_id"]),
            source_path=Path(data["source_path"]),
            sequence_id=str(data.get("sequence_id", "")),
            type=str(data.get("type", "OWN_KILL")),
            confirmation_time=float(data["confirmation_time"]),
            impact_time=float(data["impact_time"]) if data.get("impact_time") is not None else None,
            sequence_index=int(data.get("sequence_index", 0)),
            skull_count_before=int(data.get("skull_count_before", 0)),
            skull_count_after=int(data.get("skull_count_after", 0)),
            kill_count_delta=int(data.get("kill_count_delta", 1)),
            kill_type=str(data.get("kill_type", "UNKNOWN_KILL")),
            confidence=float(data.get("confidence", 0.0)),
            evidence=dict(data.get("evidence") or {}),
            dense_refinement_used=bool(data.get("dense_refinement_used", False)),
            refinement_reason=str(data.get("refinement_reason", "")),
        )


@dataclass(frozen=True)
class KillSequence(Serializable):
    sequence_id: str
    source_id: str
    source_path: Path
    first_confirmation: float
    last_confirmation: float
    first_impact: float | None
    last_impact: float | None
    kill_count: int
    headshot_count: int
    span_seconds: float
    impact_span_seconds: float | None
    sequence_type: str
    event_ids: tuple[str, ...]
    confidence: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KillSequence":
        return cls(
            sequence_id=str(data["sequence_id"]),
            source_id=str(data["source_id"]),
            source_path=Path(data["source_path"]),
            first_confirmation=float(data["first_confirmation"]),
            last_confirmation=float(data["last_confirmation"]),
            first_impact=float(data["first_impact"]) if data.get("first_impact") is not None else None,
            last_impact=float(data["last_impact"]) if data.get("last_impact") is not None else None,
            kill_count=int(data.get("kill_count", 0)),
            headshot_count=int(data.get("headshot_count", 0)),
            span_seconds=float(data.get("span_seconds", 0.0)),
            impact_span_seconds=(
                float(data["impact_span_seconds"]) if data.get("impact_span_seconds") is not None else None
            ),
            sequence_type=str(data.get("sequence_type", sequence_type_for_count(int(data.get("kill_count", 0))))),
            event_ids=tuple(str(value) for value in data.get("event_ids", [])),
            confidence=float(data.get("confidence", 0.0)),
        )


def event_with_sequence(event: OwnKillEvent, sequence_id: str, sequence_index: int) -> OwnKillEvent:
    """Return an event with the sequence identity assigned by the state machine."""
    return replace(event, sequence_id=sequence_id, sequence_index=sequence_index)
