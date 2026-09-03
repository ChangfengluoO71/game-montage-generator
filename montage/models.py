"""Serializable contracts shared by pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve(strict=False))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return {field.name: _json_value(getattr(self, field.name)) for field in fields(self)}


@dataclass
class MediaRecord(Serializable):
    file_path: Path
    file_name: str
    file_size: int
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int | None
    audio_codec: str | None
    audio_channels: int | None
    audio_sample_rate: int | None
    creation_time: str | None
    category: str
    fingerprint: dict[str, Any]
    probe_error: str | None = None


@dataclass
class VideoAnalysis(Serializable):
    source_file: Path
    sample_rate: float
    times: list[float]
    motion: list[float]
    visual: list[float]
    audio: list[float]
    continuity: list[float]
    activity: list[float]
    brightness: list[float] = field(default_factory=list)
    black_ratio: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    candidate_windows: list[dict[str, float]] = field(default_factory=list)


@dataclass
class MusicAnalysis(Serializable):
    source_file: Path
    duration: float
    tempo: float
    beats: list[float]
    strong_beats: list[float]
    bars: list[float]
    onsets: list[float]
    edit_points: list[dict[str, Any]]
    energy_times: list[float]
    rms: list[float]
    onset_strength: list[float]
    novelty: list[float]
    structure_regions: list[dict[str, Any]]
    confidence: dict[str, float]
    preview_music_in: float | None = None
    preview_music_out: float | None = None
    preview_reason: str = ""


@dataclass
class Candidate(Serializable):
    candidate_id: str
    source_file: Path
    source_start: float
    source_end: float
    duration: float
    human_selection_score: float
    audio_score: float
    motion_score: float
    visual_score: float
    continuity_score: float
    duplicate_group: str | None = None
    final_score: float = 0.0
    combat_intensity: float = 0.0
    uniqueness: float = 1.0
    source_category: str = ""
    fingerprint: list[int] = field(default_factory=list)
    rationale: str = ""

    @property
    def human_selection_prior(self) -> float:
        return self.human_selection_score

    @property
    def audio_activity(self) -> float:
        return self.audio_score

    @property
    def visual_activity(self) -> float:
        return self.visual_score


@dataclass
class EditShot(Serializable):
    source: Path
    source_in: float
    source_out: float
    duration: float
    candidate_score: float
    duplicate_group: str | None
    timeline_in: float
    timeline_out: float
    transition: str
    music_target: float | None
    music_event_type: str | None
    sync_offset: float
    rationale: str
    section: str = ""
    source_duration: float | None = None


@dataclass
class EditDecisionList(Serializable):
    kind: str
    music_source: Path
    music_in: float
    music_out: float
    duration: float
    music_reason: str
    shots: list[EditShot]


@dataclass(frozen=True)
class PayoffEvent(Serializable):
    event_id: str
    type: str
    source_time: float
    confidence: float
    strength: float
    semantic_confidence: float
    evidence: dict[str, float]
    detector_flags: tuple[str, ...] = ()
    merged_peak_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True)
class SourceSegment(Serializable):
    source: Path
    source_in: float
    source_out: float
    duration: float

    def __post_init__(self) -> None:
        if self.source_out < self.source_in or abs(self.duration - (self.source_out - self.source_in)) > 0.001:
            raise ValueError("source segment duration must match source_out - source_in within 1ms")


@dataclass(frozen=True)
class CandidateVariant(Serializable):
    variant_id: str
    parent_candidate_id: str
    source_file: Path
    source_segments: tuple[SourceSegment, ...]
    duration: float
    human_selection_prior: float
    payoff_score: float
    combat_intensity: float
    action_density: float
    continuity: float
    visual_novelty: float
    motion: float
    audio_activity: float
    danger_score: float
    uniqueness: float
    final_score: float
    duplicate_group: str | None
    payoff_events: tuple[PayoffEvent, ...]
    primary_anchor: PayoffEvent | None
    secondary_anchors: tuple[PayoffEvent, ...]
    anchor_event_time: float | None
    anchor_event_type: str | None
    anchor_event_strength: float | None
    anchor_event_confidence: float | None
    context_integrity_score: float
    penalty_values: dict[str, float]
    source_signature: str
    environment_signature: str
    weapon_or_view_signature: str
    condense_reason: str
    rationale: str

    def __post_init__(self) -> None:
        if not self.source_segments:
            raise ValueError("candidate variant requires at least one source segment")
        object.__setattr__(self, "penalty_values", MappingProxyType(dict(self.penalty_values)))


@dataclass(frozen=True)
class V2EditShot(Serializable):
    source: Path
    source_in: float
    source_out: float
    duration: float
    candidate_score: float
    duplicate_group: str | None
    timeline_in: float
    timeline_out: float
    transition: str
    music_target: float | None
    music_event_type: str | None
    sync_offset: float
    rationale: str
    section: str = ""
    source_duration: float | None = None
    source_segments: tuple[SourceSegment, ...] = ()
    parent_candidate_id: str = ""
    variant_id: str = ""
    payoff_events: tuple[PayoffEvent, ...] = ()
    anchor_event_time: float | None = None
    anchor_event_type: str | None = None
    anchor_event_strength: float | None = None
    anchor_event_confidence: float | None = None
    primary_anchor: PayoffEvent | None = None
    secondary_anchors: tuple[PayoffEvent, ...] = ()
    context_integrity_score: float = 1.0
    condense_reason: str = ""
    event_timeline: float | None = None
    event_sync_offset: float = 0.0
    cut_sync_offset: float = 0.0
    transition_compatibility_score: float = 1.0
    impact_cut: bool = False
    audio_j_cut_ms: int = 0
    audio_l_cut_ms: int = 0

    def __post_init__(self) -> None:
        if not self.source_segments:
            raise ValueError("V2 edit shot requires at least one source segment")


@dataclass(frozen=True)
class V2EditDecisionList(Serializable):
    kind: str
    music_source: Path
    baseline_music_in: float
    baseline_music_out: float
    music_in: float
    music_out: float
    duration: float
    music_reason: str
    shots: tuple[V2EditShot, ...]


@dataclass(frozen=True)
class DedupeResult(Serializable):
    groups: list[list[Candidate]]
    similarity: dict[str, float]
