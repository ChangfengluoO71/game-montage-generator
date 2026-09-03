"""Serializable contracts shared by pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


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
class DedupeResult(Serializable):
    groups: list[list[Candidate]]
    similarity: dict[str, float]
