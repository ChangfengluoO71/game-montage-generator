"""Portable, game-agnostic montage workflow contracts.

The workflow stores detector-specific information as data while keeping editing,
audio, and export behavior reusable across games.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

SCHEMA_VERSION = "game-montage-workflow-v1"


@dataclass
class EditRules:
    event_pre_seconds: float = 1.5
    event_post_seconds: float = 0.5
    merge_gap_seconds: float = 2.0
    long_gap_bridge_seconds: float = 2.0
    fade_to_black_seconds: float = 5.0

    def validate(self) -> None:
        values = (self.event_pre_seconds, self.event_post_seconds, self.merge_gap_seconds,
                  self.long_gap_bridge_seconds, self.fade_to_black_seconds)
        if any(value < 0 for value in values):
            raise ValueError("editing rule durations must be non-negative")
        if self.long_gap_bridge_seconds > self.merge_gap_seconds:
            raise ValueError("long gap bridge cannot exceed merge gap")

    def to_dict(self) -> dict[str, float]:
        self.validate()
        return {"event_pre_seconds": self.event_pre_seconds, "event_post_seconds": self.event_post_seconds,
                "merge_gap_seconds": self.merge_gap_seconds, "long_gap_bridge_seconds": self.long_gap_bridge_seconds,
                "fade_to_black_seconds": self.fade_to_black_seconds}


@dataclass
class DetectorConfig:
    detector_type: str = "template_match"
    event_label: str = "HIGHLIGHT"
    roi: dict[str, Any] = field(default_factory=lambda: {"coordinate_space": "normalized", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0})
    templates: list[dict[str, Any]] = field(default_factory=list)
    positive_samples: list[str] = field(default_factory=list)
    negative_samples: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.detector_type or not self.event_label:
            raise ValueError("detector_type and event_label are required")
        if self.roi.get("coordinate_space") != "normalized":
            raise ValueError("ROI coordinate_space must be normalized")
        coords = [self.roi.get(key) for key in ("x1", "y1", "x2", "y2")]
        if any(not isinstance(v, (int, float)) for v in coords) or not (0 <= coords[0] < coords[2] <= 1 and 0 <= coords[1] < coords[3] <= 1):
            raise ValueError("ROI must be a normalized rectangle")
        if any(not 0 <= float(v) <= 1 for v in self.thresholds.values()):
            raise ValueError("detector thresholds must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"detector_type": self.detector_type, "event_label": self.event_label, "roi": self.roi,
                "templates": self.templates, "positive_samples": self.positive_samples,
                "negative_samples": self.negative_samples, "thresholds": self.thresholds}


@dataclass
class MontageWorkflow:
    game_id: str
    display_name: str
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    edit_rules: EditRules = field(default_factory=EditRules)
    audio_output: dict[str, Any] = field(default_factory=lambda: {"game_gain_db": -6.0, "music_gain_db": -14.0, "target_lufs": -16.0, "true_peak_db": -2.0, "sample_rate": 48000, "channels": 2})
    profiles: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.game_id or not self.display_name:
            raise ValueError("game_id and display_name are required")
        self.detector.validate()
        self.edit_rules.validate()
        if self.audio_output.get("sample_rate", 48000) not in (44100, 48000):
            raise ValueError("sample_rate must be 44100 or 48000")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema": SCHEMA_VERSION, "game_id": self.game_id, "display_name": self.display_name,
                "detector": self.detector.to_dict(), "edit_rules": self.edit_rules.to_dict(),
                "audio_output": self.audio_output, "profiles": self.profiles, "metadata": self.metadata}

    def export_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MontageWorkflow":
        if data.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"unsupported workflow schema: {data.get('schema')}")
        detector = DetectorConfig(**dict(data.get("detector", {})))
        rules = EditRules(**dict(data.get("edit_rules", {})))
        workflow = cls(str(data["game_id"]), str(data["display_name"]), detector, rules,
                       dict(data.get("audio_output", {})), list(data.get("profiles", [])), dict(data.get("metadata", {})))
        workflow.validate()
        return workflow

    @classmethod
    def import_json(cls, path: Path) -> "MontageWorkflow":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def extract_audio(input_media: Path, output_audio: Path, *, ffmpeg: str = "ffmpeg") -> Path:
    """Extract the first audio stream from MP4/MKV/etc without touching input."""
    if not input_media.is_file():
        raise FileNotFoundError(input_media)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(input_media), "-map", "0:a:0", "-vn", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", "-y", str(output_audio)]
    subprocess.run(command, check=True, shell=False)
    return output_audio


def resolve_roi(roi: Mapping[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    if roi.get("coordinate_space") != "normalized":
        raise ValueError("only normalized ROI is supported")
    x1, y1, x2, y2 = (float(roi[k]) for k in ("x1", "y1", "x2", "y2"))
    return (round(width * x1), round(height * y1), round(width * x2), round(height * y2))


@dataclass
class TimelineClip:
    clip_id: str
    source_id: str
    source_in: float
    source_out: float
    timeline_in: float = 0.0
    timeline_out: float = 0.0
    event_ids: list[str] = field(default_factory=list)
    ai_reason: str = ""
    review_status: str = "pending"

    def validate(self) -> None:
        if not self.clip_id or not self.source_id:
            raise ValueError("timeline clip ids are required")
        if self.source_in < 0 or self.source_out <= self.source_in:
            raise ValueError("source clip range is invalid")
        if self.timeline_in < 0 or self.timeline_out <= self.timeline_in:
            raise ValueError("timeline clip range is invalid")
        if self.review_status not in {"pending", "keep", "remove", "duplicate", "edited"}:
            raise ValueError("unsupported clip review status")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"clip_id": self.clip_id, "source_id": self.source_id, "source_in": self.source_in,
                "source_out": self.source_out, "timeline_in": self.timeline_in, "timeline_out": self.timeline_out,
                "event_ids": self.event_ids, "ai_reason": self.ai_reason, "review_status": self.review_status}


@dataclass
class EditableProject:
    project_id: str
    workflow: MontageWorkflow
    clips: list[TimelineClip] = field(default_factory=list)
    source_ledger: list[dict[str, Any]] = field(default_factory=list)
    music_source: str | None = None
    render_settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if not self.project_id:
            raise ValueError("project_id is required")
        return {"schema": "game-montage-project-v1", "project_id": self.project_id,
                "workflow": self.workflow.to_dict(), "clips": [clip.to_dict() for clip in self.clips],
                "source_ledger": self.source_ledger, "music_source": self.music_source,
                "render_settings": self.render_settings}

    def export_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def import_json(cls, path: Path) -> "EditableProject":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "game-montage-project-v1":
            raise ValueError(f"unsupported project schema: {data.get('schema')}")
        workflow = MontageWorkflow.from_dict(data["workflow"])
        clips = [TimelineClip(**clip) for clip in data.get("clips", [])]
        project = cls(str(data["project_id"]), workflow, clips, list(data.get("source_ledger", [])),
                      data.get("music_source"), dict(data.get("render_settings", {})))
        for clip in project.clips:
            clip.validate()
        return project

