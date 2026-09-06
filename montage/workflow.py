"""Portable, game-agnostic montage workflow contracts.

The workflow stores detector-specific information as data while keeping editing,
audio, and export behavior reusable across games.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from numbers import Real
from pathlib import Path
import subprocess
from typing import Any, Mapping

SCHEMA_VERSION = "game-montage-workflow-v1"
DEFAULT_AUDIO_OUTPUT = {
    "game_gain_db": -6.0,
    "music_gain_db": -14.0,
    "target_lufs": -16.0,
    "true_peak_db": -2.0,
    "sample_rate": 48000,
    "channels": 2,
}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


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
        if any(_finite_number(value, "editing rule durations") < 0 for value in values):
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
        if not isinstance(self.roi, dict):
            raise ValueError("ROI must be an object")
        if self.roi.get("coordinate_space") != "normalized":
            raise ValueError("ROI coordinate_space must be normalized")
        if not isinstance(self.thresholds, dict):
            raise ValueError("detector thresholds must be an object")
        for name, values in (("templates", self.templates), ("positive_samples", self.positive_samples), ("negative_samples", self.negative_samples)):
            if not isinstance(values, list):
                raise ValueError(f"{name} must be an array")
        coords = [_finite_number(self.roi.get(key), f"ROI {key}") for key in ("x1", "y1", "x2", "y2")]
        if not (0 <= coords[0] < coords[2] <= 1 and 0 <= coords[1] < coords[3] <= 1):
            raise ValueError("ROI must be a normalized rectangle")
        if any(not 0 <= _finite_number(value, "detector thresholds") <= 1 for value in self.thresholds.values()):
            raise ValueError("detector thresholds must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"detector_type": self.detector_type, "event_label": self.event_label, "roi": self.roi,
                "templates": self.templates, "positive_samples": self.positive_samples,
                "negative_samples": self.negative_samples, "thresholds": self.thresholds}


@dataclass
class WorkflowRule:
    """One detector rule in the additive multi-rule workflow representation."""

    id: str
    label: str
    detector: DetectorConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("workflow rule id is required")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("workflow rule label is required")
        if not isinstance(self.metadata, dict):
            raise ValueError("workflow rule metadata must be an object")
        self.detector.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"id": self.id, "label": self.label, "detector": self.detector.to_dict(), "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowRule":
        if not isinstance(data, Mapping):
            raise ValueError("workflow rules must be objects")
        detector_data = data.get("detector")
        if detector_data is None:
            detector_data = {
                "detector_type": data.get("detector_type", data.get("type", "template_match")),
                "event_label": data.get("event_label", data.get("label", "HIGHLIGHT")),
                "roi": data.get("roi", {"coordinate_space": "normalized", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}),
                "templates": data.get("templates", []),
                "positive_samples": data.get("positive_samples", []),
                "negative_samples": data.get("negative_samples", []),
                "thresholds": data.get("thresholds", {}),
            }
        if not isinstance(detector_data, Mapping):
            raise ValueError("workflow rule detector must be an object")
        raw_metadata = data.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("workflow rule metadata must be an object")
        detector = DetectorConfig(**dict(detector_data))
        rule = cls(str(data.get("id", "")), str(data.get("label", detector.event_label)), detector, dict(raw_metadata))
        rule.validate()
        return rule


@dataclass
class MontageWorkflow:
    game_id: str
    display_name: str
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    edit_rules: EditRules = field(default_factory=EditRules)
    audio_output: dict[str, Any] = field(default_factory=lambda: {"game_gain_db": -6.0, "music_gain_db": -14.0, "target_lufs": -16.0, "true_peak_db": -2.0, "sample_rate": 48000, "channels": 2})
    profiles: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    rules: list[WorkflowRule] = field(default_factory=list)

    def validate(self) -> None:
        if not self.game_id or not self.display_name:
            raise ValueError("game_id and display_name are required")
        self.detector.validate()
        self.edit_rules.validate()
        if not isinstance(self.audio_output, dict):
            raise ValueError("audio_output must be an object")
        if self.audio_output.get("sample_rate", 48000) not in (44100, 48000):
            raise ValueError("sample_rate must be 44100 or 48000")
        for key, value in self.audio_output.items():
            if key not in {"sample_rate", "channels"}:
                _finite_number(value, f"audio_output.{key}")
        if not isinstance(self.metadata, dict):
            raise ValueError("workflow metadata must be an object")
        seen: set[str] = set()
        for rule in self.rules:
            rule.validate()
            if rule.id in seen:
                raise ValueError(f"duplicate workflow rule id: {rule.id}")
            seen.add(rule.id)
        if self.rules and self.rules[0].detector.to_dict() != self.detector.to_dict():
            raise ValueError("legacy detector must match the first workflow rule")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = {"schema": SCHEMA_VERSION, "game_id": self.game_id, "display_name": self.display_name,
                  "detector": self.detector.to_dict(), "edit_rules": self.edit_rules.to_dict(),
                  "audio_output": self.audio_output, "profiles": self.profiles, "metadata": self.metadata}
        if self.rules:
            result["rules"] = [rule.to_dict() for rule in self.rules]
        return result

    def to_desktop_dict(self) -> dict[str, Any]:
        """Export the UI-facing schema while retaining every engine rule."""
        self.validate()
        rules = self.rules or [WorkflowRule(self.detector.event_label.lower() or "rule", self.detector.event_label, self.detector)]
        desktop_rules = []
        for rule in rules:
            detector = rule.detector
            roi = detector.roi
            desktop_rule = {
                "id": rule.id,
                "label": rule.label,
                "type": detector.detector_type,
                **({"event_label": detector.event_label} if detector.event_label != rule.label else {}),
                "search_roi": [roi["x1"], roi["y1"], roi["x2"], roi["y2"]],
                "templates": detector.templates,
                "positive_samples": detector.positive_samples,
                "negative_samples": detector.negative_samples,
                "metadata": rule.metadata,
            }
            if "template" in detector.thresholds:
                desktop_rule["threshold"] = detector.thresholds["template"]
            if detector.thresholds and set(detector.thresholds) != {"template"}:
                desktop_rule["thresholds"] = detector.thresholds
            desktop_rules.append(desktop_rule)
        result = {"schema": SCHEMA_VERSION, "game": {"id": self.game_id, "display_name": self.display_name},
                  "detectors": {"rules": desktop_rules}, "edit_rules": self.edit_rules.to_dict(),
                  "audio_output": dict(self.audio_output), "metadata": self.metadata}
        if self.profiles:
            result["profiles"] = self.profiles
        return result

    def export_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def export_desktop_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_desktop_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MontageWorkflow":
        if data.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"unsupported workflow schema: {data.get('schema')}")
        if "game" in data or "detectors" in data:
            return cls.from_desktop_dict(data)
        if "game_id" not in data or "display_name" not in data:
            raise ValueError("workflow requires game_id/display_name or game/display_name")
        raw_detector = data.get("detector", {})
        raw_edit = data.get("edit_rules", {})
        raw_audio = data.get("audio_output", {})
        raw_rules = data.get("rules", [])
        raw_metadata = data.get("metadata", {})
        if not all(isinstance(value, Mapping) for value in (raw_detector, raw_edit, raw_audio, raw_metadata)):
            raise ValueError("workflow detector, edit_rules, audio_output, and metadata must be objects")
        if not isinstance(raw_rules, list):
            raise ValueError("workflow rules must be an array")
        detector = DetectorConfig(**dict(raw_detector))
        rules = EditRules(**dict(raw_edit))
        audio = dict(DEFAULT_AUDIO_OUTPUT); audio.update(dict(raw_audio))
        parsed_rules = [WorkflowRule.from_dict(item) for item in raw_rules]
        if parsed_rules and parsed_rules[0].detector.to_dict() != detector.to_dict():
            raise ValueError("legacy detector must match the first workflow rule")
        workflow = cls(str(data["game_id"]), str(data["display_name"]), detector, rules,
                       audio, list(data.get("profiles", [])), dict(raw_metadata), parsed_rules)
        workflow.validate()
        return workflow

    @classmethod
    def from_desktop_dict(cls, data: Mapping[str, Any]) -> "MontageWorkflow":
        if data.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"unsupported workflow schema: {data.get('schema')}")
        game = data.get("game")
        detectors = data.get("detectors")
        if not isinstance(game, Mapping) or not isinstance(detectors, Mapping):
            raise ValueError("desktop workflow requires game and detectors objects")
        game_id, display_name = game.get("id"), game.get("display_name")
        if not isinstance(game_id, str) or not game_id.strip() or not isinstance(display_name, str) or not display_name.strip():
            raise ValueError("desktop workflow requires game.id and game.display_name")
        raw_rules = detectors.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError("desktop workflow requires at least one detector rule")
        parsed: list[WorkflowRule] = []
        seen_ids: set[str] = set()
        for item in raw_rules:
            if not isinstance(item, Mapping):
                raise ValueError("desktop detector rules must be objects")
            roi_values = item.get("search_roi")
            if not isinstance(roi_values, (list, tuple)) or len(roi_values) != 4:
                raise ValueError("desktop rule search_roi must contain four values")
            raw_thresholds = item.get("thresholds", {})
            if not isinstance(raw_thresholds, Mapping):
                raise ValueError("desktop rule thresholds must be an object")
            threshold_value = item.get("threshold", raw_thresholds.get("template", 0.65))
            threshold = _finite_number(threshold_value, "rule threshold")
            if not 0 <= threshold <= 1:
                raise ValueError("rule threshold must be between 0 and 1")
            raw_metadata = item.get("metadata", {})
            if not isinstance(raw_metadata, Mapping):
                raise ValueError("desktop rule metadata must be an object")
            metadata = dict(raw_metadata)
            for key in ("templates", "positive_samples", "negative_samples"):
                if not isinstance(item.get(key, []), list):
                    raise ValueError(f"desktop rule {key} must be an array")
            detector = DetectorConfig(
                detector_type=str(item.get("type", "template_match")),
                event_label=str(item.get("event_label", item.get("label", "HIGHLIGHT"))),
                roi={"coordinate_space": "normalized", "x1": roi_values[0], "y1": roi_values[1], "x2": roi_values[2], "y2": roi_values[3]},
                templates=list(item.get("templates", [])),
                positive_samples=list(item.get("positive_samples", [])),
                negative_samples=list(item.get("negative_samples", [])),
                thresholds=({str(key): value for key, value in raw_thresholds.items()} | {"template": threshold}) if ("threshold" in item or "template" in raw_thresholds) else {str(key): value for key, value in raw_thresholds.items()},
            )
            base_id = str(item.get("id", "")).strip() or "rule"
            rule_id = base_id
            suffix = 2
            while rule_id in seen_ids:
                rule_id = f"{base_id}-{suffix}"
                suffix += 1
            seen_ids.add(rule_id)
            parsed.append(WorkflowRule(rule_id, str(item.get("label", "")), detector, metadata))
        raw_edit = data.get("edit_rules", {})
        raw_audio = data.get("audio_output", {})
        if not isinstance(raw_edit, Mapping) or not isinstance(raw_audio, Mapping):
            raise ValueError("desktop edit_rules and audio_output must be objects")
        if not isinstance(data.get("profiles", []), list):
            raise ValueError("desktop profiles must be an array")
        edit = EditRules(**dict(raw_edit))
        audio = dict(DEFAULT_AUDIO_OUTPUT); audio.update(dict(raw_audio))
        raw_metadata = data.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("workflow metadata must be an object")
        workflow = cls(str(game_id), str(display_name), parsed[0].detector, edit, audio,
                       list(data.get("profiles", [])), dict(raw_metadata), parsed)
        workflow.validate()
        return workflow

    @classmethod
    def import_json(cls, path: Path) -> "MontageWorkflow":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


DetectorRule = WorkflowRule


def extract_audio(input_media: Path, output_audio: Path, *, ffmpeg: str = "ffmpeg") -> Path:
    """Extract the first audio stream from MP4/MKV/etc without touching input."""
    if not input_media.is_file():
        raise FileNotFoundError(input_media)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(input_media), "-map", "0:a:0", "-vn", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", "-y", str(output_audio)]
    subprocess.run(command, check=True, shell=False)
    return output_audio


def resolve_roi(roi: Mapping[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(roi, Mapping):
        raise ValueError("ROI must be an object")
    if roi.get("coordinate_space") != "normalized":
        raise ValueError("only normalized ROI is supported")
    x1, y1, x2, y2 = (_finite_number(roi.get(k), f"ROI {k}") for k in ("x1", "y1", "x2", "y2"))
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("ROI must be a normalized rectangle")
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
        if not isinstance(self.clip_id, str) or not self.clip_id.strip() or not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("timeline clip ids are required")
        for name, value in (("source_in", self.source_in), ("source_out", self.source_out), ("timeline_in", self.timeline_in), ("timeline_out", self.timeline_out)):
            _finite_number(value, f"timeline clip {name}")
        if not isinstance(self.event_ids, list) or any(not isinstance(event_id, str) or not event_id.strip() for event_id in self.event_ids):
            raise ValueError("timeline clip event_ids must be an array of strings")
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

    def validate(self) -> None:
        if not isinstance(self.project_id, str) or not self.project_id.strip():
            raise ValueError("project_id is required")
        if not isinstance(self.workflow, MontageWorkflow):
            raise ValueError("project workflow is required")
        self.workflow.validate()
        if not isinstance(self.clips, list):
            raise ValueError("clips must be an array")
        if not isinstance(self.source_ledger, list):
            raise ValueError("source_ledger must be an array")
        if self.music_source is not None and not isinstance(self.music_source, str):
            raise ValueError("music_source must be a string or null")
        if not isinstance(self.render_settings, dict):
            raise ValueError("render_settings must be an object")
        durations: dict[str, float] = {}
        source_ids: set[str] = set()
        for source in self.source_ledger:
            if not isinstance(source, dict):
                raise ValueError("source_ledger entries must be objects")
            source_id = source.get("source_id")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError("source_ledger entries require source_id")
            if source_id in durations:
                raise ValueError(f"duplicate source id: {source_id}")
            if source_id in source_ids:
                raise ValueError(f"duplicate source id: {source_id}")
            source_ids.add(source_id)
            if "duration" in source:
                duration = _finite_number(source["duration"], f"source {source_id} duration")
                if duration <= 0:
                    raise ValueError(f"source {source_id} duration must be positive")
                durations[source_id] = duration
        seen_clip_ids: set[str] = set()
        previous_timeline_out: float | None = None
        for clip in self.clips:
            if not isinstance(clip, TimelineClip):
                raise ValueError("clips must contain TimelineClip objects")
            clip.validate()
            if clip.clip_id in seen_clip_ids:
                raise ValueError(f"duplicate clip id: {clip.clip_id}")
            seen_clip_ids.add(clip.clip_id)
            if source_ids:
                if clip.source_id not in source_ids:
                    raise ValueError(f"unknown source: {clip.source_id}")
            if clip.source_id in durations:
                if clip.source_out > durations[clip.source_id]:
                    raise ValueError(f"clip source range exceeds source bounds: {clip.source_id}")
            if previous_timeline_out is None and not math.isclose(clip.timeline_in, 0.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("timeline clips must start at zero")
            if previous_timeline_out is not None and not math.isclose(clip.timeline_in, previous_timeline_out, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("timeline clips must be contiguous")
            previous_timeline_out = clip.timeline_out

    def to_dict(self) -> dict[str, Any]:
        self.validate()
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
        if not isinstance(data, dict):
            raise ValueError("project JSON must be an object")
        if data.get("schema") != "game-montage-project-v1":
            raise ValueError(f"unsupported project schema: {data.get('schema')}")
        if not isinstance(data.get("project_id"), str) or not data["project_id"].strip():
            raise ValueError("project_id is required")
        workflow = MontageWorkflow.from_dict(data["workflow"])
        raw_clips = data.get("clips", [])
        raw_ledger = data.get("source_ledger", [])
        raw_settings = data.get("render_settings", {})
        if not isinstance(raw_clips, list) or not isinstance(raw_ledger, list) or not isinstance(raw_settings, dict):
            raise ValueError("clips, source_ledger, and render_settings must be valid containers")
        clips = [TimelineClip(**clip) for clip in raw_clips]
        project = cls(data["project_id"], workflow, clips, raw_ledger,
                      data.get("music_source"), raw_settings)
        project.validate()
        return project
