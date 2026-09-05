"""Qt independent orchestration for the desktop scan stage."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from .cache import atomic_write_json
from .config import is_within, load_config
from .kill_truth.profile import HudProfile, load_profile
from .kill_truth.scanner import V6ScanConfig, scan_source, source_id_for_path
from .media_index import probe_media
from .models import MediaRecord
from .template_detection import TemplateEvent, scan_template_source
from .toolchain import discover_toolchain
from .workflow import DetectorConfig, MontageWorkflow, WorkflowRule

ProgressCallback = Callable[[int, str], None]
LogCallback = Callable[[str], None]
_REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


@dataclass(frozen=True)
class GenerationRequest:
    workflow: MontageWorkflow | Mapping[str, Any]
    source_dir: Path
    work_dir: Path
    output_dir: Path
    music_source: Path | None = None
    source_paths: tuple[Path, ...] | None = None
    config_path: Path = _REPO_CONFIG
    use_cache: bool = True
    workflow_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dir", Path(self.source_dir).resolve(strict=False))
        object.__setattr__(self, "work_dir", Path(self.work_dir).resolve(strict=False))
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve(strict=False))
        object.__setattr__(self, "music_source", Path(self.music_source).resolve(strict=False) if self.music_source else None)
        object.__setattr__(self, "source_paths", tuple(Path(item).resolve(strict=False) for item in self.source_paths) if self.source_paths is not None else None)
        object.__setattr__(self, "config_path", Path(self.config_path).resolve(strict=False))
        object.__setattr__(self, "workflow_path", Path(self.workflow_path).resolve(strict=False) if self.workflow_path else None)

    def workflow_snapshot(self) -> MontageWorkflow:
        source = self.workflow.to_dict() if isinstance(self.workflow, MontageWorkflow) else dict(self.workflow)
        return MontageWorkflow.from_dict(json.loads(json.dumps(source, ensure_ascii=False)))


@dataclass(frozen=True)
class GenerationEvent:
    source_id: str
    event_id: str
    timestamp: float
    rule_id: str
    label: str
    confidence: float
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "event_id": self.event_id,
            "timestamp": float(self.timestamp),
            "rule_id": self.rule_id,
            "label": self.label,
            "confidence": float(self.confidence),
            "evidence": dict(self.evidence),
        }


@dataclass
class GenerationResult:
    run_dir: Path
    events_path: Path
    source_ledger_path: Path
    events: list[GenerationEvent]
    records: list[MediaRecord]
    status: str = "OK"
    diagnostics: list[str] = field(default_factory=list)
    project_path: Path | None = None
    output_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "montage-generation-result-v1",
            "status": self.status,
            "run_dir": str(self.run_dir.resolve(strict=False)),
            "events_path": str(self.events_path.resolve(strict=False)),
            "source_ledger_path": str(self.source_ledger_path.resolve(strict=False)),
            "events": [event.to_dict() for event in self.events],
            "records": [_record_dict(record) for record in self.records],
            "diagnostics": list(self.diagnostics),
            "project_path": str(self.project_path) if self.project_path else None,
            "output_path": str(self.output_path) if self.output_path else None,
        }


class GenerationFailure(RuntimeError):
    def __init__(self, message: str, result: GenerationResult | None = None) -> None:
        super().__init__(message)
        self.result = result


def _record_dict(record: object) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        return dict(record.to_dict())
    return {
        "file_path": str(Path(record.file_path).resolve(strict=False)),
        "file_name": str(record.file_name),
        "file_size": int(record.file_size),
        "duration": float(record.duration),
        "width": int(record.width),
        "height": int(record.height),
        "fps": float(record.fps),
        "codec": str(record.codec),
        "bitrate": record.bitrate,
        "audio_codec": record.audio_codec,
        "audio_channels": record.audio_channels,
        "audio_sample_rate": record.audio_sample_rate,
        "creation_time": record.creation_time,
        "category": str(record.category),
        "fingerprint": dict(record.fingerprint or {}),
        "probe_error": record.probe_error,
    }


def _resolve_path(value: object, bases: tuple[Path, ...]) -> str:
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    for base in bases:
        candidate = (base / path).resolve(strict=False)
        if candidate.exists():
            return str(candidate)
    return str((bases[0] / path).resolve(strict=False))


def _resolve_workflow_paths(workflow: MontageWorkflow, request: GenerationRequest) -> MontageWorkflow:
    bases = tuple(path for path in ((request.workflow_path.parent if request.workflow_path else None), request.config_path.parent, request.source_dir) if path is not None)
    def resolve_detector(detector: DetectorConfig) -> DetectorConfig:
        templates = []
        for item in detector.templates:
            if isinstance(item, Mapping):
                entry = dict(item)
                key = "file" if "file" in entry else "path"
                if key in entry:
                    entry[key] = _resolve_path(entry[key], bases)
                templates.append(entry)
            else:
                templates.append(_resolve_path(item, bases))
        return replace(
            detector,
            templates=templates,
            positive_samples=[_resolve_path(item, bases) for item in detector.positive_samples],
            negative_samples=[_resolve_path(item, bases) for item in detector.negative_samples],
        )
    detector = resolve_detector(workflow.detector)
    rules = [replace(rule, detector=resolve_detector(rule.detector)) for rule in workflow.rules]
    if rules:
        detector = rules[0].detector
    return replace(workflow, detector=detector, rules=rules)


def _validate_destinations(request: GenerationRequest, configured_raw: Path) -> None:
    if not request.source_dir.is_dir():
        raise ValueError(f"selected source directory does not exist: {request.source_dir}")
    for name, destination in (("work", request.work_dir), ("output", request.output_dir)):
        if is_within(destination, configured_raw) or is_within(destination, request.source_dir):
            raise ValueError(f"{name} directory cannot be inside configured RAW or selected source: {destination}")


def _selected_paths(request: GenerationRequest, suffixes: tuple[str, ...]) -> list[Path]:
    if request.source_paths is not None:
        paths = list(request.source_paths)
    else:
        allowed = {suffix.lower() for suffix in suffixes}
        paths = sorted(path for path in request.source_dir.rglob("*") if path.is_file() and path.suffix.lower() in allowed)
    selected: list[Path] = []
    for path in paths:
        resolved = path.resolve(strict=False)
        if not resolved.is_file():
            raise ValueError(f"selected source is missing: {resolved}")
        if not is_within(resolved, request.source_dir):
            raise ValueError(f"selected source must be inside selected source directory: {resolved}")
        selected.append(resolved)
    return selected


def _profile_for_record(workflow: MontageWorkflow, config: object, record: MediaRecord) -> HudProfile:
    candidates: list[str] = []
    for item in workflow.profiles:
        if not isinstance(item, Mapping):
            continue
        if (item.get("width"), item.get("height")) in {(record.width, record.height), (str(record.width), str(record.height))}:
            candidates.append(str(item.get("profile_id", "")))
    if not candidates:
        candidates.append(str(getattr(config, "v6_profile_id", "")))
    profile_paths = [config.v6_profiles_dir / profile_id / "profile.json" for profile_id in candidates if profile_id]
    profile_paths.extend(sorted(config.v6_profiles_dir.glob("*/profile.json")))
    seen_paths: set[Path] = set()
    for path in profile_paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        profile_id = path.parent.name
        if not profile_id:
            continue
        if not path.is_file():
            continue
        profile = load_profile(path)
        if (profile.width, profile.height) != (record.width, record.height):
            continue
        all_paths = profile.all_template_paths
        if not all_paths or any(not path.is_file() or path.stat().st_size <= 0 for path in all_paths):
            raise GenerationFailure(f"calibrated profile template bank is missing or empty: {path}")
        return profile
    raise GenerationFailure(f"no calibrated skull_row profile for source resolution {record.width}x{record.height}")


def _emit(callback: ProgressCallback | None, percent: int, message: str) -> None:
    if callback:
        callback(max(0, min(100, int(percent))), message)


def run_generation(
    request: GenerationRequest,
    progress: ProgressCallback | None = None,
    log: LogCallback | None = None,
) -> GenerationResult:
    workflow = _resolve_workflow_paths(request.workflow_snapshot(), request)
    loaded_config = load_config(request.config_path)
    config = replace(loaded_config, work_dir=request.work_dir, output_dir=request.output_dir)
    _validate_destinations(request, loaded_config.raw_dir)
    selected_paths = _selected_paths(request, config.video_extensions)
    run_dir = request.work_dir / "generation-runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.json"
    ledger_path = run_dir / "source_ledger.json"
    diagnostics_path = run_dir / "diagnostics.json"
    records: list[MediaRecord] = []
    ledger: list[dict[str, Any]] = []
    events: list[GenerationEvent] = []
    diagnostics: list[str] = []
    result: GenerationResult | None = None
    try:
        _emit(progress, 2, "准备扫描素材")
        toolchain = discover_toolchain(config) if selected_paths else None
        explicit_sources = request.source_paths is not None
        for index, path in enumerate(selected_paths):
            _emit(progress, 5 + int(index * 15 / max(1, len(selected_paths))), f"索引素材 {path.name}")
            if log:
                log(f"Indexing selected source: {path}")
            try:
                record = probe_media(path, toolchain)
                records.append(record)
            except Exception as exc:
                message = f"source decode/index failed for {path}: {exc}"
                diagnostics.append(message)
                ledger.append({"source_id": source_id_for_path(path), "source_path": str(path), "duration": 0.0, "width": 0, "height": 0, "fps": 0.0, "has_audio": False, "status": "SKIPPED_INDEX_ERROR", "error": str(exc)})
                if log:
                    log(message)
                if explicit_sources:
                    raise GenerationFailure(message) from exc
        for record in records:
            ledger.append({
                "source_id": source_id_for_path(record.file_path),
                "source_path": str(Path(record.file_path).resolve(strict=False)),
                "duration": float(record.duration),
                "width": int(record.width),
                "height": int(record.height),
                "fps": float(record.fps),
                "has_audio": bool(record.audio_codec),
                "status": "PENDING",
            })
        atomic_write_json(ledger_path, {"schema": "montage-source-ledger-v1", "sources": ledger})
        if not selected_paths:
            diagnostics.append("No video sources were selected")
        rules = workflow.rules or [WorkflowRule("legacy-detector", workflow.detector.event_label, workflow.detector)]
        total_jobs = max(1, len(records) * len(rules))
        completed_jobs = 0
        for source_index, record in enumerate(records):
            source_id = source_id_for_path(record.file_path)
            for rule_index, rule in enumerate(rules):
                base = 20 + int(completed_jobs * 70 / total_jobs)
                scan_span = 70.0 / total_jobs
                _emit(progress, base, f"扫描 {record.file_name} · {rule.label}")
                detector_type = rule.detector.detector_type.lower()
                if detector_type == "skull_row":
                    profile = _profile_for_record(workflow, config, record)
                    scan_result = scan_source(
                        record,
                        profile,
                        toolchain,
                        scan_config=V6ScanConfig(
                            coarse_fps=config.v6_coarse_fps,
                            dense_fps=config.v6_dense_fps,
                            panel_disappear_s=config.v6_panel_disappear_s,
                            refinement_radius_s=config.v6_refinement_radius_s,
                        ),
                        cache_dir=run_dir / "cache" if request.use_cache else None,
                        raw_dir=request.source_dir,
                        use_cache=request.use_cache,
                        progress=lambda value, message, base=base, scan_span=scan_span: _emit(progress, base + int(value * scan_span / 100.0), message),
                    )
                    if scan_result.status == "PARTIAL_ERROR":
                        raise GenerationFailure(f"PARTIAL_ERROR while scanning {record.file_name}: {'; '.join(scan_result.errors)}")
                    for event in scan_result.events:
                        events.append(GenerationEvent(source_id, f"{rule.id}-{event.event_id}", event.confirmation_time, rule.id, rule.label, event.confidence, {
                            "detector": "skull_row",
                            "kill_type": event.kill_type,
                            "positive_samples": list(rule.detector.positive_samples),
                            "negative_samples": list(rule.detector.negative_samples),
                        }))
                elif detector_type == "template_match":
                    detections = scan_template_source(
                        record,
                        rule,
                        toolchain,
                        progress=lambda value, message, base=base, scan_span=scan_span: _emit(progress, base + int(value * scan_span / 100.0), message),
                    )
                    for event in detections:
                        if isinstance(event, TemplateEvent):
                            events.append(GenerationEvent(event.source_id, f"{rule.id}-{event.source_id}-{len(events) + 1:04d}", event.timestamp, rule.id, rule.label, event.confidence, {
                                **event.evidence,
                                "positive_samples": list(rule.detector.positive_samples),
                                "negative_samples": list(rule.detector.negative_samples),
                            }))
                        else:
                            events.append(event)
                else:
                    raise GenerationFailure(f"unsupported detector type: {rule.detector.detector_type}")
                completed_jobs += 1
            for item in ledger:
                if item["source_id"] == source_id:
                    item["status"] = "OK"
        events.sort(key=lambda event: (event.source_id, event.timestamp, event.rule_id))
        if not events:
            diagnostics.append("No detection events found; no footage was fabricated")
        status = "NO_DETECTIONS" if not events else "OK"
        result = GenerationResult(run_dir, events_path, ledger_path, events, records, status, diagnostics)
        atomic_write_json(events_path, {"schema": "montage-events-v1", "status": status, "events": [event.to_dict() for event in events], "diagnostics": diagnostics})
        atomic_write_json(ledger_path, {"schema": "montage-source-ledger-v1", "sources": ledger})
        atomic_write_json(diagnostics_path, {"status": status, "diagnostics": diagnostics})
        _emit(progress, 100, "扫描完成" if events else "扫描完成：未检测到事件")
        return result
    except GenerationFailure as exc:
        diagnostics.append(str(exc))
        if result is None:
            result = GenerationResult(run_dir, events_path, ledger_path, events, records, "FAILED", diagnostics)
        atomic_write_json(events_path, {"schema": "montage-events-v1", "status": "FAILED", "events": [event.to_dict() for event in events], "diagnostics": diagnostics})
        atomic_write_json(ledger_path, {"schema": "montage-source-ledger-v1", "sources": ledger})
        atomic_write_json(diagnostics_path, {"status": "FAILED", "diagnostics": diagnostics})
        raise GenerationFailure(str(exc), result) from exc
    except Exception as exc:
        diagnostics.append(str(exc))
        if result is None:
            result = GenerationResult(run_dir, events_path, ledger_path, events, records, "FAILED", diagnostics)
        atomic_write_json(events_path, {"schema": "montage-events-v1", "status": "FAILED", "events": [event.to_dict() for event in events], "diagnostics": diagnostics})
        atomic_write_json(ledger_path, {"schema": "montage-source-ledger-v1", "sources": ledger})
        atomic_write_json(diagnostics_path, {"status": "FAILED", "diagnostics": diagnostics})
        raise GenerationFailure(f"generation failed: {exc}", result) from exc
