"""Command-line orchestration for the Battlefield montage pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import statistics
import subprocess
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import soundfile as sf

from montage.cache import (
    assert_baseline_unchanged,
    atomic_write_json,
    baseline_manifest,
    cache_key,
    file_fingerprint,
    read_cached_json,
    v2_cache_key,
    write_cached_json,
)
from montage.candidate import generate_candidates, write_candidates
from montage.config import PipelineConfig, ensure_runtime_dirs, load_config
from montage.dedupe import deduplicate_candidates, fingerprint_candidate
from montage.ffmpeg_renderer import probe_output, render_edit
from montage.media_index import build_media_index, write_media_index
from montage.models import (Candidate, CandidateVariant, DedupeResult, EditDecisionList, EditShot,
                            MediaRecord, MusicAnalysis, PayoffEvent, SourceSegment, V2EditDecisionList,
                            V2EditShot, VideoAnalysis)
from montage.music_analysis import analyze_music, analyze_music_v2
from montage.proxy import build_proxy
from montage.ranking import score_candidates, score_variant, verified_kill_events
from montage.review import render_review_assets
from montage.toolchain import Toolchain, discover_toolchain, run_command
from montage.timeline import build_preview_edit, write_edit_list
from montage.video_analysis import write_video_analysis, analyze_video_activity
from montage.audio_analysis import analyze_audio_waveform, extract_analysis_audio
from montage.condense import build_condensed_variants
from montage.dedupe import choose_v2_representative, deduplicate_variants, fingerprint_variant, write_v2_dedupe_summary
from montage.beam_timeline import build_v2_preview_edit, validate_v2_edit
from montage.payoff_detection import detect_payoff_events, ocr_available, write_payoff_events
from montage.curation import event_is_editorially_excluded, variant_is_editorially_excluded
from montage.candidate import write_v2_candidates
from montage.timeline_visualization import render_v2_timeline_plot
from montage.v2_report import build_v2_sync_report, render_v2_markdown_report, write_v2_report
from montage.v2_renderer import _preflight_v2_sources, render_v2_edit
from montage.kill_truth.cli import run_v6_command


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class PipelineState:
    config: PipelineConfig
    toolchain: Toolchain
    logger: logging.Logger
    records: list[MediaRecord]
    music: MusicAnalysis | None = None
    analyses: dict[str, VideoAnalysis] | None = None
    candidates: list[Candidate] | None = None
    dedupe: DedupeResult | None = None
    edit: EditDecisionList | None = None


@dataclass
class V2PipelineState:
    config: PipelineConfig
    toolchain: Toolchain
    baseline_manifest: dict[str, object]
    music: MusicAnalysis
    music_window: Any
    payoff_events: list[PayoffEvent]
    variants: list[CandidateVariant]
    dedupe: Any
    edit: V2EditDecisionList
    sync_report: dict[str, object]
    rejected: dict[str, int]


def _logger(config: PipelineConfig) -> logging.Logger:
    ensure_runtime_dirs(config)
    logger = logging.getLogger(f"battlefield-montage-{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(config.logs_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _gpu_report() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "query": (result.stdout or "").strip(),
        "error": (result.stderr or "").strip() if result.returncode else "",
    }


def write_environment_report(toolchain: Toolchain, config: PipelineConfig, path: Path) -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "raw_dir": str(config.raw_dir),
        "work_dir": str(config.work_dir),
        "output_dir": str(config.output_dir),
        "music_file": str(config.music_file),
        "profile": config.v5_profile if config.v5_profile != "off" else "v2",
        "selected_ffmpeg": str(toolchain.ffmpeg.resolve(strict=False)),
        "selected_ffprobe": str(toolchain.ffprobe.resolve(strict=False)),
        "gpu": _gpu_report(),
        "toolchain": toolchain.to_dict(),
    }
    atomic_write_json(path, payload)


def _format_v2_time(seconds: float) -> str:
    """Format a timeline/source timestamp as stable ``mm:ss.mmm`` text."""
    value = float(seconds)
    sign = "-" if value < 0 else ""
    milliseconds = int(round(abs(value) * 1000.0))
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds_part, milliseconds_part = divmod(remainder, 1000)
    return f"{sign}{minutes:02d}:{seconds_part:02d}.{milliseconds_part:03d}"


def _format_v2_offset(seconds: float) -> str:
    """Format a sync offset as a signed, human-readable millisecond value."""
    return f"{int(round(float(seconds) * 1000.0)):+d}ms"


def _log_v2_toolchain(logger: logging.Logger, toolchain: Toolchain) -> None:
    """Record the exact toolchain used by the V2 analysis run."""
    logger.info(
        "V2 FFmpeg selected path=%s version=%s",
        toolchain.ffmpeg.resolve(strict=False), toolchain.ffmpeg_version,
    )
    logger.info(
        "V2 ffprobe selected path=%s version=%s",
        toolchain.ffprobe.resolve(strict=False), toolchain.ffprobe_version,
    )
    logger.info(
        "V2 NVENC runtime h264_nvenc=%s hevc_nvenc=%s",
        toolchain.nvenc_h264, toolchain.nvenc_hevc,
    )
    logger.info("V2 toolchain selection reason=%s", toolchain.selected_reason)


def _record_from_dict(data: dict[str, Any]) -> MediaRecord:
    return MediaRecord(
        file_path=Path(data["file_path"]),
        file_name=str(data["file_name"]),
        file_size=int(data["file_size"]),
        duration=float(data["duration"]),
        width=int(data["width"]),
        height=int(data["height"]),
        fps=float(data["fps"]),
        codec=str(data["codec"]),
        bitrate=int(data["bitrate"]) if data.get("bitrate") not in (None, "") else None,
        audio_codec=str(data["audio_codec"]) if data.get("audio_codec") else None,
        audio_channels=int(data["audio_channels"]) if data.get("audio_channels") not in (None, "") else None,
        audio_sample_rate=int(data["audio_sample_rate"]) if data.get("audio_sample_rate") not in (None, "") else None,
        creation_time=str(data["creation_time"]) if data.get("creation_time") else None,
        category=str(data["category"]),
        fingerprint=dict(data.get("fingerprint") or {}),
        probe_error=data.get("probe_error"),
    )


def _load_records(path: Path) -> list[MediaRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", data) if isinstance(data, dict) else data
    return [_record_from_dict(record) for record in records]


def _video_from_dict(data: dict[str, Any]) -> VideoAnalysis:
    return VideoAnalysis(
        source_file=Path(data["source_file"]),
        sample_rate=float(data["sample_rate"]),
        times=[float(value) for value in data.get("times", [])],
        motion=[float(value) for value in data.get("motion", [])],
        visual=[float(value) for value in data.get("visual", [])],
        audio=[float(value) for value in data.get("audio", [])],
        continuity=[float(value) for value in data.get("continuity", [])],
        activity=[float(value) for value in data.get("activity", [])],
        brightness=[float(value) for value in data.get("brightness", [])],
        black_ratio=[float(value) for value in data.get("black_ratio", [])],
        entropy=[float(value) for value in data.get("entropy", [])],
        candidate_windows=list(data.get("candidate_windows", [])),
    )


def _candidate_from_dict(data: dict[str, Any]) -> Candidate:
    return Candidate(
        candidate_id=str(data["candidate_id"]),
        source_file=Path(data["source_file"]),
        source_start=float(data["source_start"]),
        source_end=float(data["source_end"]),
        duration=float(data["duration"]),
        human_selection_score=float(data["human_selection_score"]),
        audio_score=float(data["audio_score"]),
        motion_score=float(data["motion_score"]),
        visual_score=float(data["visual_score"]),
        continuity_score=float(data["continuity_score"]),
        duplicate_group=data.get("duplicate_group"),
        final_score=float(data.get("final_score", 0.0)),
        combat_intensity=float(data.get("combat_intensity", 0.0)),
        uniqueness=float(data.get("uniqueness", 1.0)),
        source_category=str(data.get("source_category", "")),
        fingerprint=[int(value) for value in data.get("fingerprint", [])],
        rationale=str(data.get("rationale", "")),
    )


def _edit_from_dict(data: dict[str, Any]) -> EditDecisionList:
    shots = [
        EditShot(
            source=Path(item["source"]),
            source_in=float(item["source_in"]),
            source_out=float(item["source_out"]),
            duration=float(item["duration"]),
            candidate_score=float(item["candidate_score"]),
            duplicate_group=item.get("duplicate_group"),
            timeline_in=float(item["timeline_in"]),
            timeline_out=float(item["timeline_out"]),
            transition=str(item["transition"]),
            music_target=float(item["music_target"]) if item.get("music_target") is not None else None,
            music_event_type=item.get("music_event_type"),
            sync_offset=float(item.get("sync_offset", 0.0)),
            rationale=str(item.get("rationale", "")),
            section=str(item.get("section", "")),
            source_duration=float(item["source_duration"]) if item.get("source_duration") is not None else None,
        )
        for item in data.get("shots", [])
    ]
    return EditDecisionList(
        kind=str(data.get("kind", "preview")),
        music_source=Path(data["music_source"]),
        music_in=float(data["music_in"]),
        music_out=float(data["music_out"]),
        duration=float(data["duration"]),
        music_reason=str(data.get("music_reason", "")),
        shots=shots,
    )


def _event_from_dict(data: dict[str, Any]) -> PayoffEvent:
    return PayoffEvent(str(data["event_id"]), str(data["type"]), float(data["source_time"]),
                       float(data["confidence"]), float(data["strength"]), float(data.get("semantic_confidence", 0.0)),
                       dict(data.get("evidence") or {}), tuple(data.get("detector_flags") or ()), int(data.get("merged_peak_count", 1)))


def _segment_from_dict(data: dict[str, Any]) -> SourceSegment:
    return SourceSegment(Path(data["source"]), float(data["source_in"]), float(data["source_out"]), float(data["duration"]))


def _v2_shot_from_dict(data: dict[str, Any]) -> V2EditShot:
    primary = _event_from_dict(data["primary_anchor"]) if data.get("primary_anchor") else None
    return V2EditShot(
        source=Path(data["source"]), source_in=float(data["source_in"]), source_out=float(data["source_out"]),
        duration=float(data["duration"]), candidate_score=float(data.get("candidate_score", 0.0)),
        duplicate_group=data.get("duplicate_group"), timeline_in=float(data["timeline_in"]),
        timeline_out=float(data["timeline_out"]), transition=str(data.get("transition", "hard_cut")),
        music_target=data.get("music_target"), music_event_type=data.get("music_event_type"),
        sync_offset=float(data.get("sync_offset", 0.0)), rationale=str(data.get("rationale", "")),
        section=str(data.get("section", "")), source_duration=data.get("source_duration"),
        source_segments=tuple(_segment_from_dict(item) for item in data.get("source_segments", [])),
        parent_candidate_id=str(data.get("parent_candidate_id", "")), variant_id=str(data.get("variant_id", "")),
        payoff_events=tuple(_event_from_dict(item) for item in data.get("payoff_events", [])),
        anchor_event_time=data.get("anchor_event_time"), anchor_event_type=data.get("anchor_event_type"),
        anchor_event_strength=data.get("anchor_event_strength"), anchor_event_confidence=data.get("anchor_event_confidence"),
        primary_anchor=primary, secondary_anchors=tuple(_event_from_dict(item) for item in data.get("secondary_anchors", [])),
        context_integrity_score=float(data.get("context_integrity_score", 1.0)), condense_reason=str(data.get("condense_reason", "")),
        event_timeline=data.get("event_timeline"), event_sync_offset=float(data.get("event_sync_offset", 0.0)),
        cut_sync_offset=float(data.get("cut_sync_offset", 0.0)), transition_compatibility_score=float(data.get("transition_compatibility_score", 1.0)),
        impact_cut=bool(data.get("impact_cut", False)), audio_j_cut_ms=int(data.get("audio_j_cut_ms", 0)),
        audio_l_cut_ms=int(data.get("audio_l_cut_ms", 0)), source_signature=str(data.get("source_signature", "")),
        environment_signature=str(data.get("environment_signature", "")), weapon_or_view_signature=str(data.get("weapon_or_view_signature", "")),
    )


def _v2_edit_from_dict(data: dict[str, Any]) -> V2EditDecisionList:
    return V2EditDecisionList(str(data.get("kind", "preview_v2")), Path(data["music_source"]),
                              float(data["baseline_music_in"]), float(data["baseline_music_out"]),
                              float(data["music_in"]), float(data["music_out"]), float(data["duration"]),
                              str(data.get("music_reason", "")), tuple(_v2_shot_from_dict(item) for item in data.get("shots", [])))


def _load_v2_music(config: PipelineConfig) -> MusicAnalysis:
    payload = json.loads(config.music_v2_cache_path.read_text(encoding="utf-8"))
    data = payload.get("data", payload)
    data = data.get("analysis", data)
    return _load_music_payload(data)


def _load_music_payload(data: dict[str, Any]) -> MusicAnalysis:
    return MusicAnalysis(Path(data["source_file"]), float(data["duration"]), float(data["tempo"]),
        [float(v) for v in data.get("beats", [])], [float(v) for v in data.get("strong_beats", [])],
        [float(v) for v in data.get("bars", [])], [float(v) for v in data.get("onsets", [])], list(data.get("edit_points", [])),
        [float(v) for v in data.get("energy_times", [])], [float(v) for v in data.get("rms", [])],
        [float(v) for v in data.get("onset_strength", [])], [float(v) for v in data.get("novelty", [])],
        list(data.get("structure_regions", [])), {str(k): float(v) for k, v in (data.get("confidence") or {}).items()},
        data.get("preview_music_in"), data.get("preview_music_out"), str(data.get("preview_reason", "")))


def _variant_from_dict(data: dict[str, Any], config: PipelineConfig | None = None) -> CandidateVariant:
    primary = _event_from_dict(data["primary_anchor"]) if data.get("primary_anchor") else None
    environment_signature = str(data.get("environment_signature", ""))
    human_selection_prior = float(data.get("human_selection_prior", 0.0))
    if config is not None and config.v5_profile == "quality" and environment_signature == "short_clip":
        human_selection_prior = max(human_selection_prior, float(config.v2_short_clip_prior))
    return CandidateVariant(
        variant_id=str(data["variant_id"]), parent_candidate_id=str(data.get("parent_candidate_id", "")),
        source_file=Path(data["source_file"]), source_segments=tuple(_segment_from_dict(item) for item in data["source_segments"]),
        duration=float(data["duration"]), human_selection_prior=human_selection_prior,
        payoff_score=float(data.get("payoff_score", 0.0)), combat_intensity=float(data.get("combat_intensity", 0.0)),
        action_density=float(data.get("action_density", 0.0)), continuity=float(data.get("continuity", 0.0)),
        visual_novelty=float(data.get("visual_novelty", 0.0)), motion=float(data.get("motion", 0.0)),
        audio_activity=float(data.get("audio_activity", 0.0)), danger_score=float(data.get("danger_score", 0.0)),
        uniqueness=float(data.get("uniqueness", 1.0)), final_score=float(data.get("final_score", 0.0)),
        duplicate_group=data.get("duplicate_group"), payoff_events=tuple(_event_from_dict(item) for item in data.get("payoff_events", [])),
        primary_anchor=primary, secondary_anchors=tuple(_event_from_dict(item) for item in data.get("secondary_anchors", [])),
        anchor_event_time=data.get("anchor_event_time"), anchor_event_type=data.get("anchor_event_type"),
        anchor_event_strength=data.get("anchor_event_strength"), anchor_event_confidence=data.get("anchor_event_confidence"),
        context_integrity_score=float(data.get("context_integrity_score", 1.0)), penalty_values=dict(data.get("penalty_values", {})),
        source_signature=str(data.get("source_signature", "")), environment_signature=environment_signature,
        weapon_or_view_signature=str(data.get("weapon_or_view_signature", "")), condense_reason=str(data.get("condense_reason", "")),
        rationale=str(data.get("rationale", "")), score_components=dict(data.get("score_components", {})),
        rapid_multikill_score=float(data.get("rapid_multikill_score", 0.0)), rapid_multikill_bonus=float(data.get("rapid_multikill_bonus", 0.0)),
    )


def _load_v2_variants(config: PipelineConfig) -> list[CandidateVariant]:
    path = config.highlight_candidates_v2_path
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("variants", payload.get("candidates", payload)) if isinstance(payload, dict) else payload
    return [_variant_from_dict(row, config) for row in rows]


def _filter_cached_variants_by_editorial_exclusions(
    variants: Sequence[CandidateVariant], config: PipelineConfig,
) -> tuple[list[CandidateVariant], int]:
    """Apply current exclusions to cached variants without re-decoding media."""
    kept = [variant for variant in variants if not variant_is_editorially_excluded(variant, config)]
    return kept, len(variants) - len(kept)


def _score_v2_variants(variants: list[CandidateVariant], config: PipelineConfig) -> list[CandidateVariant]:
    """Apply the auditable V1.1 score before V2 dedupe and beam selection."""
    return [score_variant(variant, config) for variant in variants]


def _load_cached_payoff_events(config: PipelineConfig) -> list[PayoffEvent]:
    if not config.payoff_events_v2_path.exists():
        return []
    payload = json.loads(config.payoff_events_v2_path.read_text(encoding="utf-8"))
    rows = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise RuntimeError("cached payoff_events_v2.json has invalid events")
    if isinstance(payload, dict) and payload.get("event_count") not in (None, len(rows)):
        raise RuntimeError("cached payoff_events_v2.json has an inconsistent event count")
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("cached payoff_events_v2.json contains an invalid event")
    return [_event_from_dict(row) for row in rows]


def _v2_cache_inputs(config: PipelineConfig) -> list[dict[str, object]]:
    """Fingerprint the V1 analysis inputs which feed aggregate V2 artifacts."""
    paths = [config.analysis_dir / "highlight_candidates.json", config.analysis_dir / "media_index.json"]
    paths.extend(sorted(config.analysis_dir.glob("video_*.json")))
    paths.extend(sorted(config.analysis_dir.glob("audio_*.json")))
    inputs = [file_fingerprint(path) for path in paths if path.is_file()]
    candidate_path = config.analysis_dir / "highlight_candidates.json"
    if candidate_path.is_file():
        try:
            rows = json.loads(candidate_path.read_text(encoding="utf-8")).get("candidates", [])
            source_paths = {Path(row["source_file"]) for row in rows if isinstance(row, dict) and row.get("source_file")}
            inputs.extend(file_fingerprint(path) for path in sorted(source_paths) if path.is_file())
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
    if config.music_v2_cache_path.is_file():
        inputs.append(file_fingerprint(config.music_v2_cache_path))
    return sorted(inputs, key=lambda item: str(item.get("absolute_path", "")))


def _v2_artifact_cache_key(config: PipelineConfig, toolchain: Toolchain) -> str:
    return v2_cache_key(
        {"inputs": _v2_cache_inputs(config)},
        "v2_variants_and_events",
        {
            "stage_version": "v2-aggregate-artifacts-v5-quality-5" if config.v5_profile == "quality" else "v2-aggregate-artifacts-6",
            "ffmpeg_version": toolchain.ffmpeg_version,
            "detector_version": config.payoff_detector_version,
            "payoff_analysis_fps": config.payoff_analysis_fps,
            "event_merge_window_ms": config.event_merge_window_ms,
            "strong_anchor_threshold": config.strong_anchor_threshold,
            "weak_anchor_threshold": config.weak_anchor_threshold,
            "payoff_evidence_threshold": config.payoff_evidence_threshold,
            "long_clip_threshold": config.long_clip_threshold,
            "v2_excluded_ranges": [list(item) for item in config.v2_excluded_ranges],
            "v2_short_clip_max_duration": config.v2_short_clip_max_duration,
            "v2_short_clip_prior": config.v2_short_clip_prior,
            "v5_profile": config.v5_profile,
            "v5_min_kill_semantic_confidence": config.v5_min_kill_semantic_confidence,
            "v5_min_killfeed_evidence": config.v5_min_killfeed_evidence,
            "v5_min_reward_evidence": config.v5_min_reward_evidence,
            "v5_min_verified_kills_per_shot": config.v5_min_verified_kills_per_shot,
            "v5_kill_density_window_s": config.v5_kill_density_window_s,
            "v5_kill_density_target": config.v5_kill_density_target,
            "v5_kill_count_weight": config.v5_kill_count_weight,
            "v5_kill_density_weight": config.v5_kill_density_weight,
            "v5_max_sequences_per_candidate": config.v5_max_sequences_per_candidate,
            "v5_sequence_context_s": config.v5_sequence_context_s,
            "v5_min_rapid_context_tail_s": config.v5_min_rapid_context_tail_s,
            "roi_profile": {
                str(name): [float(value) for value in bounds]
                if isinstance(bounds, (list, tuple)) else bounds
                for name, bounds in sorted(config.roi_profile.items(), key=lambda item: str(item[0]))
            },
        },
    )


def _v2_artifact_cache_hit(config: PipelineConfig, expected_key: str) -> bool:
    """Accept cached variants only when their validated event artifact shares identity."""
    try:
        candidates = json.loads(config.highlight_candidates_v2_path.read_text(encoding="utf-8"))
        events = json.loads(config.payoff_events_v2_path.read_text(encoding="utf-8"))
        if not isinstance(candidates, dict) or candidates.get("cache_key") != expected_key:
            return False
        if not isinstance(events, dict) or events.get("cache_key") != expected_key:
            return False
        rows = candidates.get("candidates")
        event_rows = events.get("events")
        if not isinstance(rows, list) or not rows or not isinstance(event_rows, list):
            return False
        required_components = set(config.v2_weights) | {"rapid_multikill_score", "rapid_multikill_bonus"}
        if config.v5_profile == "quality":
            required_components |= {"verified_kill_count", "verified_kill_score", "rapid_kill_count", "kill_density"}
        for row in rows:
            if not isinstance(row, dict):
                return False
            components = row.get("score_components")
            if not isinstance(components, dict) or not required_components.issubset(components):
                return False
            try:
                values = [
                    float(row["final_score"]),
                    float(row["rapid_multikill_score"]),
                    float(row["rapid_multikill_bonus"]),
                    *(float(components[name]) for name in required_components),
                ]
            except (KeyError, TypeError, ValueError):
                return False
            if not all(math.isfinite(value) for value in values):
                return False
        if not config.dedupe_summary_v2_path.exists() or not isinstance(
            json.loads(config.dedupe_summary_v2_path.read_text(encoding="utf-8")), dict
        ):
            return False
        if events.get("event_count") not in (None, len(event_rows)):
            return False
        _load_cached_payoff_events(config)
        return True
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


def _cached_audio_evidence(config: PipelineConfig, candidate: Candidate, records: list[MediaRecord],
                           toolchain: Toolchain) -> dict[str, Any]:
    """Load the V1 per-source audio cache, computing it only when absent."""
    record = next((item for item in records if item.file_path.resolve(strict=False) == candidate.source_file.resolve(strict=False)), None)
    source = record.file_path if record is not None else candidate.source_file
    if record is not None and record.duration > config.long_clip_threshold:
        proxies = sorted(config.proxy_dir.glob(f"*_{record.file_path.stem}.mp4"))
        if proxies:
            source = proxies[0]
    parameters = {
        "analysis_fps": config.analysis_fps,
        "sample_rate": config.analysis_sample_rate,
        "source_for_analysis": str(source),
        "ffmpeg_version": toolchain.ffmpeg_version,
    }
    fingerprint = record.fingerprint if record is not None else file_fingerprint(source)
    key = cache_key(fingerprint, "video_analysis", parameters)
    audio_cache = config.analysis_dir / f"audio_{key[:20]}.json"
    cached = read_cached_json(audio_cache, key)
    if isinstance(cached, dict) and cached.get("times") is not None:
        return cached
    wav_path = config.cache_dir / f"{key[:20]}_payoff_audio.wav"
    extract_analysis_audio(source, wav_path, toolchain, config.analysis_sample_rate)
    samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    evidence = analyze_audio_waveform(samples, sample_rate)
    write_cached_json(audio_cache, key, evidence)
    return evidence


def _write_v2_edit(edit: V2EditDecisionList, config: PipelineConfig) -> None:
    atomic_write_json(config.preview_v2_edit_path, edit.to_dict())
    config.preview_v2_timeline_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, shot in enumerate(edit.shots):
        music_target = "none" if shot.music_target is None else _format_v2_time(shot.music_target)
        rationale = shot.rationale.strip() or "No placement rationale recorded."
        lines.extend([
            f"Shot {i + 1:02d}",
            f"Timeline: {_format_v2_time(shot.timeline_in)} - {_format_v2_time(shot.timeline_out)}",
            f"Source: {shot.source}",
            f"Source range: {_format_v2_time(shot.source_in)} - {_format_v2_time(shot.source_out)}",
            f"Score: {shot.candidate_score:.3f}",
            f"Verified kills: {len(verified_kill_events(shot.payoff_events, config))}",
            f"Duplicate group: {shot.duplicate_group or 'none'}",
            f"Music: target {music_target}; event {shot.music_event_type or 'none'}",
            f"Sync: event {_format_v2_offset(shot.event_sync_offset)}; cut {_format_v2_offset(shot.cut_sync_offset)}",
            f"Transition: {shot.transition}",
            f"Reason: {rationale}",
            "",
        ])
    config.preview_v2_timeline_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v2_analysis_pipeline(config: PipelineConfig) -> V2PipelineState:
    """Run V2 analysis and artifact generation only; never calls a renderer."""
    ensure_runtime_dirs(config)
    logger = _logger(config)
    raw_before = _raw_manifest(config)
    atomic_write_json(config.raw_manifest_v2_before_path, raw_before)
    toolchain = discover_toolchain(config)
    write_environment_report(toolchain, config, config.environment_v2_path)
    _log_v2_toolchain(logger, toolchain)
    logger.info("V2 thresholds strong=%.3f weak=%.3f merge=%dms beam_width=%d", config.strong_anchor_threshold,
                config.weak_anchor_threshold, config.event_merge_window_ms, config.beam_width)
    logger.info("V2 OCR available=%s encoder=%s output=%s stop_rule=preview-only", ocr_available(),
                config.nvenc.get("video_encoder", "h264_nvenc"), config.v2_output_path)
    if not config.baseline_output_path.exists():
        raise FileNotFoundError(f"immutable V1 baseline is missing: {config.baseline_output_path}")
    manifest = baseline_manifest(config.baseline_output_path)
    atomic_write_json(config.baseline_manifest_v2_path, manifest)
    baseline, _ = _load_edit_and_music(config)
    music, window = analyze_music_v2(config, toolchain, baseline)
    logger.info("V2 music range %.3f-%.3f (baseline %.3f-%.3f)", window.v2_music_in, window.v2_music_out,
                window.baseline_music_in, window.baseline_music_out)
    aggregate_key = _v2_artifact_cache_key(config, toolchain)
    variants = _load_v2_variants(config)
    logger.info("V2 cache %s; candidate cache %s", "hit" if config.music_v2_cache_path.exists() else "miss",
                "hit" if config.highlight_candidates_v2_path.exists() else "miss")
    events: list[PayoffEvent] = []
    rejected: dict[str, int] = {}
    cache_hit = bool(variants) and _v2_artifact_cache_hit(config, aggregate_key)
    cached_excluded_count = 0
    if cache_hit:
        variants, cached_excluded_count = _filter_cached_variants_by_editorial_exclusions(variants, config)
        if cached_excluded_count:
            logger.info("V2 cache filtered %d variants by current editorial exclusions", cached_excluded_count)
    if not cache_hit:
        variants = []
        candidates_path = config.analysis_dir / "highlight_candidates.json"
        if candidates_path.exists():
            rows = json.loads(candidates_path.read_text(encoding="utf-8")).get("candidates", [])
            candidates = [_candidate_from_dict(row) for row in rows]
            analyses = {}
            for analysis_path in config.analysis_dir.glob("video_*.json"):
                try:
                    analysis = _video_from_dict(json.loads(analysis_path.read_text(encoding="utf-8")))
                    analyses[str(analysis.source_file)] = analysis
                except (KeyError, TypeError, ValueError):
                    continue
            for candidate in candidates:
                analysis = analyses.get(str(candidate.source_file))
                if analysis is None:
                    continue
                records = _load_records(config.analysis_dir / "media_index.json") if (config.analysis_dir / "media_index.json").exists() else []
                audio_evidence = _cached_audio_evidence(config, candidate, records, toolchain)
                found = detect_payoff_events(candidate.source_file, candidate.source_start, candidate.source_end,
                                             analysis, audio_evidence, config, toolchain)
                excluded_events = [
                    event for event in found
                    if event_is_editorially_excluded(event, candidate.source_file, config)
                ]
                if excluded_events:
                    rejected["editorial_exclusion"] = rejected.get("editorial_exclusion", 0) + len(excluded_events)
                found = [event for event in found if event not in excluded_events]
                events.extend(found)
                generated = build_condensed_variants(candidate, found, music, config)
                excluded_variants = [variant for variant in generated if variant_is_editorially_excluded(variant, config)]
                if excluded_variants:
                    rejected["editorial_exclusion"] = rejected.get("editorial_exclusion", 0) + len(excluded_variants)
                variants.extend(_score_v2_variants(
                    [variant for variant in generated if variant not in excluded_variants], config
                ))
            if variants:
                fingerprints = {variant.variant_id: fingerprint_variant(variant, variant.source_file, toolchain, config)
                                for variant in variants}
                dedupe = deduplicate_variants(
                    variants,
                    fingerprints,
                    config.dedupe_threshold,
                    strict_source_overlap=config.v5_profile == "quality",
                )
                variants = [choose_v2_representative(group, "fast") for group in dedupe.groups]
                write_v2_candidates(variants, config.highlight_candidates_v2_path,
                                    config.highlight_candidates_v2_csv_path, config=config)
                write_v2_dedupe_summary(dedupe, config.dedupe_summary_v2_path)
                candidate_payload = json.loads(config.highlight_candidates_v2_path.read_text(encoding="utf-8"))
                candidate_payload["cache_key"] = aggregate_key
                candidate_payload["editorial_exclusion_count"] = rejected.get("editorial_exclusion", 0)
                candidate_payload["editorial_exclusions"] = [
                    {"source_name": source_name, "start": start, "end": end, "reason": reason}
                    for source_name, start, end, reason in config.v2_excluded_ranges
                ]
                atomic_write_json(config.highlight_candidates_v2_path, candidate_payload)
            else:
                dedupe = None
        else:
            dedupe = None
    else:
        events = _load_cached_payoff_events(config)
        dedupe = None
        if cached_excluded_count:
            rejected["editorial_exclusion"] = cached_excluded_count
        try:
            cached_candidates = json.loads(config.highlight_candidates_v2_path.read_text(encoding="utf-8"))
            cached_exclusions = int(cached_candidates.get("editorial_exclusion_count", 0))
            if cached_exclusions >= 0:
                rejected["editorial_exclusion"] = rejected.get("editorial_exclusion", 0) + cached_exclusions
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        if config.dedupe_summary_v2_path.exists():
            cached_dedupe = json.loads(config.dedupe_summary_v2_path.read_text(encoding="utf-8"))
            if not isinstance(cached_dedupe, dict):
                raise RuntimeError("cached dedupe_summary_v2.json is invalid")
            dedupe = cached_dedupe
    write_payoff_events(events, config.payoff_events_v2_path, config)
    event_payload = json.loads(config.payoff_events_v2_path.read_text(encoding="utf-8"))
    event_payload["cache_key"] = aggregate_key
    atomic_write_json(config.payoff_events_v2_path, event_payload)
    if not variants:
        raise RuntimeError("V2 candidate variants are unavailable; run V1 analysis artifacts first")
    edit = build_v2_preview_edit(variants, music, baseline, config)
    validate_v2_edit(edit, config)
    _write_v2_edit(edit, config)
    report = build_v2_sync_report(
        edit, baseline, variants, rejected,
        lookback_window=config.recent_source_window,
        editorial_exclusions=config.v2_excluded_ranges,
        config=config,
    )
    write_v2_report(report, config.preview_v2_sync_report_path)
    render_v2_timeline_plot(edit, music, config.preview_v2_timeline_image_path)
    raw_after = _raw_manifest(config)
    if raw_before != raw_after:
        raise RuntimeError("RAW manifest changed during V2 analysis")
    assert_baseline_unchanged(manifest, config.baseline_output_path)
    atomic_write_json(config.raw_manifest_v2_after_path, raw_after)
    logger.info("V2 analysis complete; render stop rule active; output=%s", config.v2_output_path)
    return V2PipelineState(config, toolchain, manifest, music, window, events, variants, dedupe, edit, report, rejected)


def render_preview_v2_stage(config: PipelineConfig, state: V2PipelineState | None = None) -> Path:
    required = (config.preview_v2_edit_path, config.preview_v2_sync_report_path, config.preview_v2_timeline_image_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("V2 render gate failed; missing: " + ", ".join(missing))
    toolchain = state.toolchain if state is not None else discover_toolchain(config)
    before_path = config.baseline_manifest_v2_path
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else baseline_manifest(config.baseline_output_path)
    edit = _v2_edit_from_dict(json.loads(config.preview_v2_edit_path.read_text(encoding="utf-8")))
    output = config.v2_output_path
    rendered = render_v2_edit(edit, config, toolchain, output)
    assert_baseline_unchanged(before, config.baseline_output_path)
    render_v2_markdown_report(json.loads(config.preview_v2_sync_report_path.read_text(encoding="utf-8")),
                              config.preview_v2_report_path, toolchain=toolchain, output=rendered)
    return rendered


def verify_preview_v2(config: PipelineConfig) -> int:
    required = (config.preview_v2_edit_path, config.preview_v2_sync_report_path, config.preview_v2_timeline_image_path,
                config.v2_output_path)
    if any(not path.exists() for path in required):
        return 1
    try:
        toolchain = discover_toolchain(config)
        edit = _v2_edit_from_dict(json.loads(config.preview_v2_edit_path.read_text(encoding="utf-8")))
        validate_v2_edit(edit, config)
        _preflight_v2_sources(edit, config, toolchain)
        probe = probe_output(config.v2_output_path, toolchain)
        decode = run_command([toolchain.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", config.v2_output_path,
                              "-map", "0", "-f", "null", "NUL"], check=False)
        baseline = json.loads(config.baseline_manifest_v2_path.read_text(encoding="utf-8"))
        assert_baseline_unchanged(baseline, config.baseline_output_path)
        raw_before_path = config.raw_manifest_v2_before_path
        raw_integrity = True
        if raw_before_path.exists():
            raw_integrity = _raw_manifest(config) == json.loads(raw_before_path.read_text(encoding="utf-8"))
        full_outputs = [
            config.output_dir / name for name in (
                "fast_montage.mp4", "full_highlights.mp4",
                "Battlefield_Fast_Montage.mp4", "Battlefield_Full_Highlights.mp4",
            )
        ]
        shot_groups = [shot.duplicate_group for shot in edit.shots if shot.duplicate_group]
        ranges_valid = all(shot.source_in >= 0 and shot.source_out > shot.source_in and shot.duration > 0 for shot in edit.shots)
        integrity_valid = all(shot.context_integrity_score >= config.minimum_context_integrity for shot in edit.shots)
        dedupe_valid = len(shot_groups) == len(set(shot_groups))
        cadence_valid = abs(float(probe.get("fps", 0.0)) - config.target_fps) <= max(0.5, config.target_fps * 0.02)
        valid = bool(probe.get("has_video") and probe.get("has_audio") and int(probe.get("width", 0)) == config.output_width
                     and int(probe.get("height", 0)) == config.output_height
                     and config.preview_min_duration <= float(probe.get("duration", 0.0)) <= config.preview_max_duration
                     and cadence_valid and ranges_valid and integrity_valid and dedupe_valid and raw_integrity
                     and decode.returncode == 0 and not any(path.exists() for path in full_outputs))
        print(json.dumps({"valid": valid, "probe": probe, "decode_returncode": decode.returncode,
                          "checks": {"cadence": cadence_valid, "ranges": ranges_valid,
                                     "dedupe": dedupe_valid, "integrity": integrity_valid,
                                     "raw_integrity": raw_integrity,
                                     "full_outputs_absent": not any(path.exists() for path in full_outputs)}},
                         ensure_ascii=False, indent=2))
        return 0 if valid else 1
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1


def _p95(values: list[float]) -> float:
    """Return the nearest-rank 95th percentile, including the upper tail."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(len(ordered) * 0.95))
    return ordered[min(len(ordered), rank) - 1]


def _transition_counts(shots: list[EditShot]) -> dict[str, int]:
    """Count transitions between shots; the opening shot has no transition."""
    counts: dict[str, int] = {}
    for shot in shots[1:]:
        counts[shot.transition] = counts.get(shot.transition, 0) + 1
    return counts


def _raw_manifest(config: PipelineConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(config.raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in config.video_extensions:
            fingerprint = file_fingerprint(path)
            rows.append({"path": str(path.resolve()), "size": fingerprint["size"], "mtime": fingerprint["mtime"]})
    return rows


def _new_state(config: PipelineConfig) -> PipelineState:
    logger = _logger(config)
    logger.info("Environment audit")
    toolchain = discover_toolchain(config)
    write_environment_report(toolchain, config, config.analysis_dir / "environment.json")
    logger.info("FFmpeg selected: %s (%s)", toolchain.ffmpeg, toolchain.ffmpeg_version)
    logger.info("ffprobe selected: %s (%s)", toolchain.ffprobe, toolchain.ffprobe_version)
    logger.info("NVENC h264=%s hevc=%s", toolchain.nvenc_h264, toolchain.nvenc_hevc)
    return PipelineState(config=config, toolchain=toolchain, logger=logger, records=[])


def _run_index_stage(state: PipelineState) -> None:
    state.logger.info("INDEX start")
    state.records = build_media_index(state.config, state.toolchain, state.logger)
    write_media_index(
        state.records,
        state.config.analysis_dir / "media_index.json",
        state.config.analysis_dir / "media_index.csv",
    )
    state.logger.info("INDEX complete: %d records", len(state.records))


def _load_or_index(state: PipelineState) -> None:
    index_path = state.config.analysis_dir / "media_index.json"
    if index_path.exists():
        state.records = _load_records(index_path)
    else:
        _run_index_stage(state)


def _run_music_stage(state: PipelineState) -> None:
    state.logger.info("MUSIC start")
    state.music = analyze_music(state.config, state.toolchain)
    state.logger.info("MUSIC complete: %.2fs, %.2f BPM", state.music.duration, state.music.tempo)


def _load_music_from_cache(config: PipelineConfig) -> MusicAnalysis:
    cache_path = config.music_analysis_dir / "music_analysis_cache.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    data = payload.get("data", payload)
    return MusicAnalysis(
        source_file=Path(data["source_file"]),
        duration=float(data["duration"]),
        tempo=float(data["tempo"]),
        beats=[float(value) for value in data.get("beats", [])],
        strong_beats=[float(value) for value in data.get("strong_beats", [])],
        bars=[float(value) for value in data.get("bars", [])],
        onsets=[float(value) for value in data.get("onsets", [])],
        edit_points=list(data.get("edit_points", [])),
        energy_times=[float(value) for value in data.get("energy_times", [])],
        rms=[float(value) for value in data.get("rms", [])],
        onset_strength=[float(value) for value in data.get("onset_strength", [])],
        novelty=[float(value) for value in data.get("novelty", [])],
        structure_regions=list(data.get("structure_regions", [])),
        confidence={str(key): float(value) for key, value in (data.get("confidence") or {}).items()},
        preview_music_in=data.get("preview_music_in"),
        preview_music_out=data.get("preview_music_out"),
        preview_reason=str(data.get("preview_reason", "")),
    )


def _analysis_cache_paths(state: PipelineState, record: MediaRecord, source: Path) -> tuple[Path, Path, str]:
    parameters = {
        "analysis_fps": state.config.analysis_fps,
        "sample_rate": state.config.analysis_sample_rate,
        "source_for_analysis": str(source),
        "ffmpeg_version": state.toolchain.ffmpeg_version,
    }
    key = cache_key(record.fingerprint, "video_analysis", parameters)
    return state.config.analysis_dir / f"video_{key[:20]}.json", state.config.analysis_dir / f"audio_{key[:20]}.json", key


def _run_video_stage(state: PipelineState) -> None:
    state.logger.info("PROXY start")
    proxies: dict[str, Path] = {}
    for record in state.records:
        try:
            proxy = build_proxy(record, state.config, state.toolchain)
            if proxy:
                proxies[str(record.file_path)] = proxy
        except Exception as exc:
            state.logger.error("PROXY skipped %s: %s", record.file_path, exc)
    state.logger.info("PROXY complete: %d long recordings", len(proxies))
    state.logger.info("ANALYZE start")
    analyses: dict[str, VideoAnalysis] = {}
    for record in state.records:
        source = proxies.get(str(record.file_path), record.file_path)
        video_cache, audio_cache, key = _analysis_cache_paths(state, record, source)
        try:
            if video_cache.exists() and audio_cache.exists():
                analyses[str(record.file_path)] = _video_from_dict(json.loads(video_cache.read_text(encoding="utf-8")))
                continue
            wav_path = state.config.cache_dir / f"{key[:20]}_analysis.wav"
            extract_analysis_audio(source, wav_path, state.toolchain, state.config.analysis_sample_rate)
            samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
            audio_features = read_cached_json(audio_cache, key)
            if not isinstance(audio_features, dict):
                audio_features = analyze_audio_waveform(samples, sample_rate)
                write_cached_json(audio_cache, key, audio_features)
            analysis = analyze_video_activity(record, source, audio_features, state.config, state.toolchain)
            atomic_write_json(video_cache, analysis.to_dict())
            plot_path = state.config.analysis_dir / f"activity_{key[:16]}.png"
            write_video_analysis(analysis, video_cache, plot_path)
            analyses[str(record.file_path)] = analysis
        except Exception as exc:
            state.logger.error("ANALYZE skipped %s: %s", record.file_path, exc)
    state.analyses = analyses
    state.logger.info("ANALYZE complete: %d sources", len(analyses))


def _run_candidate_stage(state: PipelineState) -> None:
    if state.analyses is None:
        raise RuntimeError("Video analysis must run before candidates")
    state.logger.info("DEDUPE candidate generation start")
    candidates = score_candidates(generate_candidates(state.records, state.analyses, state.config), state.config)
    fingerprint_dir = state.config.analysis_dir / "fingerprints"
    fingerprint_dir.mkdir(parents=True, exist_ok=True)
    fingerprints: dict[str, list[int]] = {}
    for candidate in candidates:
        record = next((item for item in state.records if item.file_path == candidate.source_file), None)
        if record is None:
            continue
        key = cache_key(record.fingerprint, "fingerprint", {"candidate_id": candidate.candidate_id, "duration": candidate.duration})
        path = fingerprint_dir / f"{candidate.candidate_id}.json"
        cached = read_cached_json(path, key)
        if isinstance(cached, list):
            fingerprints[candidate.candidate_id] = [int(value) for value in cached]
        else:
            values = fingerprint_candidate(candidate, candidate.source_file, state.toolchain)
            fingerprints[candidate.candidate_id] = values
            write_cached_json(path, key, values)
    result = deduplicate_candidates(candidates, fingerprints, state.config.dedupe_threshold)
    deduped = [candidate for group in result.groups for candidate in group]
    state.candidates = score_candidates(deduped, state.config)
    state.dedupe = result
    write_candidates(
        state.candidates,
        state.config.analysis_dir / "highlight_candidates.json",
        state.config.analysis_dir / "highlight_candidates.csv",
    )
    atomic_write_json(state.config.analysis_dir / "dedupe_summary.json", result.to_dict())
    state.logger.info("DEDUPE complete: %d candidates, %d groups", len(state.candidates), len(result.groups))


def _run_review_and_edit_stage(state: PipelineState) -> None:
    if state.music is None or state.candidates is None:
        raise RuntimeError("Music and candidates must exist before edit construction")
    state.logger.info("EDIT start")
    render_review_assets(state.candidates, state.config, state.toolchain)
    state.edit = build_preview_edit(state.candidates, state.music, state.config)
    write_edit_list(
        state.edit,
        state.config.preview_analysis_dir / "preview_edit.json",
        state.config.preview_analysis_dir / "preview_timeline.txt",
    )
    state.logger.info("EDIT complete: %.3fs, %d shots", state.edit.duration, len(state.edit.shots))


def _run_analysis_pipeline(config: PipelineConfig, logger: logging.Logger | None = None) -> PipelineState:
    del logger
    state = _new_state(config)
    before = config.analysis_dir / "raw_manifest_before.json"
    if not before.exists():
        atomic_write_json(before, _raw_manifest(config))
    _run_index_stage(state)
    _run_music_stage(state)
    _run_video_stage(state)
    _run_candidate_stage(state)
    _run_review_and_edit_stage(state)
    return state


def _load_edit_and_music(config: PipelineConfig) -> tuple[EditDecisionList, MusicAnalysis]:
    edit_path = config.preview_analysis_dir / "preview_edit.json"
    if not edit_path.exists():
        raise FileNotFoundError("preview_edit.json is missing; run `python main.py all --dry-run` first")
    if not (config.music_analysis_dir / "music_analysis_cache.json").exists():
        raise FileNotFoundError("music analysis is missing; run `python main.py all --dry-run` first")
    return _edit_from_dict(json.loads(edit_path.read_text(encoding="utf-8"))), _load_music_from_cache(config)


def write_preview_report(
    edit: EditDecisionList,
    music: MusicAnalysis,
    candidates: list[Candidate],
    output_probe: dict[str, object],
    toolchain: Toolchain,
    path: Path,
) -> None:
    durations = [shot.duration for shot in edit.shots]
    offsets = [abs(shot.sync_offset) for shot in edit.shots]
    transition_counts = _transition_counts(edit.shots)
    candidate_lookup = {candidate.candidate_id: candidate for candidate in candidates}
    selected_sources = [shot.source.name for shot in edit.shots]
    lines = [
        "# Battlefield Montage V1 Preview Report",
        "",
        "## A. Video analysis result",
        f"Analyzed candidates: {len(candidates)}; selected shots: {len(edit.shots)}.",
        f"Preview duration: {edit.duration:.3f}s; output probe: {json.dumps(output_probe, ensure_ascii=False)}.",
        "Gameplay continuity is prioritized over exact beat alignment.",
        "",
        "## B. Music structure analysis",
        f"Source: {music.source_file}",
        f"Duration: {music.duration:.3f}s; estimated BPM: {music.tempo:.3f}.",
        f"Preview music_in: {edit.music_in:.3f}; music_out: {edit.music_out:.3f}.",
        f"Reason: {edit.music_reason}",
        f"Confidence: {json.dumps(music.confidence, ensure_ascii=False)}",
        "Section roles are heuristic energy labels; confidence does not assert verified verse/chorus semantics.",
        f"Regions: {json.dumps(music.structure_regions, ensure_ascii=False)}",
        "",
        "## C. Selected candidates",
        *[f"- {shot.source.name} [{shot.source_in:.3f}, {shot.source_out:.3f}] score={shot.candidate_score:.3f}" for shot in edit.shots],
        "",
        "## D. Dedupe",
        "Duplicate groups are represented at most once in the preview EDL.",
        *[f"- {shot.duplicate_group or 'none'}: {shot.source.name}" for shot in edit.shots],
        "",
        "## E. Music interval",
        f"{edit.music_in:.3f}s -> {edit.music_out:.3f}s; {edit.music_reason}",
        "",
        "## F-H. Shot and transition statistics",
        f"Average shot: {statistics.mean(durations) if durations else 0.0:.3f}s",
        f"Shortest shot: {min(durations) if durations else 0.0:.3f}s",
        f"Longest shot: {max(durations) if durations else 0.0:.3f}s",
        f"Transitions: {json.dumps(transition_counts, ensure_ascii=False)}",
        "",
        "## I. Beat sync offsets",
        f"Mean absolute offset: {statistics.mean(offsets) * 1000 if offsets else 0.0:.2f}ms",
        f"P95 absolute offset: {_p95(offsets) * 1000:.2f}ms",
        "",
        "## J. Game audio strategy",
        "Game audio remains in the mix. Base music is reduced, high-activity shots apply additional music ducking and a small game gain boost, with sidechain attack 50ms and release 500ms; final loudness targets approximately -14 LUFS / -1 dBTP.",
        "",
        "## K. FFmpeg/NVENC",
        f"FFmpeg: {toolchain.ffmpeg} ({toolchain.ffmpeg_version})",
        f"ffprobe: {toolchain.ffprobe} ({toolchain.ffprobe_version})",
        f"h264_nvenc runtime: {toolchain.nvenc_h264}; hevc_nvenc runtime: {toolchain.nvenc_hevc}",
        "",
        "## L. Preview path",
        f"`{path.parent.parent.parent / 'output' / 'preview_60s.mp4'}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_preview_stage(config: PipelineConfig, state: PipelineState | None = None) -> Path:
    if state is None:
        state = _new_state(config)
    edit, music = _load_edit_and_music(config)
    state.logger.info("RENDER preview start")
    output = config.output_dir / "preview_60s.mp4"
    render_edit(edit, music, config, state.toolchain, output)
    probe = probe_output(output, state.toolchain)
    if not probe["has_video"] or not probe["has_audio"]:
        raise RuntimeError(f"Preview is missing a required stream: {probe}")
    if int(probe["width"]) != config.output_width or int(probe["height"]) != config.output_height:
        raise RuntimeError(f"Preview dimensions are wrong: {probe}")
    before_path = config.analysis_dir / "raw_manifest_before.json"
    if before_path.exists():
        before = json.loads(before_path.read_text(encoding="utf-8"))
        after = _raw_manifest(config)
        if before != after:
            raise RuntimeError("RAW manifest changed during pipeline run")
    candidates_path = config.analysis_dir / "highlight_candidates.json"
    candidates_data = json.loads(candidates_path.read_text(encoding="utf-8")) if candidates_path.exists() else {"candidates": []}
    candidates = [_candidate_from_dict(item) for item in candidates_data.get("candidates", [])]
    write_preview_report(edit, music, candidates, probe, state.toolchain, config.analysis_dir / "preview_report.md")
    state.logger.info("RENDER preview complete: %s", output)
    return output


def _verify_preview(config: PipelineConfig) -> int:
    state = _new_state(config)
    output = config.output_dir / "preview_60s.mp4"
    probe = probe_output(output, state.toolchain)
    valid = bool(probe["has_video"] and probe["has_audio"] and int(probe["width"]) == config.output_width and int(probe["height"]) == config.output_height and 45.0 <= float(probe["duration"]) <= 60.0)
    print(json.dumps({"valid": valid, "probe": probe}, ensure_ascii=False, indent=2))
    return 0 if valid else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Battlefield gameplay-aware highlight montage pipeline")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("index", "analyze-video", "analyze-music", "candidates", "review", "render-preview", "render-fast", "render-full", "verify-preview", "render-preview-v2", "verify-preview-v2", "render-preview-v2-long", "verify-preview-v2-long", "render-preview-v2-quality", "verify-preview-v2-quality", "v6-calibrate", "v6-review", "v6-verify"):
        subparsers.add_parser(command)
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--dry-run", action="store_true", help="run analysis and EDL generation without rendering video")
    all_v2_parser = subparsers.add_parser("all-v2")
    all_v2_parser.add_argument("--dry-run", action="store_true", help="run V2 analysis and EDL generation without rendering video")
    all_v2_long_parser = subparsers.add_parser("all-v2-long")
    all_v2_long_parser.add_argument("--dry-run", action="store_true", help="run the 75-90 second V2 analysis and EDL without rendering video")
    all_v2_quality_parser = subparsers.add_parser("all-v2-quality")
    all_v2_quality_parser.add_argument("--dry-run", action="store_true", help="run the 120-150 second quality analysis and EDL without rendering video")
    v6_scan_parser = subparsers.add_parser("v6-scan")
    v6_scan_parser.add_argument("--dry-run", action="store_true", help="emit direct-RAW crop commands without decoding")
    return parser


def _long_v2_config(config: PipelineConfig) -> PipelineConfig:
    """Return the explicit longer-preview profile without changing normal V2 defaults."""
    return replace(
        config,
        preview_min_duration=45.0,
        preview_max_duration=90.0,
        v2_output_name="preview_quality_v4.mp4",
        v2_min_shots=3,
        v2_max_shots=8,
        v2_short_clip_max_duration=30.0,
        v2_short_clip_prior=0.85,
        beam_max_expansions=max(config.beam_max_expansions, 32768),
        v2_music_window_policy="representative",
    )


def _quality_v2_config(config: PipelineConfig) -> PipelineConfig:
    """Return the V5 quality profile with a strict verified-kill admission gate."""
    return replace(
        config,
        preview_min_duration=120.0,
        preview_max_duration=150.0,
        v2_output_name="preview_quality_v5.mp4",
        v2_min_shots=10,
        v2_max_shots=24,
        v2_short_clip_max_duration=30.0,
        v2_short_clip_prior=0.85,
        beam_max_expansions=max(config.beam_max_expansions, 65536),
        v2_music_window_policy="representative",
        v5_profile="quality",
        v5_min_kill_semantic_confidence=0.80,
        v5_min_killfeed_evidence=0.25,
        v5_min_reward_evidence=0.10,
        v5_min_verified_kills_per_shot=1,
        v5_kill_density_window_s=6.0,
        v5_kill_density_target=4,
        v5_kill_count_weight=0.20,
        v5_kill_density_weight=0.35,
        v5_max_sequences_per_candidate=6,
        v5_sequence_context_s=1.25,
        v5_min_rapid_context_tail_s=0.25,
    )


def run_pipeline(command: str, config_path: Path, dry_run: bool = False) -> int:
    if command in {"render-fast", "render-full"}:
        print("Full Fast Montage and Full Highlights rendering is intentionally blocked until the human approves preview_60s.mp4.", file=sys.stderr)
        return 2
    config = load_config(Path(config_path))
    if command in {"v6-calibrate", "v6-scan", "v6-review", "v6-verify"}:
        return run_v6_command(command, Path(config_path), dry_run=dry_run)
    if command in {"all-v2-long", "render-preview-v2-long", "verify-preview-v2-long"}:
        config = _long_v2_config(config)
    if command in {"all-v2-quality", "render-preview-v2-quality", "verify-preview-v2-quality"}:
        config = _quality_v2_config(config)
    if command == "verify-preview":
        return _verify_preview(config)
    if command in {"verify-preview-v2", "verify-preview-v2-long", "verify-preview-v2-quality"}:
        return verify_preview_v2(config)
    if command == "render-preview":
        render_preview_stage(config)
        return 0
    if command in {"render-preview-v2", "render-preview-v2-long", "render-preview-v2-quality"}:
        render_preview_v2_stage(config)
        return 0
    if command in {"all-v2", "all-v2-long", "all-v2-quality"}:
        v2_state = run_v2_analysis_pipeline(config)
        if dry_run:
            return 0
        render_preview_v2_stage(config, v2_state)
        return 0
    if command == "all":
        state = _run_analysis_pipeline(config)
        if dry_run:
            if state is not None:
                state.logger.info("DRY-RUN complete: render skipped")
            return 0
        render_preview_stage(config, state)
        return 0
    state = _new_state(config)
    if command == "index":
        _run_index_stage(state)
    elif command == "analyze-music":
        _run_music_stage(state)
    elif command == "analyze-video":
        _load_or_index(state)
        _run_video_stage(state)
    elif command in {"candidates", "review"}:
        _load_or_index(state)
        if not (state.config.music_analysis_dir / "music_analysis_cache.json").exists():
            _run_music_stage(state)
        state.analyses = {}
        for path in state.config.analysis_dir.glob("video_*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            analysis = _video_from_dict(data)
            state.analyses[str(analysis.source_file)] = analysis
        _run_candidate_stage(state)
        if command == "review":
            _run_review_and_edit_stage(state)
    else:
        raise ValueError(f"Unknown command: {command}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_pipeline(args.command, args.config, getattr(args, "dry_run", False))


if __name__ == "__main__":
    raise SystemExit(main())
