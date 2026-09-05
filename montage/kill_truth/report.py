"""Honest V6 Kill Truth delivery reports.

The report deliberately keeps detector calibration metrics separate from
semantic OwnKillEvent metrics.  A detector can pass a small frame calibration
set while the full event precision/recall is still unknown until a human gold
set exists.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..cache import atomic_write_json
from ..models import MediaRecord
from ..toolchain import Toolchain
from .metrics import match_event_times
from .models import KillSequence, OwnKillEvent
from .profile import HudProfile
from .scanner import V6ScanResult


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(float(value), digits) if value is not None else None


def _event_map(results: Iterable[V6ScanResult]) -> dict[str, OwnKillEvent]:
    return {event.event_id: event for result in results for event in result.events}


def _rapid_intervals(results: Iterable[V6ScanResult]) -> list[float]:
    intervals: list[float] = []
    for result in results:
        by_sequence: dict[str, list[OwnKillEvent]] = {}
        for event in result.events:
            by_sequence.setdefault(event.sequence_id, []).append(event)
        for events in by_sequence.values():
            ordered = sorted(events, key=lambda item: item.confirmation_time)
            intervals.extend(
                max(0.0, ordered[index].confirmation_time - ordered[index - 1].confirmation_time)
                for index in range(1, len(ordered))
            )
    return intervals


def _gold_metrics(
    results: Iterable[V6ScanResult],
    gold_path: Path,
) -> dict[str, Any] | None:
    if not gold_path.exists():
        return None
    try:
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {"status": "INVALID_GOLD_SET", "event_metrics": None}
    if str(gold.get("status", "READY")).upper() != "READY":
        return {"status": "GOLD_SET_NOT_READY", "event_metrics": None}
    detected_by_source: dict[str, list[float]] = {}
    for result in results:
        detected_by_source[result.source_id] = [event.confirmation_time for event in result.events]
    truth_by_source: dict[str, list[float]] = {}
    for item in gold.get("events", []):
        source_id = str(item.get("source_id", ""))
        if source_id:
            truth_by_source.setdefault(source_id, []).append(float(item["confirmation_time"]))
    tolerance = float(gold.get("tolerance_seconds", 0.5))
    per_source: dict[str, Any] = {}
    aggregate_detected: list[float] = []
    aggregate_truth: list[float] = []
    for source_id in sorted(set(detected_by_source) | set(truth_by_source)):
        metrics = match_event_times(
            detected_by_source.get(source_id, []),
            truth_by_source.get(source_id, []),
            tolerance=tolerance,
        )
        per_source[source_id] = metrics
        aggregate_detected.extend(detected_by_source.get(source_id, []))
        aggregate_truth.extend(truth_by_source.get(source_id, []))
    overall = match_event_times(aggregate_detected, aggregate_truth, tolerance=tolerance)
    return {
        "status": "COMPUTED",
        "tolerance_seconds": tolerance,
        "event_metrics": overall,
        "per_source": per_source,
        "sequence_metrics": None,
        "rapid_multikill_recall": None,
        "note": "Sequence and rapid-pair recall require sequence-level manual labels.",
    }


def build_v6_report(
    results: Iterable[V6ScanResult],
    records: Iterable[MediaRecord],
    profile: HudProfile,
    toolchain: Toolchain,
    *,
    calibration: Mapping[str, Any] | None = None,
    gold_path: Path | None = None,
) -> dict[str, Any]:
    materialized_results = list(results)
    materialized_records = list(records)
    sequences = [sequence for result in materialized_results for sequence in result.sequences]
    events = [event for result in materialized_results for event in result.events]
    intervals = _rapid_intervals(materialized_results)
    type_counts = Counter(sequence.sequence_type for sequence in sequences)
    resolution_counts = Counter((record.width, record.height) for record in materialized_records)
    scanned_ids = {result.source_id for result in materialized_results}
    unprofiled_by_resolution = Counter(
        (record.width, record.height)
        for record in materialized_records
        if record.file_path.resolve(strict=False).stem not in scanned_ids
    )
    dense_requests = sum(len(result.refinement_requests) for result in materialized_results)
    dense_events = [event for event in events if event.dense_refinement_used]
    recovered_jump_requests = 0
    for result in materialized_results:
        for request in result.refinement_requests:
            refined = [
                event
                for event in result.events
                if event.dense_refinement_used
                and request["window_start"] <= event.confirmation_time <= request["window_end"]
            ]
            if len(refined) >= 2 and all(event.kill_count_delta == 1 for event in refined):
                recovered_jump_requests += 1
    calibration_payload = dict(calibration or {})
    gold_metrics = _gold_metrics(materialized_results, gold_path) if gold_path else None
    if len(materialized_results) < len(materialized_records):
        stop_reason = "STOP_PENDING_RESOLUTION_PROFILES_AND_GOLD_SET"
    elif not gold_metrics or gold_metrics.get("status") != "COMPUTED":
        stop_reason = "STOP_PENDING_GOLD_SET"
    else:
        stop_reason = "STOP_AFTER_V6_TRUTH_REVIEW"
    return {
        "schema": "v6-kill-truth-report-v1",
        "stage": "v6-kill-truth",
        "status": stop_reason,
        "scope": {
            "raw_source_count": len(materialized_records),
            "successful_scanned_source_count": len(materialized_results),
            "unprofiled_source_count": len(materialized_records) - len(materialized_results),
            "resolution_inventory": {
                f"{width}x{height}": count for (width, height), count in sorted(resolution_counts.items())
            },
            "unprofiled_by_resolution": {
                f"{width}x{height}": count
                for (width, height), count in sorted(unprofiled_by_resolution.items())
            },
        },
        "detector": {
            "hud_profile_id": profile.profile_id,
            "resolution": [profile.width, profile.height],
            "search_roi": list(profile.search_roi),
            "row_rois": [list(item) for item in profile.row_rois],
            "detector_version": profile.detector_version,
            "normal_template_count": len(profile.normal_template_paths),
            "headshot_template_count": len(profile.headshot_template_paths),
            "headshot_bank_complete": len(profile.headshot_template_paths) >= 3,
            "thresholds": {
                "normal": profile.normal_threshold,
                "headshot": profile.headshot_threshold,
                "geometry": profile.geometry_threshold,
                "panel_structure": profile.panel_structure_threshold,
                "shape": profile.shape_threshold,
                "orange_color": profile.orange_color_threshold,
            },
        },
        "facts": {
            "own_kill_event_count": len(events),
            "kill_sequence_count": len(sequences),
            "sequence_type_counts": {
                name: type_counts.get(name, 0)
                for name in (
                    "SINGLE_KILL",
                    "DOUBLE_KILL",
                    "TRIPLE_KILL",
                    "QUAD_KILL",
                    "MULTI_KILL_5_PLUS",
                )
            },
            "headshot_event_count": sum(event.kill_count_delta for event in events if event.kill_type == "HEADSHOT"),
            "event_type_counts": dict(Counter(event.type for event in events)),
            "rapid_interval_count_lt_1s": sum(interval < 1.0 for interval in intervals),
            "rapid_interval_count_lt_500ms": sum(interval < 0.5 for interval in intervals),
            "shortest_confirmation_interval_seconds": _round(min(intervals) if intervals else None),
        },
        "refinement": {
            "dense_request_count": dense_requests,
            "dense_frame_count": sum(result.dense_frames_scanned for result in materialized_results),
            "dense_event_count": len(dense_events),
            "coarse_jump_requests_recovered_to_individual_events": recovered_jump_requests,
        },
        "calibration": calibration_payload,
        "gold_set": gold_metrics or {"status": "GOLD_SET_NOT_READY", "event_metrics": None},
        "toolchain": {
            "ffmpeg": str(toolchain.ffmpeg.resolve(strict=False)),
            "ffprobe": str(toolchain.ffprobe.resolve(strict=False)),
            "ffmpeg_version": toolchain.ffmpeg_version,
            "ffprobe_version": toolchain.ffprobe_version,
            "h264_nvenc_runtime": toolchain.nvenc_h264,
            "hevc_nvenc_runtime": toolchain.nvenc_hevc,
        },
        "stop_rule": {
            "montage_allowed": False,
            "music_allowed": False,
            "beat_sync_allowed": False,
            "render_allowed": False,
            "reason": stop_reason,
        },
        "known_limitations": [
            "Only the calibrated 1920x1200 HUD profile was scanned; other resolutions remain UNPROFILED.",
            "Frame calibration metrics are not semantic OwnKillEvent precision/recall.",
            "Headshot template bank is incomplete and kill_type remains UNKNOWN_KILL when local evidence is insufficient.",
            "impact_time is nullable and is not used as kill truth.",
            "A human gold set is required before reporting semantic TP/FP/FN or rapid multi-kill recall.",
        ],
        "known_failure_modes": [
            "Training-range bright geometry was a false-positive mode in the previous detector version; the current calibrated-y gate excludes the observed residual block.",
            "Panel fade/occlusion can cause temporary count drops; the state machine preserves the last confirmed count and dense refinement accepts a target reached before a later drop.",
            "Non-calibrated resolution HUD scale/placement has not been inferred or guessed.",
        ],
    }


def render_v6_report(
    results: Iterable[V6ScanResult],
    records: Iterable[MediaRecord],
    profile: HudProfile,
    toolchain: Toolchain,
    *,
    calibration: Mapping[str, Any] | None = None,
    gold_path: Path | None = None,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    payload = build_v6_report(
        results,
        records,
        profile,
        toolchain,
        calibration=calibration,
        gold_path=gold_path,
    )
    atomic_write_json(json_path, payload)
    facts = payload["facts"]
    scope = payload["scope"]
    gold = payload["gold_set"]
    lines = [
        "# Battlefield V6 Kill Truth Report",
        "",
        f"- Status: **{payload['status']}**",
        f"- RAW scope: `{scope['successful_scanned_source_count']}/{scope['raw_source_count']}` successfully scanned; `{scope['unprofiled_source_count']}` UNPROFILED",
        f"- OwnKillEvent: `{facts['own_kill_event_count']}`",
        f"- KillSequence: `{facts['kill_sequence_count']}`",
        f"- Sequence types: `{facts['sequence_type_counts']}`",
        f"- Headshot event count (detector classification): `{facts['headshot_event_count']}`",
        f"- Shortest confirmation interval: `{facts['shortest_confirmation_interval_seconds']}` s",
        f"- Dense refinement requests/frames/events: `{payload['refinement']['dense_request_count']}` / `{payload['refinement']['dense_frame_count']}` / `{payload['refinement']['dense_event_count']}`",
        f"- Coarse jumps recovered to individual events: `{payload['refinement']['coarse_jump_requests_recovered_to_individual_events']}`",
        "",
        "## Gold-set metrics",
        "",
        f"- Status: `{gold.get('status')}`",
        f"- Event TP/FP/FN/Precision/Recall/F1: `{(gold.get('event_metrics') or 'NOT_AVAILABLE')}`",
        "- Semantic metrics are intentionally not inferred from frame calibration or detector confidence.",
        "",
        "## Profile and runtime",
        "",
        f"- HUD profile: `{payload['detector']['hud_profile_id']}` @ `{payload['detector']['resolution']}`",
        f"- Search ROI: `{payload['detector']['search_roi']}`; row ROI: `{payload['detector']['row_rois']}`",
        f"- Detector version: `{payload['detector']['detector_version']}`",
        f"- FFmpeg: `{payload['toolchain']['ffmpeg']}` ({payload['toolchain']['ffmpeg_version']})",
        f"- ffprobe: `{payload['toolchain']['ffprobe']}` ({payload['toolchain']['ffprobe_version']})",
        f"- NVENC runtime: h264=`{payload['toolchain']['h264_nvenc']}`, hevc=`{payload['toolchain']['hevc_nvenc']}`",
        "",
        "## Explicit STOP",
        "",
        "Montage、Music、Beat Sync、Transition optimization、Fast/Full render 均未执行；等待人工 review、gold set 和未 profile 分辨率的独立校准。",
        "",
    ]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return payload
