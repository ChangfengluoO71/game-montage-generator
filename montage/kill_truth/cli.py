"""CLI orchestration for the isolated V6 Kill Truth phase."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from ..cache import atomic_write_json
from ..config import PipelineConfig, ensure_runtime_dirs, load_config
from ..media_index import build_media_index, write_media_index
from ..models import MediaRecord
from ..toolchain import Toolchain, discover_toolchain
from .calibration import default_profile, render_calibration_report, run_calibration
from .curation import init_curation_ledger
from .debug import render_skull_state_timeline
from .metrics import match_event_times
from .profile import HudProfile, load_profile
from .review import render_kill_review
from .scanner import V6ScanConfig, V6ScanResult, scan_source, source_id_for_path


def _logger(config: PipelineConfig) -> logging.Logger:
    ensure_runtime_dirs(config)
    logger = logging.getLogger(f"battlefield-v6-{uuid.uuid4().hex}")
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


def _write_environment(config: PipelineConfig, toolchain: Toolchain, profile: HudProfile | None = None) -> None:
    atomic_write_json(
        config.v6_environment_path,
        {
            "schema": "v6-environment-v1",
            "stage": "v6-kill-truth",
            "python": sys.version,
            "platform": platform.platform(),
            "raw_dir": str(config.raw_dir.resolve(strict=False)),
            "work_dir": str(config.work_dir.resolve(strict=False)),
            "output_dir": str(config.output_dir.resolve(strict=False)),
            "selected_ffmpeg": str(toolchain.ffmpeg.resolve(strict=False)),
            "selected_ffprobe": str(toolchain.ffprobe.resolve(strict=False)),
            "ffmpeg_version": toolchain.ffmpeg_version,
            "ffprobe_version": toolchain.ffprobe_version,
            "nvenc_h264_runtime": toolchain.nvenc_h264,
            "nvenc_hevc_runtime": toolchain.nvenc_hevc,
            "toolchain_selection_reason": toolchain.selected_reason,
            "toolchain": toolchain.to_dict(),
            "hud_profile_id": profile.profile_id if profile else None,
            "gpu": _gpu_report(),
        },
    )


def _print_json(payload: Any) -> None:
    """Keep CLI diagnostics printable on a legacy Windows console code page."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    print(json.dumps(payload, ensure_ascii=False, indent=2))


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


def _records(config: PipelineConfig, toolchain: Toolchain, logger: logging.Logger) -> list[MediaRecord]:
    index_path = config.analysis_dir / "media_index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            records = [_record_from_dict(item) for item in payload.get("records", [])]
            if records:
                logger.info("V6 media index cache hit: %s records", len(records))
                return records
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("V6 media index cache unusable; rebuilding: %s", exc)
    records = build_media_index(config, toolchain, logger)
    write_media_index(records, config.analysis_dir / "media_index.json", config.analysis_dir / "media_index.csv")
    return records


def _ensure_curation_ledger(config: PipelineConfig, records: list[MediaRecord]) -> None:
    if config.curation_ledger_path.exists():
        try:
            payload = json.loads(config.curation_ledger_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            payload = init_curation_ledger([])
    else:
        payload = init_curation_ledger([])
    sources = dict(payload.get("sources") or {})
    for record in records:
        source_id = source_id_for_path(record.file_path)
        sources.setdefault(
            source_id,
            {
                "source_id": source_id,
                "source_path": str(record.file_path.resolve(strict=False)),
                "status": "UNREVIEWED",
                "editorial_excluded": False,
                "notes": "",
                "updated_at": None,
            },
        )
    atomic_write_json(
        config.curation_ledger_path,
        {
            "schema": "curation-ledger-v1",
            "semantic_truth_independent": True,
            "sources": sources,
        },
    )


def run_v6_calibrate(config_path: Path) -> int:
    config = load_config(config_path)
    logger = _logger(config)
    toolchain = discover_toolchain(config)
    logger.info("V6 FFmpeg selected path=%s version=%s", toolchain.ffmpeg.resolve(strict=False), toolchain.ffmpeg_version)
    logger.info("V6 ffprobe selected path=%s version=%s", toolchain.ffprobe.resolve(strict=False), toolchain.ffprobe_version)
    logger.info("V6 NVENC runtime h264_nvenc=%s hevc_nvenc=%s", toolchain.nvenc_h264, toolchain.nvenc_hevc)
    records = _records(config, toolchain, logger)
    _ensure_curation_ledger(config, records)
    profile = default_profile(config, toolchain)
    _write_environment(config, toolchain, profile)
    result = run_calibration(config, toolchain, records, profile)
    report_path = render_calibration_report(config, profile, result, toolchain)
    logger.info(
        "V6 calibration gate=%s frame_panel_precision=%.3f frame_panel_recall=%.3f report=%s",
        result["calibration_gate"],
        float(result["frame_panel_precision"]),
        float(result["frame_panel_recall"]),
        report_path,
    )
    _print_json({"calibration": result, "report": str(report_path)})
    return 0 if result["calibration_gate"] == "PASS" else 1


def _load_calibrated_profile(config: PipelineConfig) -> HudProfile:
    path = config.v6_profiles_dir / config.v6_profile_id / "profile.json"
    if not path.exists():
        raise RuntimeError(f"V6 profile is missing; run v6-calibrate first: {path}")
    return load_profile(path)


def _write_indexes(
    config: PipelineConfig,
    results: list[V6ScanResult],
    *,
    records: list[MediaRecord] | None = None,
    profile: HudProfile | None = None,
) -> None:
    events = [event.to_dict() for result in results for event in result.events]
    sequences = [sequence.to_dict() for result in results for sequence in result.sequences]
    result_by_source = {result.source_id: result for result in results}
    source_summaries: list[dict[str, Any]] = []
    source_records = records or [
        MediaRecord(
            file_path=result.source_path,
            file_name=result.source_path.name,
            file_size=0,
            duration=result.duration,
            width=result.width,
            height=result.height,
            fps=0.0,
            codec="",
            bitrate=None,
            audio_codec=None,
            audio_channels=None,
            audio_sample_rate=None,
            creation_time=None,
            category="",
            fingerprint={},
        )
        for result in results
    ]
    for record in source_records:
        source_id = source_id_for_path(record.file_path)
        result = result_by_source.get(source_id)
        if result is not None:
            source_summaries.append(
                {
                    "source_id": result.source_id,
                    "source_path": str(result.source_path.resolve(strict=False)),
                    "resolution": [result.width, result.height],
                    "status": result.status,
                    "hud_profile_id": result.hud_profile_id,
                    "frames_scanned": result.frames_scanned,
                    "dense_frames_scanned": result.dense_frames_scanned,
                    "event_count": len(result.events),
                    "sequence_count": len(result.sequences),
                    "refinement_request_count": len(result.refinement_requests),
                    "error_count": len(result.errors),
                }
            )
        else:
            source_summaries.append(
                {
                    "source_id": source_id,
                    "source_path": str(record.file_path.resolve(strict=False)),
                    "resolution": [record.width, record.height],
                    "status": "UNPROFILED",
                    "hud_profile_id": None,
                    "frames_scanned": 0,
                    "dense_frames_scanned": 0,
                    "event_count": 0,
                    "sequence_count": 0,
                    "refinement_request_count": 0,
                    "error_count": 0,
                    "reason": "no calibrated HUD profile for this resolution",
                }
            )
    source_count = len(source_summaries)
    unprofiled_count = sum(item["status"] == "UNPROFILED" for item in source_summaries)
    error_count = sum(item["status"] == "PARTIAL_ERROR" for item in source_summaries)
    index_status = "COMPLETE" if not unprofiled_count and not error_count else (
        "PARTIAL_ERROR" if error_count else "PARTIAL_UNPROFILED"
    )
    profile_resolution = [profile.width, profile.height] if profile else None
    envelope = {
        "schema": "v6-kill-event-index-v1",
        "status": index_status,
        "source_count": source_count,
        "scanned_source_count": len(results),
        "unprofiled_source_count": unprofiled_count,
        "calibrated_profile_resolution": profile_resolution,
        "event_count": len(events),
        "events": events,
        "sources": source_summaries,
    }
    atomic_write_json(config.v6_event_index_path, envelope)
    atomic_write_json(
        config.v6_sequence_index_path,
        {
            "schema": "v6-kill-sequence-index-v1",
            "status": index_status,
            "source_count": source_count,
            "scanned_source_count": len(results),
            "unprofiled_source_count": unprofiled_count,
            "sequence_count": len(sequences),
            "sequences": sequences,
            "sources": source_summaries,
        },
    )


def run_v6_scan(config_path: Path, *, dry_run: bool = False) -> int:
    config = load_config(config_path)
    logger = _logger(config)
    calibration_path = config.v6_kill_truth_dir / "calibration" / "calibration_results.json"
    if not calibration_path.exists():
        print("V6 scan blocked: calibration results do not exist; run v6-calibrate first.", file=sys.stderr)
        return 2
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("calibration_gate") != "PASS" or not calibration.get("full_scan_authorized", False):
        print("V6 scan blocked: calibration gate is not PASS; no full RAW scan was started.", file=sys.stderr)
        return 2
    toolchain = discover_toolchain(config)
    profile = _load_calibrated_profile(config)
    _write_environment(config, toolchain, profile)
    records = _records(config, toolchain, logger)
    settings = V6ScanConfig(
        coarse_fps=config.v6_coarse_fps,
        dense_fps=config.v6_dense_fps,
        panel_disappear_s=config.v6_panel_disappear_s,
        refinement_radius_s=config.v6_refinement_radius_s,
    )
    if dry_run:
        from .scanner import dry_run_source

        payload = {
            "schema": "v6-scan-dry-run-v1",
            "source_count": len(records),
            "profile_id": profile.profile_id,
            "sources": [
                {
                    **dry_run_source(record, profile, toolchain, settings),
                    "calibrated": (record.width, record.height) == (profile.width, profile.height),
                    "will_scan": (record.width, record.height) == (profile.width, profile.height),
                }
                for record in records
            ],
        }
        path = config.v6_reports_dir / "v6_scan_dry_run.json"
        atomic_write_json(path, payload)
        _print_json(payload)
        return 0
    results: list[V6ScanResult] = []
    for index, record in enumerate(records, start=1):
        if (record.width, record.height) != (profile.width, profile.height):
            logger.warning("V6 source %s/%s skipped: no calibrated profile for %sx%s", index, len(records), record.width, record.height)
            continue
        logger.info("V6 scanning source %s/%s: %s", index, len(records), record.file_name)
        results.append(
            scan_source(
                record,
                profile,
                toolchain,
                scan_config=settings,
                cache_dir=config.v6_cache_dir,
                raw_dir=config.raw_dir,
            )
        )
    _write_indexes(config, results, records=records, profile=profile)
    _print_json({"scanned_sources": len(results), "events": sum(len(item.events) for item in results)})
    return 0


def _load_cached_results(config: PipelineConfig) -> list[V6ScanResult]:
    results: list[V6ScanResult] = []
    for path in config.v6_cache_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, dict) and data.get("schema") == "v6-scan-result-v1":
                results.append(V6ScanResult.from_dict(data))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return results


def run_v6_review(config_path: Path) -> int:
    config = load_config(config_path)
    logger = _logger(config)
    profile = _load_calibrated_profile(config)
    results = _load_cached_results(config)
    html_path = render_kill_review(results, {profile.profile_id: profile}, config.v6_review_dir, discover_toolchain(config))
    for result in results:
        render_skull_state_timeline(result.coarse_states, result.events, config.v6_debug_dir / f"{result.source_id}_skull_state_timeline.png", title=result.source_path.name)
    logger.info("V6 kill review generated: %s", html_path)
    print(str(html_path))
    return 0


def run_v6_verify(config_path: Path) -> int:
    config = load_config(config_path)
    if not config.v6_event_index_path.exists():
        print("V6 verify pending: kill_event_index_v6.json does not exist.", file=sys.stderr)
        return 2
    if not config.v6_gold_set_path.exists():
        payload = {"status": "GOLD_SET_NOT_READY", "semantic_metrics": None}
        atomic_write_json(config.v6_reports_dir / "v6_verify_report.json", payload)
        _print_json(payload)
        return 2
    gold = json.loads(config.v6_gold_set_path.read_text(encoding="utf-8"))
    if str(gold.get("status", "")).upper() != "READY" or bool((gold.get("provenance") or {}).get("detector_reverse_generated", True)):
        payload = {"status": "GOLD_SET_NOT_READY", "semantic_metrics": None}
        atomic_write_json(config.v6_reports_dir / "v6_verify_report.json", payload)
        _print_json(payload)
        return 2
    tolerance = float(gold.get("tolerance_seconds", 0.5))
    detected_payload = json.loads(config.v6_event_index_path.read_text(encoding="utf-8"))
    segment_windows: dict[str, list[tuple[float, float]]] = {}
    for segment in gold.get("segments", []):
        try:
            segment_windows.setdefault(str(segment["source_id"]), []).append((float(segment["start_seconds"]), float(segment["end_seconds"])))
        except (KeyError, TypeError, ValueError):
            continue
    detected_by_source: dict[str, list[float]] = {}
    for item in detected_payload.get("events", []):
        source_id = str(item.get("source_id", ""))
        timestamp = float(item["confirmation_time"])
        windows = segment_windows.get(source_id)
        if windows and any(start <= timestamp <= end for start, end in windows):
            detected_by_source.setdefault(source_id, []).append(timestamp)
    truth_by_source: dict[str, list[float]] = {}
    for item in gold.get("events", []):
        if str(item.get("manual_label", "")).upper() == "OWN_KILL":
            truth_by_source.setdefault(str(item.get("source_id", "")), []).append(float(item["confirmation_time"]))
    per_source = {}
    for source_id in sorted(set(detected_by_source) | set(truth_by_source)):
        per_source[source_id] = match_event_times(detected_by_source.get(source_id, []), truth_by_source.get(source_id, []), tolerance=tolerance)
    # Aggregate only source-local matches; never match timestamps across files.
    metrics = {
        "tp": sum(item["tp"] for item in per_source.values()),
        "fp": sum(item["fp"] for item in per_source.values()),
        "fn": sum(item["fn"] for item in per_source.values()),
        "matches": [pair for item in per_source.values() for pair in item["matches"]],
        "false_positives": [value for item in per_source.values() for value in item["false_positives"]],
        "false_negatives": [value for item in per_source.values() for value in item["false_negatives"]],
    }
    metrics["precision"] = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if metrics["tp"] + metrics["fp"] else 0.0
    metrics["recall"] = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if metrics["tp"] + metrics["fn"] else 0.0
    metrics["f1"] = 2.0 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"]) if metrics["precision"] + metrics["recall"] else 0.0
    report = {"status": "COMPUTED", "tolerance_seconds": tolerance, "semantic_metrics": metrics, "per_source": per_source, "sequence_metrics": None, "rapid_multikill_recall": None, "stop_rule": {"montage_allowed": False, "reason": "Precision-first gate requires sequence and rapid metrics; current gold set has no manually labelled sequences."}}
    atomic_write_json(config.v6_reports_dir / "v6_verify_report.json", report)
    _print_json(report)
    return 0


def run_v6_command(command: str, config_path: Path, *, dry_run: bool = False) -> int:
    if command == "v6-calibrate":
        return run_v6_calibrate(config_path)
    if command == "v6-scan":
        return run_v6_scan(config_path, dry_run=dry_run)
    if command == "v6-review":
        return run_v6_review(config_path)
    if command == "v6-verify":
        return run_v6_verify(config_path)
    raise ValueError(f"Unknown V6 command: {command}")
