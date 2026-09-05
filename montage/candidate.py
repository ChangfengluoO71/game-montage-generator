"""Candidate window generation from video activity analyses."""

from __future__ import annotations

import csv
import io
import json
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .cache import atomic_write_bytes
from .config import PipelineConfig, is_within, load_config
from .models import Candidate, CandidateVariant, MediaRecord, VideoAnalysis
from .video_analysis import derive_candidate_feature_runs


def _candidate_id(source: Path, start: float, end: float) -> str:
    raw = f"{source.resolve(strict=False)}|{start:.3f}|{end:.3f}".encode("utf-8")
    return "cand-" + hashlib.sha1(raw).hexdigest()[:12]


def _window_values(values: Sequence[float], times: Sequence[float], start: float, end: float) -> np.ndarray:
    clocks = np.asarray(times, dtype=float)
    numbers = np.asarray(values, dtype=float)
    if clocks.size == 0 or numbers.size == 0:
        return np.zeros(1, dtype=float)
    size = min(clocks.size, numbers.size)
    selected = numbers[:size][(clocks[:size] >= start) & (clocks[:size] <= end)]
    return selected if selected.size else numbers[:size]


def _mean_score(values: Sequence[float], times: Sequence[float], start: float, end: float, high: bool = False) -> float:
    selected = _window_values(values, times, start, end)
    if selected.size == 0:
        return 0.0
    value = np.percentile(selected, 80) if high else np.mean(selected)
    return float(np.clip(value, 0.0, 1.0))


def trim_non_content_edges(candidate: Candidate, analysis: VideoAnalysis) -> Candidate:
    if not analysis.times or not analysis.black_ratio:
        return candidate
    times = np.asarray(analysis.times, dtype=float)
    black = np.asarray(analysis.black_ratio, dtype=float)
    size = min(times.size, black.size)
    if size == 0:
        return candidate
    start = candidate.source_start
    end = candidate.source_end
    for index in range(size):
        if times[index] < start or black[index] < 0.90:
            break
        start = max(start, float(times[index]))
    for index in range(size - 1, -1, -1):
        if times[index] > end or black[index] < 0.90:
            break
        end = min(end, float(times[index]))
    if end - start < 1.0 or end <= start:
        return candidate
    return Candidate(**{**candidate.__dict__, "source_start": start, "source_end": end, "duration": end - start})


def _fallback_window(analysis: VideoAnalysis, duration: float, config: PipelineConfig) -> tuple[float, float] | None:
    if not analysis.times or not analysis.activity:
        return None
    activity = np.asarray(analysis.activity, dtype=float)
    times = np.asarray(analysis.times, dtype=float)
    index = int(np.argmax(activity))
    center = float(times[min(index, times.size - 1)])
    core_start = max(0.0, center - config.highlight_min_duration)
    core_end = min(duration, core_start + max(config.highlight_min_duration, 8.0))
    if core_end - core_start < config.highlight_min_duration:
        core_start = max(0.0, core_end - config.highlight_min_duration)
    return core_start, core_end


_FEATURE_RUN_NAMES = {
    "stationary_ads",
    "same_view",
    "downtime",
    "repetitive_fire",
    "same_source_recent",
    "same_environment_recent",
    "danger_escalation",
    "motion",
    "visual_novelty",
}


def _feature_runs(window: Mapping[str, float], analysis: VideoAnalysis, start: float, end: float) -> dict[str, float]:
    values = derive_candidate_feature_runs(
        analysis.times, analysis.motion, analysis.visual, analysis.audio,
        analysis.continuity, analysis.activity, start, end,
    )
    core_start = float(window.get("start", start))
    core_end = float(window.get("end", end))
    if abs(core_start - start) <= 0.001 and abs(core_end - end) <= 0.001:
        values.update({
            name: float(value)
            for name, value in window.items()
            if name in _FEATURE_RUN_NAMES and isinstance(value, (int, float))
        })
    return values


def generate_candidates(
    records: Sequence[MediaRecord],
    analyses: Mapping[str, VideoAnalysis],
    config: PipelineConfig,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for record in sorted(records, key=lambda item: str(item.file_path)):
        analysis = analyses.get(str(record.file_path)) or analyses.get(record.file_path.as_posix())
        if analysis is None:
            continue
        if record.duration <= config.short_clip_threshold:
            windows = [(0.0, record.duration, _feature_runs({}, analysis, 0.0, record.duration))]
            human_prior = config.v2_short_clip_prior
        else:
            core_windows = [
                window
                for window in analysis.candidate_windows
                if float(window.get("end", 0.0)) > float(window.get("start", 0.0))
            ]
            if not core_windows:
                fallback = _fallback_window(analysis, record.duration, config)
                core_windows = [{"start": fallback[0], "end": fallback[1]}] if fallback else []
            windows = []
            for window in core_windows:
                start = max(0.0, float(window["start"]) - config.pre_roll)
                end = min(record.duration, float(window["end"]) + config.post_roll)
                windows.append((start, end, _feature_runs(window, analysis, start, end)))
            human_prior = 0.0
        for start, end, feature_runs in windows:
            if end - start < max(1.0, config.highlight_min_duration):
                continue
            candidate = Candidate(
                candidate_id=_candidate_id(record.file_path, start, end),
                source_file=record.file_path,
                source_start=round(start, 3),
                source_end=round(end, 3),
                duration=round(end - start, 3),
                human_selection_score=human_prior,
                audio_score=_mean_score(analysis.audio, analysis.times, start, end, high=True),
                motion_score=_mean_score(analysis.motion, analysis.times, start, end, high=True),
                visual_score=_mean_score(analysis.visual, analysis.times, start, end, high=True),
                continuity_score=_mean_score(analysis.continuity, analysis.times, start, end),
                combat_intensity=_mean_score(analysis.activity, analysis.times, start, end, high=True),
                source_category=record.category,
                rationale="Human-saved short clip retained as one coherent candidate." if human_prior else "Merged activity/audio window from a long recording with context roll.",
                feature_runs=feature_runs,
            )
            candidates.append(trim_non_content_edges(candidate, analysis) if human_prior else candidate)
    return candidates


def write_candidates(candidates: Sequence[Candidate], json_path: Path, csv_path: Path) -> None:
    records = [candidate.to_dict() for candidate in candidates]
    atomic_write_bytes(json_path, json.dumps({"candidates": records}, ensure_ascii=False, indent=2).encode("utf-8"))
    fields = [
        "candidate_id",
        "source_file",
        "source_start",
        "source_end",
        "duration",
        "human_selection_score",
        "audio_score",
        "motion_score",
        "visual_score",
        "continuity_score",
        "duplicate_group",
        "final_score",
        "combat_intensity",
        "uniqueness",
        "source_category",
        "rationale",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for candidate in candidates:
        writer.writerow(candidate.to_dict())
    atomic_write_bytes(csv_path, buffer.getvalue().encode("utf-8-sig"))


def _default_writer_config() -> PipelineConfig:
    return load_config(Path(__file__).resolve().parents[1] / "config.yaml")


def _assert_v2_candidate_destinations(json_path: Path, csv_path: Path, config: PipelineConfig) -> None:
    for path in (json_path, csv_path):
        destination = path.resolve(strict=False)
        if (
            destination == config.baseline_output_path.resolve(strict=False)
            or is_within(destination, config.raw_dir)
            or not is_within(destination, config.work_dir)
        ):
            raise ValueError(f"V2 candidate artifacts must remain beneath work_dir: {destination}")


def write_v2_candidates(
    variants: Sequence[CandidateVariant], json_path: Path, csv_path: Path, config: PipelineConfig | None = None
) -> None:
    """Write V2 candidate diagnostics, including every score component and penalty."""
    config = config or _default_writer_config()
    _assert_v2_candidate_destinations(json_path, csv_path, config)
    records = [variant.to_dict() for variant in variants]
    atomic_write_bytes(json_path, json.dumps({"candidates": records}, ensure_ascii=False, indent=2).encode("utf-8"))
    fields = sorted({key for record in records for key in record} | {
        "variant_id", "parent_candidate_id", "final_score", "score_components", "penalty_values",
    } | {
        name
        for record in records
        for mapping_name in ("score_components", "penalty_values")
        for name in (record.get(mapping_name) or {})
    })
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        flattened = dict(record)
        for name, value in (record.get("score_components") or {}).items():
            flattened[name] = value
        for name, value in (record.get("penalty_values") or {}).items():
            flattened[name] = value
        writer.writerow(flattened)
    atomic_write_bytes(csv_path, buffer.getvalue().encode("utf-8-sig"))
