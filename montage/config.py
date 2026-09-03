"""Configuration loading and runtime directory policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    raw_dir: Path
    work_dir: Path
    output_dir: Path
    music_file: Path
    video_extensions: tuple[str, ...]
    proxy_resolution: tuple[int, int]
    short_clip_threshold: float
    long_clip_threshold: float
    pre_roll: float
    post_roll: float
    analysis_fps: float
    analysis_sample_rate: int
    highlight_min_duration: float
    highlight_max_duration: float
    activity_threshold_percentile: float
    activity_merge_gap: float
    fingerprint_interval: float
    dedupe_threshold: float
    output_width: int
    output_height: int
    target_fps: float
    preview_min_duration: float
    preview_max_duration: float
    preview_preferred_shot_max_duration: float
    baseline_music_in: float
    baseline_music_out: float
    v2_output_name: str
    payoff_analysis_fps: float
    event_merge_window_ms: int
    strong_anchor_threshold: float
    weak_anchor_threshold: float
    max_anchor_count_per_candidate: int
    preferred_macro_duration: tuple[float, float]
    hero_max_duration: float
    beam_width: int
    beam_max_expansions: int
    recent_source_window: int
    recent_environment_window: int
    baseline_music_max_shift: float
    audio_overlap_ms: tuple[int, int]
    impact_tail_max_ms: int
    v2_weights: dict[str, float]
    beam_weights: dict[str, float]
    penalty_weights: dict[str, float]
    roi_profile: dict[str, Any]
    weights: dict[str, float]
    fast_montage_target: str | float
    full_highlight_target: float
    nvenc: dict[str, Any]
    audio_mix: dict[str, Any]

    @property
    def analysis_dir(self) -> Path:
        return self.work_dir / "analysis"

    @property
    def music_analysis_dir(self) -> Path:
        return self.analysis_dir / "music"

    @property
    def music_v2_analysis_dir(self) -> Path:
        return self.analysis_dir / "music_v2"

    @property
    def v2_output_path(self) -> Path:
        return self.output_dir / self.v2_output_name

    @property
    def baseline_output_path(self) -> Path:
        return self.output_dir / "preview_60s.mp4"

    @property
    def music_v2_beat_map_path(self) -> Path:
        return self.music_v2_analysis_dir / "beat_map_v2.json"

    @property
    def music_v2_structure_path(self) -> Path:
        return self.music_v2_analysis_dir / "music_structure_v2.json"

    @property
    def music_v2_energy_curve_path(self) -> Path:
        return self.music_v2_analysis_dir / "energy_curve_v2.csv"

    @property
    def music_v2_analysis_image_path(self) -> Path:
        return self.music_v2_analysis_dir / "music_analysis_v2.png"

    @property
    def payoff_events_v2_path(self) -> Path:
        return self.analysis_dir / "payoff_events_v2.json"

    @property
    def highlight_candidates_v2_path(self) -> Path:
        return self.analysis_dir / "highlight_candidates_v2.json"

    @property
    def highlight_candidates_v2_csv_path(self) -> Path:
        return self.analysis_dir / "highlight_candidates_v2.csv"

    @property
    def dedupe_summary_v2_path(self) -> Path:
        return self.analysis_dir / "dedupe_summary_v2.json"

    @property
    def preview_v2_analysis_dir(self) -> Path:
        return self.analysis_dir / "preview"

    @property
    def preview_v2_edit_path(self) -> Path:
        return self.preview_v2_analysis_dir / "preview_v2_edit.json"

    @property
    def preview_v2_timeline_path(self) -> Path:
        return self.preview_v2_analysis_dir / "preview_v2_timeline.txt"

    @property
    def preview_v2_sync_report_path(self) -> Path:
        return self.preview_v2_analysis_dir / "preview_v2_sync_report.json"

    @property
    def preview_v2_report_path(self) -> Path:
        return self.preview_v2_analysis_dir / "preview_v2_report.md"

    @property
    def preview_v2_timeline_image_path(self) -> Path:
        return self.preview_v2_analysis_dir / "preview_v2_timeline.png"

    @property
    def preview_analysis_dir(self) -> Path:
        return self.analysis_dir / "preview"

    @property
    def proxy_dir(self) -> Path:
        return self.work_dir / "proxy"

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache"

    @property
    def highlights_dir(self) -> Path:
        return self.work_dir / "highlights"

    @property
    def review_dir(self) -> Path:
        return self.work_dir / "review"

    @property
    def logs_dir(self) -> Path:
        return self.work_dir / "logs"


def load_config(path: Path) -> PipelineConfig:
    config_path = path.resolve(strict=True)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    extensions = tuple(str(value).lower() for value in raw.get("video_extensions", [".mp4", ".mkv", ".mov", ".webm"]))
    weights = {str(key): float(value) for key, value in (raw.get("weights") or {}).items()}
    nvenc = dict(raw.get("nvenc") or {})
    audio_mix = dict(raw.get("audio_mix") or {})
    v2_weights = {str(key): float(value) for key, value in (raw.get("v2_weights") or {}).items()}
    beam_weights = {str(key): float(value) for key, value in (raw.get("beam_weights") or {}).items()}
    penalty_weights = {str(key): float(value) for key, value in (raw.get("penalty_weights") or {}).items()}
    target = raw.get("fast_montage_target", "music")
    if not isinstance(target, (str, int, float)):
        target = "music"
    return PipelineConfig(
        config_path=config_path,
        raw_dir=_path(raw["raw_dir"]),
        work_dir=_path(raw["work_dir"]),
        output_dir=_path(raw["output_dir"]),
        music_file=_path(raw["music_file"]),
        video_extensions=extensions,
        proxy_resolution=_pair(raw.get("proxy_resolution"), (960, 540)),
        short_clip_threshold=float(raw.get("short_clip_threshold", 90.0)),
        long_clip_threshold=float(raw.get("long_clip_threshold", 300.0)),
        pre_roll=float(raw.get("pre_roll", 4.0)),
        post_roll=float(raw.get("post_roll", 6.0)),
        analysis_fps=float(raw.get("analysis_fps", 2.0)),
        analysis_sample_rate=int(raw.get("analysis_sample_rate", 22050)),
        highlight_min_duration=float(raw.get("highlight_min_duration", 3.0)),
        highlight_max_duration=float(raw.get("highlight_max_duration", 30.0)),
        activity_threshold_percentile=float(raw.get("activity_threshold_percentile", 75.0)),
        activity_merge_gap=float(raw.get("activity_merge_gap", 3.0)),
        fingerprint_interval=float(raw.get("fingerprint_interval", 2.0)),
        dedupe_threshold=float(raw.get("dedupe_threshold", 0.78)),
        output_width=int(raw.get("output_width", 1920)),
        output_height=int(raw.get("output_height", 1200)),
        target_fps=float(raw.get("target_fps", 60.0)),
        preview_min_duration=float(raw.get("preview_min_duration", 45.0)),
        preview_max_duration=float(raw.get("preview_max_duration", 60.0)),
        preview_preferred_shot_max_duration=float(raw.get("preview_preferred_shot_max_duration", 20.0)),
        baseline_music_in=float(raw.get("baseline_music_in", 19.0)),
        baseline_music_out=float(raw.get("baseline_music_out", 74.252)),
        v2_output_name=str(raw.get("v2_output_name", "preview_60s_v2.mp4")),
        payoff_analysis_fps=float(raw.get("payoff_analysis_fps", 6.0)),
        event_merge_window_ms=int(raw.get("event_merge_window_ms", 700)),
        strong_anchor_threshold=float(raw.get("strong_anchor_threshold", 0.75)),
        weak_anchor_threshold=float(raw.get("weak_anchor_threshold", 0.55)),
        max_anchor_count_per_candidate=int(raw.get("max_anchor_count_per_candidate", 3)),
        preferred_macro_duration=tuple(float(v) for v in raw.get("preferred_macro_duration", [2.0, 10.0])),
        hero_max_duration=float(raw.get("hero_max_duration", 12.0)),
        beam_width=int(raw.get("beam_width", 16)),
        beam_max_expansions=int(raw.get("beam_max_expansions", 32)),
        recent_source_window=int(raw.get("recent_source_window", 2)),
        recent_environment_window=int(raw.get("recent_environment_window", 2)),
        baseline_music_max_shift=float(raw.get("baseline_music_max_shift", 0.5)),
        audio_overlap_ms=tuple(int(v) for v in raw.get("audio_overlap_ms", [100, 250])),
        impact_tail_max_ms=int(raw.get("impact_tail_max_ms", 400)),
        v2_weights=v2_weights,
        beam_weights=beam_weights,
        penalty_weights=penalty_weights,
        roi_profile=dict(raw.get("roi_profile") or {}),
        weights=weights,
        fast_montage_target=target,
        full_highlight_target=float(raw.get("full_highlight_target", 230.0)),
        nvenc=nvenc,
        audio_mix=audio_mix,
    )


def ensure_runtime_dirs(config: PipelineConfig) -> None:
    for directory in (
        config.work_dir,
        config.output_dir,
        config.analysis_dir,
        config.music_analysis_dir,
        config.music_v2_analysis_dir,
        config.preview_analysis_dir,
        config.preview_v2_analysis_dir,
        config.proxy_dir,
        config.cache_dir,
        config.highlights_dir,
        config.review_dir,
        config.logs_dir,
    ):
        if is_within(directory, config.raw_dir):
            raise ValueError(f"Generated directory cannot be inside raw: {directory}")
        directory.mkdir(parents=True, exist_ok=True)


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def assert_source_read_only(path: Path, raw_dir: Path) -> None:
    resolved_path = path.resolve(strict=False)
    resolved_raw = raw_dir.resolve(strict=False)
    if not path.exists() or path.is_dir() or not is_within(resolved_path, resolved_raw):
        raise ValueError(f"Only an existing file below raw may be used as a source: {path}")
