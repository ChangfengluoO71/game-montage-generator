"""Configuration loading and runtime directory policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    return int(value[0]), int(value[1])


def _finite_float(value: Any, default: float, minimum: float = 0.0, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    parsed = max(minimum, parsed)
    return min(maximum, parsed) if maximum is not None else parsed


def _excluded_ranges(value: Any) -> tuple[tuple[str, float, float, str], ...]:
    """Parse explicit filename/time editorial exclusions without touching sources."""
    if not isinstance(value, (list, tuple)):
        return ()
    parsed: list[tuple[str, float, float, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source_name = str(item.get("source_name", "")).strip()
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if not source_name or not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start:
            continue
        reason = str(item.get("reason", "editorial_exclusion")).strip() or "editorial_exclusion"
        parsed.append((source_name, start, end, reason))
    return tuple(parsed)


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
    payoff_detector_version: str
    payoff_evidence_threshold: float
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
    minimum_context_integrity: float
    rapid_multikill_window_s: float
    rapid_multikill_min_events: int
    rapid_multikill_bonus_weight: float
    hero_quality_margin: float
    v2_weights: dict[str, float]
    beam_weights: dict[str, float]
    penalty_weights: dict[str, float]
    roi_profile: dict[str, Any]
    weights: dict[str, float]
    fast_montage_target: str | float
    full_highlight_target: float
    nvenc: dict[str, Any]
    audio_mix: dict[str, Any]
    boundary_window_ms: int = 500
    anchor_sync_tolerance: float = 0.75
    v2_excluded_ranges: tuple[tuple[str, float, float, str], ...] = ()
    v2_min_shots: int = 8
    v2_max_shots: int = 14
    v2_short_clip_max_duration: float = 30.0
    v2_short_clip_prior: float = 0.85
    v2_music_window_policy: str = "baseline"
    v5_profile: str = "off"
    v5_min_kill_semantic_confidence: float = 0.80
    v5_min_killfeed_evidence: float = 0.25
    v5_min_reward_evidence: float = 0.10
    v5_min_verified_kills_per_shot: int = 0
    v5_kill_density_window_s: float = 6.0
    v5_kill_density_target: int = 4
    v5_kill_count_weight: float = 0.0
    v5_kill_density_weight: float = 0.0
    v5_max_sequences_per_candidate: int = 1
    v5_sequence_context_s: float = 1.25
    v5_min_rapid_context_tail_s: float = 0.25
    v6_profile_id: str = "cfl_bf6_1920x1200_v1"
    v6_detector_version: str = "v6-skull-row-5-calibrated-y-gate"
    v6_coarse_fps: float = 12.0
    v6_dense_fps: float = 60.0
    v6_panel_disappear_s: float = 0.75
    v6_refinement_radius_s: float = 1.0
    v6_initial_search_roi: tuple[float, float, float, float] = (0.25, 0.42, 0.66, 0.75)

    @property
    def analysis_dir(self) -> Path:
        return self.work_dir / "analysis"

    @property
    def music_analysis_dir(self) -> Path:
        return self.analysis_dir / "music"

    @property
    def music_v2_analysis_dir(self) -> Path:
        return self.analysis_dir / ("music_v5" if self.v5_profile == "quality" else "music_v2")

    @property
    def v2_output_path(self) -> Path:
        output = (self.output_dir / self.v2_output_name).resolve(strict=False)
        output_root = self.output_dir.resolve(strict=False)
        if output == self.baseline_output_path.resolve(strict=False):
            raise ValueError("V2 output cannot collide with the immutable baseline")
        if output == output_root:
            raise ValueError("V2 output must be a file destination beneath output_dir")
        if not is_within(output, output_root):
            raise ValueError(f"V2 output must remain beneath output_dir: {output}")
        return output

    @property
    def baseline_output_path(self) -> Path:
        return self.output_dir / "preview_60s.mp4"

    @property
    def music_v2_beat_map_path(self) -> Path:
        return self.music_v2_analysis_dir / ("beat_map_v5.json" if self.v5_profile == "quality" else "beat_map_v2.json")

    @property
    def music_v2_structure_path(self) -> Path:
        return self.music_v2_analysis_dir / ("music_structure_v5.json" if self.v5_profile == "quality" else "music_structure_v2.json")

    @property
    def music_v2_energy_curve_path(self) -> Path:
        return self.music_v2_analysis_dir / ("energy_curve_v5.csv" if self.v5_profile == "quality" else "energy_curve_v2.csv")

    @property
    def music_v2_analysis_image_path(self) -> Path:
        return self.music_v2_analysis_dir / ("music_analysis_v5.png" if self.v5_profile == "quality" else "music_analysis_v2.png")

    @property
    def music_v2_cache_path(self) -> Path:
        return self.music_v2_analysis_dir / ("music_analysis_v5_cache.json" if self.v5_profile == "quality" else "music_analysis_v2_cache.json")

    @property
    def payoff_events_v2_path(self) -> Path:
        return self.analysis_dir / ("payoff_events_v5.json" if self.v5_profile == "quality" else "payoff_events_v2.json")

    @property
    def highlight_candidates_v2_path(self) -> Path:
        return self.analysis_dir / ("highlight_candidates_v5.json" if self.v5_profile == "quality" else "highlight_candidates_v2.json")

    @property
    def highlight_candidates_v2_csv_path(self) -> Path:
        return self.analysis_dir / ("highlight_candidates_v5.csv" if self.v5_profile == "quality" else "highlight_candidates_v2.csv")

    @property
    def dedupe_summary_v2_path(self) -> Path:
        return self.analysis_dir / ("dedupe_summary_v5.json" if self.v5_profile == "quality" else "dedupe_summary_v2.json")

    @property
    def preview_v2_analysis_dir(self) -> Path:
        return self.analysis_dir / ("preview_v5" if self.v5_profile == "quality" else "preview")

    @property
    def preview_v2_edit_path(self) -> Path:
        return self.preview_v2_analysis_dir / ("preview_v5_edit.json" if self.v5_profile == "quality" else "preview_v2_edit.json")

    @property
    def preview_v2_timeline_path(self) -> Path:
        return self.preview_v2_analysis_dir / ("preview_v5_timeline.txt" if self.v5_profile == "quality" else "preview_v2_timeline.txt")

    @property
    def preview_v2_sync_report_path(self) -> Path:
        return self.preview_v2_analysis_dir / ("preview_v5_sync_report.json" if self.v5_profile == "quality" else "preview_v2_sync_report.json")

    @property
    def preview_v2_report_path(self) -> Path:
        return self.preview_v2_analysis_dir / ("preview_v5_report.md" if self.v5_profile == "quality" else "preview_v2_report.md")

    @property
    def preview_v2_timeline_image_path(self) -> Path:
        return self.preview_v2_analysis_dir / ("preview_v5_timeline.png" if self.v5_profile == "quality" else "preview_v2_timeline.png")

    @property
    def environment_v2_path(self) -> Path:
        return self.analysis_dir / ("environment_v5.json" if self.v5_profile == "quality" else "environment_v2.json")

    @property
    def baseline_manifest_v2_path(self) -> Path:
        return self.analysis_dir / ("baseline_manifest_v5.json" if self.v5_profile == "quality" else "baseline_manifest_v2.json")

    @property
    def raw_manifest_v2_before_path(self) -> Path:
        return self.analysis_dir / ("raw_manifest_v5_before.json" if self.v5_profile == "quality" else "raw_manifest_v2_before.json")

    @property
    def raw_manifest_v2_after_path(self) -> Path:
        return self.analysis_dir / ("raw_manifest_v5_after.json" if self.v5_profile == "quality" else "raw_manifest_v2_after.json")

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
    def render_v2_dir(self) -> Path:
        return self.cache_dir / ("render_v5" if self.v5_profile == "quality" else "render_v2")

    @property
    def highlights_dir(self) -> Path:
        return self.work_dir / "highlights"

    @property
    def review_dir(self) -> Path:
        return self.work_dir / "review"

    @property
    def logs_dir(self) -> Path:
        return self.work_dir / "logs"

    @property
    def v6_kill_truth_dir(self) -> Path:
        return self.analysis_dir / "v6_kill_truth"

    @property
    def v6_profiles_dir(self) -> Path:
        return self.v6_kill_truth_dir / "profiles"

    @property
    def v6_events_dir(self) -> Path:
        return self.v6_kill_truth_dir / "events"

    @property
    def v6_sequences_dir(self) -> Path:
        return self.v6_kill_truth_dir / "sequences"

    @property
    def v6_review_dir(self) -> Path:
        return self.v6_kill_truth_dir / "review"

    @property
    def v6_debug_dir(self) -> Path:
        return self.v6_kill_truth_dir / "debug"

    @property
    def v6_cache_dir(self) -> Path:
        return self.v6_kill_truth_dir / "cache"

    @property
    def v6_reports_dir(self) -> Path:
        return self.v6_kill_truth_dir / "reports"

    @property
    def v6_event_index_path(self) -> Path:
        return self.v6_kill_truth_dir / "kill_event_index_v6.json"

    @property
    def v6_sequence_index_path(self) -> Path:
        return self.v6_kill_truth_dir / "kill_sequence_index_v6.json"

    @property
    def v6_review_html_path(self) -> Path:
        return self.v6_review_dir / "kill_review.html"

    @property
    def v6_calibration_report_path(self) -> Path:
        return self.v6_reports_dir / "v6_calibration_report.md"

    @property
    def v6_report_json_path(self) -> Path:
        return self.v6_reports_dir / "v6_report.json"

    @property
    def v6_report_md_path(self) -> Path:
        return self.v6_reports_dir / "v6_report.md"

    @property
    def v6_environment_path(self) -> Path:
        return self.v6_kill_truth_dir / "environment.json"

    @property
    def v6_gold_set_path(self) -> Path:
        # The handoff contract keeps the independently authored gold set at
        # work/analysis/v6_kill_truth/v6_gold_set.json, not inside reports.
        return self.v6_kill_truth_dir / "v6_gold_set.json"

    @property
    def curation_dir(self) -> Path:
        return self.analysis_dir / "curation"

    @property
    def curation_ledger_path(self) -> Path:
        return self.curation_dir / "curation_ledger.json"


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
    v2_max_shots = max(1, int(raw.get("v2_max_shots", 14)))
    v2_min_shots = max(1, min(v2_max_shots, int(raw.get("v2_min_shots", 8))))
    v2_short_clip_max_duration = max(1.0, float(raw.get("v2_short_clip_max_duration", 30.0)))
    v2_short_clip_prior = max(0.0, min(1.0, float(raw.get("v2_short_clip_prior", 0.85))))
    v5_profile = str(raw.get("v5_profile", "off")).strip().lower()
    if v5_profile not in {"off", "quality"}:
        v5_profile = "off"
    v5_min_kill_semantic_confidence = _finite_float(raw.get("v5_min_kill_semantic_confidence", 0.80), 0.80, 0.0, 1.0)
    v5_min_killfeed_evidence = _finite_float(raw.get("v5_min_killfeed_evidence", 0.25), 0.25, 0.0, 1.0)
    v5_min_reward_evidence = _finite_float(raw.get("v5_min_reward_evidence", 0.10), 0.10, 0.0, 1.0)
    v5_min_verified_kills_per_shot = int(_finite_float(
        raw.get("v5_min_verified_kills_per_shot", 0), 0.0, 0.0, 100.0,
    ))
    v5_kill_density_window_s = _finite_float(raw.get("v5_kill_density_window_s", 6.0), 6.0, 0.25, 60.0)
    v5_kill_density_target = max(1, int(_finite_float(
        raw.get("v5_kill_density_target", 4), 4.0, 1.0, 100.0,
    )))
    v5_max_sequences_per_candidate = max(1, int(_finite_float(
        raw.get("v5_max_sequences_per_candidate", 1), 1.0, 1.0, 100.0,
    )))
    v5_sequence_context_s = _finite_float(raw.get("v5_sequence_context_s", 1.25), 1.25, 0.1, 10.0)
    v5_min_rapid_context_tail_s = _finite_float(raw.get("v5_min_rapid_context_tail_s", 0.25), 0.25, 0.0, 5.0)
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
        payoff_detector_version=str(raw.get("payoff_detector_version", "payoff-v2")),
        payoff_evidence_threshold=float(raw.get("payoff_evidence_threshold", 0.35)),
        event_merge_window_ms=int(raw.get("event_merge_window_ms", 700)),
        strong_anchor_threshold=float(raw.get("strong_anchor_threshold", 0.75)),
        weak_anchor_threshold=float(raw.get("weak_anchor_threshold", 0.55)),
        max_anchor_count_per_candidate=int(raw.get("max_anchor_count_per_candidate", 3)),
        preferred_macro_duration=tuple(float(v) for v in raw.get("preferred_macro_duration", [2.0, 10.0])),
        hero_max_duration=float(raw.get("hero_max_duration", 12.0)),
        beam_width=int(raw.get("beam_width", 16)),
        beam_max_expansions=int(raw.get("beam_max_expansions", 8192)),
        recent_source_window=int(raw.get("recent_source_window", 2)),
        recent_environment_window=int(raw.get("recent_environment_window", 2)),
        baseline_music_max_shift=float(raw.get("baseline_music_max_shift", 0.5)),
        audio_overlap_ms=tuple(int(v) for v in raw.get("audio_overlap_ms", [100, 250])),
        impact_tail_max_ms=int(raw.get("impact_tail_max_ms", 400)),
        minimum_context_integrity=float(raw.get("minimum_context_integrity", 0.70)),
        rapid_multikill_window_s=float(raw.get("rapid_multikill_window_s", 4.0)),
        rapid_multikill_min_events=int(raw.get("rapid_multikill_min_events", 2)),
        rapid_multikill_bonus_weight=float(raw.get("rapid_multikill_bonus_weight", 0.12)),
        hero_quality_margin=float(raw.get("hero_quality_margin", 0.05)),
        v2_weights=v2_weights,
        beam_weights=beam_weights,
        penalty_weights=penalty_weights,
        roi_profile=dict(raw.get("roi_profile") or {}),
        weights=weights,
        fast_montage_target=target,
        full_highlight_target=float(raw.get("full_highlight_target", 230.0)),
        nvenc=nvenc,
        audio_mix=audio_mix,
        boundary_window_ms=int(raw.get("boundary_window_ms", 500)),
        anchor_sync_tolerance=float(raw.get("anchor_sync_tolerance", 0.75)),
        v2_excluded_ranges=_excluded_ranges(raw.get("v2_excluded_ranges", [])),
        v2_min_shots=v2_min_shots,
        v2_max_shots=v2_max_shots,
        v2_short_clip_max_duration=v2_short_clip_max_duration,
        v2_short_clip_prior=v2_short_clip_prior,
        v2_music_window_policy=str(raw.get("v2_music_window_policy", "baseline")).strip().lower() or "baseline",
        v5_profile=v5_profile,
        v5_min_kill_semantic_confidence=v5_min_kill_semantic_confidence,
        v5_min_killfeed_evidence=v5_min_killfeed_evidence,
        v5_min_reward_evidence=v5_min_reward_evidence,
        v5_min_verified_kills_per_shot=v5_min_verified_kills_per_shot,
        v5_kill_density_window_s=v5_kill_density_window_s,
        v5_kill_density_target=v5_kill_density_target,
        v5_kill_count_weight=_finite_float(raw.get("v5_kill_count_weight", 0.0), 0.0, 0.0, 1.0),
        v5_kill_density_weight=_finite_float(raw.get("v5_kill_density_weight", 0.0), 0.0, 0.0, 1.0),
        v5_max_sequences_per_candidate=v5_max_sequences_per_candidate,
        v5_sequence_context_s=v5_sequence_context_s,
        v5_min_rapid_context_tail_s=v5_min_rapid_context_tail_s,
        v6_profile_id=str(raw.get("v6_profile_id", "cfl_bf6_1920x1200_v1")),
        v6_detector_version=str(raw.get("v6_detector_version", "v6-skull-row-5-calibrated-y-gate")),
        v6_coarse_fps=_finite_float(raw.get("v6_coarse_fps", 12.0), 12.0, 1.0, 30.0),
        v6_dense_fps=_finite_float(raw.get("v6_dense_fps", 60.0), 60.0, 1.0, 120.0),
        v6_panel_disappear_s=_finite_float(raw.get("v6_panel_disappear_s", 0.75), 0.75, 0.1, 5.0),
        v6_refinement_radius_s=_finite_float(raw.get("v6_refinement_radius_s", 1.0), 1.0, 0.1, 5.0),
        v6_initial_search_roi=tuple(float(item) for item in raw.get("v6_initial_search_roi", [0.25, 0.42, 0.66, 0.75])),
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
        config.v6_kill_truth_dir,
        config.v6_profiles_dir,
        config.v6_events_dir,
        config.v6_sequences_dir,
        config.v6_review_dir,
        config.v6_debug_dir,
        config.v6_cache_dir,
        config.v6_reports_dir,
        config.curation_dir,
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
