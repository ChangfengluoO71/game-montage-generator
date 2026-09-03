import json
from dataclasses import replace
from pathlib import Path

import pytest

import main
from montage.models import (
    Candidate,
    CandidateVariant,
    EditDecisionList,
    EditShot,
    MusicAnalysis,
    PayoffEvent,
    SourceSegment,
    V2EditDecisionList,
    V2EditShot,
)
from montage.timeline_visualization import render_v2_timeline_plot
from montage.v2_report import build_v2_sync_report, render_v2_markdown_report


def _event(event_id: str, event_type: str, source_time: float, confidence: float = 0.9) -> PayoffEvent:
    return PayoffEvent(event_id, event_type, source_time, confidence, confidence, confidence, {"motion": confidence})


def _v2_edit(tmp_path: Path) -> V2EditDecisionList:
    source = tmp_path / "raw" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    event = _event("hero", "hero_play", 8.0)
    segment = SourceSegment(source, 0.0, 4.0, 4.0)
    shot = V2EditShot(
        source, 0.0, 4.0, 4.0, 0.9, "group-1", 0.0, 4.0, "hard_cut", 20.0,
        "strong_beat", 0.1, "setup action payoff", source_segments=(segment,),
        variant_id="variant-1", parent_candidate_id="candidate-1", payoff_events=(event,),
        primary_anchor=event, anchor_event_time=8.0, anchor_event_type="hero_play",
        anchor_event_strength=0.9, anchor_event_confidence=0.9,
        event_timeline=3.0, event_sync_offset=0.1, cut_sync_offset=0.2,
        transition_compatibility_score=0.8,
    )
    return V2EditDecisionList(
        "preview_v2", tmp_path / "music.wav", 19.0, 74.252, 19.0, 74.252,
        48.0, "baseline locked", (shot,),
    )


def _baseline() -> EditDecisionList:
    return EditDecisionList("preview", Path("music.wav"), 19.0, 74.252, 55.252, "locked", [])


def _unscored_rapid_variant(tmp_path: Path) -> CandidateVariant:
    source = tmp_path / "raw" / "rapid.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.touch()
    events = (
        _event("kill-1", "kill", 2.0, 0.9),
        _event("kill-2", "kill", 5.0, 0.9),
    )
    segment = SourceSegment(source, 0.0, 6.0, 6.0)
    return CandidateVariant(
        variant_id="rapid-variant", parent_candidate_id="candidate-1", source_file=source,
        source_segments=(segment,), duration=6.0, human_selection_prior=0.8, payoff_score=0.9,
        combat_intensity=0.8, action_density=0.8, continuity=0.9, visual_novelty=0.7,
        motion=0.8, audio_activity=0.8, danger_score=0.7, uniqueness=0.9, final_score=0.0,
        duplicate_group="rapid-group", payoff_events=events, primary_anchor=events[-1],
        secondary_anchors=(events[0],), anchor_event_time=events[-1].source_time,
        anchor_event_type=events[-1].type, anchor_event_strength=events[-1].strength,
        anchor_event_confidence=events[-1].confidence, context_integrity_score=1.0,
        penalty_values={}, source_signature="rapid-source", environment_signature="rapid-map",
        weapon_or_view_signature="rifle", condense_reason="",
        rationale="rapid consecutive kills retain the complete action",
    )


def _scored_cache_row(config) -> dict[str, object]:
    components = {name: 0.0 for name in config.v2_weights}
    components.update({"rapid_multikill_score": 0.0, "rapid_multikill_bonus": 0.0})
    return {
        "variant_id": "cached", "final_score": 0.5, "score_components": components,
        "rapid_multikill_score": 0.0, "rapid_multikill_bonus": 0.0,
    }


def _music() -> MusicAnalysis:
    return MusicAnalysis(
        Path("music.wav"), 100.0, 120.0, [20.0, 21.0, 22.0], [20.0], [20.0], [20.5],
        [], [20.0, 21.0, 22.0], [0.2, 0.5, 0.8], [0.1, 0.4, 0.9], [0.2, 0.5, 0.8],
        [{"start": 20.0, "end": 30.0, "label": "build"}], {"beat": 0.9, "phrase": 0.8},
    )


def test_v2_commands_are_exposed():
    parser = main.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert {"all-v2", "render-preview-v2", "verify-preview-v2"} <= set(choices)


def test_report_contains_baseline_and_event_metrics(tmp_path):
    report = build_v2_sync_report(_v2_edit(tmp_path), _baseline(), [], {"stationary_ads": 2})
    assert report["baseline_music_range"] == [19.0, 74.252]
    assert report["v2_music_range"] == [19.0, 74.252]
    assert report["music_in"] == 19.0
    assert report["music_out"] == 74.252
    assert report["music_reason"] == "baseline locked"
    assert "event_sync_P95" in report
    assert "cut_sync_P95" in report
    assert "transition_compatibility_mean" in report
    assert "rejected_by_stationary_ads" in report
    assert "rapid_multikill" in report
    assert report["diagnostic_evidence"] is True
    assert report["comparisons"]["macro_shot_count"] == {"v1": 0, "v2": 1, "delta": 1}
    assert report["comparisons"]["event_sync_P95"]["v1"] == 0.0
    assert "v1_macro_shot_count" in report
    assert "delta_macro_shot_count" in report
    assert report["comparisons"]["music_range"]["delta"] == [0.0, 0.0]
    assert report["comparisons"]["rejected_by_stationary_ads"]["v1"] is None
    assert report["comparisons"]["rejected_by_stationary_ads"]["v1_available"] is False
    assert report["comparisons"]["rejected_by_stationary_ads"]["v1_reason"]

    markdown_path = tmp_path / "preview_v2_report.md"
    render_v2_markdown_report(report, markdown_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Music interval" in markdown
    assert "- music_in: 19.000s" in markdown
    assert "- music_out: 74.252s" in markdown
    assert "- reason: baseline locked" in markdown


def test_report_distinguishes_unavailable_v1_penalties_from_zero(tmp_path):
    report = build_v2_sync_report(_v2_edit(tmp_path), _baseline(), [], {}, lookback_window=2)
    pair = report["comparisons"]["stationary_ads_duration_ratio"]
    assert pair["v1"] is None
    assert pair["v1_available"] is False
    assert pair["v1_reason"] == "V1 penalty artifacts are unavailable"
    assert pair["v2"] == 0.0
    assert pair["delta"] is None


def test_stale_event_artifact_is_not_a_valid_variant_cache_hit(tmp_path):
    config = replace(main.load_config(Path(__file__).parents[1] / "config.yaml"), work_dir=tmp_path)
    config.analysis_dir.mkdir(parents=True, exist_ok=True)
    stale = _event("stale", "kill", 1.0)
    config.payoff_events_v2_path.write_text(
        json.dumps({"cache_key": "old-inputs", "events": [stale.to_dict()], "event_count": 1}),
        encoding="utf-8",
    )
    config.highlight_candidates_v2_path.write_text(
        json.dumps({"cache_key": "old-inputs", "candidates": []}), encoding="utf-8"
    )
    assert main._v2_artifact_cache_hit(config, "new-inputs") is False


def test_current_key_unscored_v2_artifact_is_not_a_cache_hit(tmp_path, fake_toolchain):
    config = replace(main.load_config(Path(__file__).parents[1] / "config.yaml"), work_dir=tmp_path)
    config.analysis_dir.mkdir(parents=True, exist_ok=True)
    key = main._v2_artifact_cache_key(config, fake_toolchain)
    config.highlight_candidates_v2_path.write_text(
        json.dumps({
            "cache_key": key,
            "candidates": [{
                "variant_id": "stale-zero-score",
                "final_score": 0.0,
                "score_components": {},
                "rapid_multikill_score": 0.0,
                "rapid_multikill_bonus": 0.0,
            }],
        }),
        encoding="utf-8",
    )
    config.payoff_events_v2_path.write_text(
        json.dumps({"cache_key": key, "events": [], "event_count": 0}), encoding="utf-8"
    )
    config.dedupe_summary_v2_path.write_text(json.dumps({"groups": []}), encoding="utf-8")

    assert main._v2_artifact_cache_hit(config, key) is False


def test_detector_input_changes_invalidate_cached_v2_events_and_variants(tmp_path, fake_toolchain):
    base = replace(main.load_config(Path(__file__).parents[1] / "config.yaml"), work_dir=tmp_path)
    base.analysis_dir.mkdir(parents=True, exist_ok=True)
    base_key = main._v2_artifact_cache_key(base, fake_toolchain)
    base.highlight_candidates_v2_path.write_text(
        json.dumps({"cache_key": base_key, "candidates": [_scored_cache_row(base)]}), encoding="utf-8"
    )
    base.payoff_events_v2_path.write_text(
        json.dumps({"cache_key": base_key, "events": [], "event_count": 0}), encoding="utf-8"
    )
    base.dedupe_summary_v2_path.write_text(json.dumps({"groups": []}), encoding="utf-8")

    assert main._v2_artifact_cache_hit(base, base_key) is True
    mutations = (
        replace(base, roi_profile={**base.roi_profile, "kill_feed": [0.67, 0.03, 0.99, 0.35]}),
        replace(base, payoff_evidence_threshold=base.payoff_evidence_threshold + 0.01),
        replace(base, long_clip_threshold=base.long_clip_threshold + 1.0),
    )
    assert all(main._v2_artifact_cache_key(config, fake_toolchain) != base_key for config in mutations)
    assert all(not main._v2_artifact_cache_hit(config, main._v2_artifact_cache_key(config, fake_toolchain)) for config in mutations)


def test_v2_orchestration_scores_condensed_variants_and_rapid_multikill(tmp_path):
    config = replace(
        main.load_config(Path(__file__).parents[1] / "config.yaml"),
        raw_dir=tmp_path / "raw", work_dir=tmp_path / "work", output_dir=tmp_path / "output",
    )

    scored = main._score_v2_variants([_unscored_rapid_variant(tmp_path)], config)[0]

    assert scored.final_score > 0.0
    assert scored.score_components
    assert scored.rapid_multikill_score == 1.0
    assert scored.rapid_multikill_bonus == config.rapid_multikill_bonus_weight
    assert scored.score_components["rapid_multikill_bonus"] == config.rapid_multikill_bonus_weight


def test_plot_is_created_below_work(tmp_path):
    path = tmp_path / "preview_v2_timeline.png"
    render_v2_timeline_plot(_v2_edit(tmp_path), _music(), path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_normalizes_music_source_times_to_edit_time(tmp_path, monkeypatch):
    edit = _v2_edit(tmp_path)
    captured = []

    def record_markers(values, label, color, axis):
        captured.append((label, list(values)))

    monkeypatch.setattr("montage.timeline_visualization._markers", record_markers)
    render_v2_timeline_plot(edit, _music(), tmp_path / "normalized.png")
    assert next(values for label, values in captured if label == "beat") == [1.0, 2.0, 3.0]


def test_baseline_edit_shot_shape_remains_supported():
    shot = EditShot(Path("clip.mp4"), 0.0, 4.0, 4.0, 0.8, None, 0.0, 4.0, "hard_cut", 20.0, "beat", 0.0, "ok")
    assert _baseline().shots == []
    assert shot.duration == 4.0


def test_cached_payoff_events_are_loaded_without_being_replaced(tmp_path):
    config = main.load_config(Path(__file__).parents[1] / "config.yaml")
    event = _event("cached", "multikill", 4.0)
    config = replace(config, work_dir=tmp_path, raw_dir=tmp_path / "raw")
    config.analysis_dir.mkdir(parents=True, exist_ok=True)
    path = config.payoff_events_v2_path
    path.write_text(json.dumps({"events": [event.to_dict()], "event_count": 1}), encoding="utf-8")
    loaded = main._load_cached_payoff_events(config)
    assert loaded[0].event_id == "cached"
    assert path.read_text(encoding="utf-8").count("cached") == 1


def test_report_counts_same_source_penalties_across_configured_lookback(tmp_path):
    edit = _v2_edit(tmp_path)
    shots = tuple(replace(shot, source_signature="same" if index in {0, 2} else f"other-{index}")
                  for index, shot in enumerate((edit.shots[0], edit.shots[0], edit.shots[0])))
    edit = replace(edit, shots=shots, duration=12.0)
    report = build_v2_sync_report(edit, _baseline(), [], {}, lookback_window=2)
    assert report["same_source_recent_penalty_count"] == 1


def test_cached_audio_helper_returns_per_source_evidence(tmp_path, monkeypatch, fake_toolchain):
    config = replace(main.load_config(Path(__file__).parents[1] / "config.yaml"),
                     work_dir=tmp_path, raw_dir=tmp_path / "raw")
    source = config.raw_dir / "source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    candidate = Candidate("c1", source, 0.0, 4.0, 4.0, 0.5, 0.5, 0.5, 0.5, 0.5)
    monkeypatch.setattr(main, "extract_analysis_audio", lambda *args, **kwargs: args[1])
    monkeypatch.setattr(main.sf, "read", lambda *args, **kwargs: ([0.1, 0.2], 22050))
    monkeypatch.setattr(main, "analyze_audio_waveform", lambda *args: {"times": [0.0], "onset_strength": [0.9]})
    evidence = main._cached_audio_evidence(config, candidate, [], fake_toolchain)
    assert evidence["onset_strength"] == [0.9]


def test_v2_pipeline_logs_selected_toolchain_details_before_processing(tmp_path, monkeypatch, fake_toolchain):
    config = replace(
        main.load_config(Path(__file__).parents[1] / "config.yaml"),
        raw_dir=tmp_path / "raw", work_dir=tmp_path / "work", output_dir=tmp_path / "output",
    )

    class RecordingLogger:
        def __init__(self):
            self.messages = []

        def info(self, message, *args):
            self.messages.append(message % args if args else message)

    logger = RecordingLogger()
    monkeypatch.setattr(main, "_logger", lambda _: logger)
    monkeypatch.setattr(main, "discover_toolchain", lambda _: fake_toolchain)
    monkeypatch.setattr(main, "write_environment_report", lambda *args: None)

    with pytest.raises(FileNotFoundError, match="immutable V1 baseline"):
        main.run_v2_analysis_pipeline(config)

    joined = "\n".join(logger.messages)
    assert f"V2 FFmpeg selected path={fake_toolchain.ffmpeg.resolve(strict=False)} version=8.0" in joined
    assert f"V2 ffprobe selected path={fake_toolchain.ffprobe.resolve(strict=False)} version=8.0" in joined
    assert "V2 NVENC runtime h264_nvenc=True hevc_nvenc=True" in joined
