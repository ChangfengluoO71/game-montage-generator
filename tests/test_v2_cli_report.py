from pathlib import Path

import main
from montage.models import (
    EditDecisionList,
    EditShot,
    MusicAnalysis,
    PayoffEvent,
    SourceSegment,
    V2EditDecisionList,
    V2EditShot,
)
from montage.timeline_visualization import render_v2_timeline_plot
from montage.v2_report import build_v2_sync_report


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
    assert "event_sync_P95" in report
    assert "cut_sync_P95" in report
    assert "transition_compatibility_mean" in report
    assert "rejected_by_stationary_ads" in report
    assert "rapid_multikill" in report
    assert report["diagnostic_evidence"] is True


def test_plot_is_created_below_work(tmp_path):
    path = tmp_path / "preview_v2_timeline.png"
    render_v2_timeline_plot(_v2_edit(tmp_path), _music(), path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_baseline_edit_shot_shape_remains_supported():
    shot = EditShot(Path("clip.mp4"), 0.0, 4.0, 4.0, 0.8, None, 0.0, 4.0, "hard_cut", 20.0, "beat", 0.0, "ok")
    assert _baseline().shots == []
    assert shot.duration == 4.0
