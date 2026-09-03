from dataclasses import replace
from functools import partial
import json
from pathlib import Path

import numpy as np
import pytest

from montage.config import load_config
from montage.models import PayoffEvent, VideoAnalysis
from montage.payoff_detection import (
    EvidenceSample,
    classify_semantics,
    detect_payoff_events,
    fuse_evidence,
    merge_event_peaks,
    write_payoff_events,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
test_config = partial(load_config, PROJECT_ROOT / "config.yaml")
test_config.__test__ = False


def make_event(event_id: str, source_time: float, confidence: float) -> PayoffEvent:
    return PayoffEvent(
        event_id=event_id,
        type="combat_climax",
        source_time=source_time,
        confidence=confidence,
        strength=confidence,
        semantic_confidence=0.0,
        evidence={"motion_peak": confidence},
        detector_flags=(event_id,),
    )


def test_lone_killfeed_change_is_not_a_kill():
    sample = EvidenceSample(source_time=12.4, killfeed_change=0.95, motion_peak=0.1)
    event = fuse_evidence(sample, test_config())
    assert event.type in {"visual_transient", "combat_climax"}
    assert event.semantic_confidence < test_config().weak_anchor_threshold


def test_independent_evidence_fuses_without_fake_semantics():
    sample = EvidenceSample(source_time=12.43, reward_roi_change=0.81,
                            crosshair_event=0.76, audio_transient=0.91,
                            motion_peak=0.68, killfeed_change=0.44)
    event = fuse_evidence(sample, test_config())
    assert event.confidence >= 0.75
    assert set(event.evidence) >= {"reward_roi_change", "crosshair_event",
                                   "audio_transient", "motion_peak", "killfeed_change"}


def test_low_level_signal_does_not_emit_an_event():
    assert fuse_evidence(EvidenceSample(source_time=12.43, motion_peak=0.05), test_config()) is None


def test_peaks_within_700ms_merge():
    merged = merge_event_peaks([make_event("a", 10.0, 0.72),
                                make_event("b", 10.55, 0.81)], 700)
    assert len(merged) == 1
    assert merged[0].source_time == 10.55
    assert merged[0].merged_peak_count == 2


def test_semantic_kill_requires_two_independent_evidence_families():
    event_type, semantic_confidence = classify_semantics(
        {"killfeed_change": 0.95, "crosshair_event": 0.84, "audio_transient": 0.82},
        test_config(),
    )

    assert event_type == "kill"
    assert semantic_confidence >= test_config().weak_anchor_threshold


def test_roi_change_cannot_claim_vehicle_or_objective_semantics():
    for evidence in (
        {"reward_roi_change": 1.0},
        {"objective_roi_change": 1.0},
        {"damage_border_change": 1.0},
    ):
        event_type, semantic_confidence = classify_semantics(evidence, test_config())
        assert event_type not in {"kill", "multikill", "vehicle_destroy", "objective"}
        assert semantic_confidence < test_config().weak_anchor_threshold


def test_merge_preserves_peak_evidence_flags_inside_one_refractory_window():
    first = PayoffEvent("first", "impact_event", 20.0, 0.65, 0.65, 0.1,
                        {"crosshair_event": 0.65}, ("crosshair",))
    second = PayoffEvent("second", "combat_climax", 20.35, 0.88, 0.83, 0.2,
                         {"audio_transient": 0.88}, ("impact",))
    third = PayoffEvent("third", "danger_climax", 20.7, 0.70, 0.72, 0.1,
                        {"damage_border_change": 0.70}, ("danger",))

    merged = merge_event_peaks([third, first, second], 700)

    assert len(merged) == 1
    assert merged[0].source_time == 20.35
    assert merged[0].merged_peak_count == 3
    assert merged[0].evidence == {
        "crosshair_event": 0.65,
        "audio_transient": 0.88,
        "damage_border_change": 0.70,
    }
    assert merged[0].detector_flags == ("crosshair", "impact", "danger")


def test_refractory_merge_does_not_chain_beyond_its_first_peak():
    events = [make_event("a", 10.0, 0.72), make_event("b", 10.65, 0.81), make_event("c", 11.3, 0.78)]

    merged = merge_event_peaks(events, 700)

    assert [event.merged_peak_count for event in merged] == [2, 1]


def test_merged_semantics_use_active_config_thresholds():
    events = [
        PayoffEvent("feed", "visual_transient", 10.0, 0.70, 0.70, 0.0, {"killfeed_change": 0.60}),
        PayoffEvent("impact", "impact_event", 10.2, 0.70, 0.70, 0.0, {"crosshair_event": 0.60}),
    ]
    relaxed = replace(test_config(), payoff_evidence_threshold=0.35)
    strict = replace(test_config(), payoff_evidence_threshold=0.70)

    assert merge_event_peaks(events, 700, relaxed)[0].type == "kill"
    strict_event = merge_event_peaks(events, 700, strict)[0]
    assert strict_event.type != "kill"
    assert strict_event.semantic_confidence == 0.0


def test_detector_uses_source_start_for_window_timestamps_and_records_cache(tmp_path, fake_toolchain, monkeypatch):
    raw_dir = tmp_path / "raw"
    source = raw_dir / "window.mp4"
    raw_dir.mkdir()
    source.write_bytes(b"fixture")
    config = replace(
        test_config(),
        raw_dir=raw_dir,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        roi_profile={
            "version": 3,
            "reward_score": [0.0, 0.0, 0.5, 0.5],
            "kill_feed": [0.5, 0.0, 1.0, 0.5],
            "crosshair": [0.25, 0.25, 0.75, 0.75],
            "objective": [0.0, 0.5, 0.5, 1.0],
            "damage_border": [0.0, 0.0, 1.0, 1.0],
        },
    )
    analysis = VideoAnalysis(source, 6.0, [0.0, 0.2], [0.1, 0.9], [0.1, 0.8], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.full((8, 8, 3), 255, dtype=np.uint8)]
    monkeypatch.setattr("montage.payoff_detection._decode_color_frames", lambda *args: frames)
    monkeypatch.setattr("montage.payoff_detection.ocr_available", lambda: False)

    events = detect_payoff_events(
        source, 30.0, 30.5, analysis,
        {"times": [0.2], "onset_strength": [0.9], "rms": [0.8], "impact": [0.7]},
        config, fake_toolchain,
    )

    assert events
    assert all(30.0 <= event.source_time <= 30.5 for event in events)
    cached = list(config.cache_dir.glob("payoff_events_*.json"))
    assert len(cached) == 1
    payload = cached[0].read_text(encoding="utf-8")
    assert '"ocr_available": false' in payload
    assert '"raw_events"' in payload and '"merged_events"' in payload


def test_detector_selects_only_local_peaks_before_merging(tmp_path, fake_toolchain, monkeypatch):
    raw_dir = tmp_path / "raw"
    source = raw_dir / "noisy.mp4"
    raw_dir.mkdir()
    source.write_bytes(b"fixture")
    config = replace(test_config(), raw_dir=raw_dir, work_dir=tmp_path / "work", output_dir=tmp_path / "output")
    analysis = VideoAnalysis(source, 6.0, [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6], [0.55, 0.75, 0.65, 0.55, 0.45], [0.0] * 5, [0.0] * 5, [0.0] * 5, [0.0] * 5)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(5)]
    monkeypatch.setattr("montage.payoff_detection._decode_color_frames", lambda *args: frames)
    monkeypatch.setattr("montage.payoff_detection.ocr_available", lambda: False)

    events = detect_payoff_events(source, 1.0, 1.8, analysis, {"times": []}, config, fake_toolchain)

    assert len(events) == 1
    cache_payload = json.loads(next(config.cache_dir.glob("payoff_events_*.json")).read_text(encoding="utf-8"))
    assert len(cache_payload["data"]["raw_events"]) == 1


def test_long_candidate_refines_only_proxy_selected_source_subranges(tmp_path, fake_toolchain, monkeypatch):
    raw_dir = tmp_path / "raw"
    source = raw_dir / "long-window.mp4"
    proxy = tmp_path / "work" / "proxy" / "candidate.mp4"
    raw_dir.mkdir()
    source.write_bytes(b"fixture")
    config = replace(test_config(), raw_dir=raw_dir, work_dir=tmp_path / "work", output_dir=tmp_path / "output", long_clip_threshold=0.5)
    analysis = VideoAnalysis(source, 6.0, [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    calls = []
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)] * 5 + [np.full((8, 8, 3), 255, dtype=np.uint8)] + [np.full((8, 8, 3), 255, dtype=np.uint8)] * 12
    monkeypatch.setattr("montage.payoff_detection._build_long_window_proxy", lambda *args: proxy, raising=False)
    monkeypatch.setattr("montage.payoff_detection._decode_color_frames", lambda path, *args: calls.append((path, *args[-2:])) or frames)
    monkeypatch.setattr("montage.payoff_detection.ocr_available", lambda: False)

    detect_payoff_events(source, 30.0, 33.0, analysis, {"times": []}, config, fake_toolchain)

    assert calls[0][0] == proxy
    refinement_ranges = [(start, end) for path, start, end in calls[1:] if path == source]
    assert refinement_ranges
    assert all((start, end) != (30.0, 33.0) for start, end in refinement_ranges)
    assert all(30.0 <= start < end <= 33.0 for start, end in refinement_ranges)


def test_write_payoff_events_emits_auditable_json(tmp_path):
    raw_dir = tmp_path / "raw"
    config = replace(test_config(), raw_dir=raw_dir, work_dir=tmp_path / "work", output_dir=tmp_path / "output")
    path = config.work_dir / "analysis" / "payoff_events_v2.json"
    write_payoff_events([make_event("synthetic", 12.43, 0.84)], path, config)

    payload = path.read_text(encoding="utf-8")
    assert '"event_id": "synthetic"' in payload
    assert '"source_time": 12.43' in payload


def test_write_payoff_events_rejects_raw_destination(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config = replace(test_config(), raw_dir=raw_dir, work_dir=tmp_path / "work", output_dir=tmp_path / "output")

    with pytest.raises(ValueError, match="work"):
        write_payoff_events([make_event("synthetic", 12.43, 0.84)], raw_dir / "payoff_events_v2.json", config)
