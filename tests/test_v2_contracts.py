from pathlib import Path

import pytest

from dataclasses import replace

from montage.cache import assert_baseline_unchanged, baseline_manifest, v2_cache_key
from montage.models import CandidateVariant, PayoffEvent, SourceSegment, V2EditShot


def make_event(event_id: str, source_time: float) -> PayoffEvent:
    return PayoffEvent(
        event_id=event_id,
        type="combat_climax",
        source_time=source_time,
        confidence=0.84,
        strength=0.88,
        semantic_confidence=0.41,
        evidence={"motion_peak": 0.68},
    )


def make_v2_shot(tmp_path: Path, event: PayoffEvent) -> V2EditShot:
    source = tmp_path / "clip.mp4"
    segment = SourceSegment(source, 1.0, 3.0, 2.0)
    return V2EditShot(
        source=source,
        source_in=1.0,
        source_out=3.0,
        duration=2.0,
        candidate_score=0.8,
        duplicate_group=None,
        timeline_in=0.0,
        timeline_out=2.0,
        transition="hard_cut",
        music_target=None,
        music_event_type=None,
        sync_offset=0.0,
        rationale="test",
        source_segments=(segment,),
        parent_candidate_id="candidate-1",
        variant_id="variant-1",
        payoff_events=(event,),
        anchor_event_time=event.source_time,
        anchor_event_type=event.type,
        anchor_event_strength=event.strength,
        anchor_event_confidence=event.confidence,
        primary_anchor=event,
        secondary_anchors=(),
        context_integrity_score=1.0,
        condense_reason="",
        event_timeline=1.0,
        event_sync_offset=0.0,
        cut_sync_offset=0.0,
        transition_compatibility_score=1.0,
        impact_cut=False,
        audio_j_cut_ms=0,
        audio_l_cut_ms=0,
    )


def test_v2_defaults(base_config):
    assert base_config.event_merge_window_ms == 700
    assert base_config.strong_anchor_threshold == 0.75
    assert base_config.weak_anchor_threshold == 0.55
    assert base_config.beam_width == 16
    assert (base_config.baseline_music_in, base_config.baseline_music_out) == (19.0, 74.252)


def test_v2_shot_contains_source_segments_and_sync_fields(tmp_path):
    shot = make_v2_shot(tmp_path, make_event("e1", 2.0))
    payload = shot.to_dict()
    assert {"source_segments", "primary_anchor", "context_integrity_score",
            "condense_reason", "event_timeline", "event_sync_offset",
            "cut_sync_offset", "transition_compatibility_score",
            "audio_j_cut_ms", "audio_l_cut_ms"} <= set(payload)


def test_baseline_guard_detects_change(tmp_path):
    path = tmp_path / "preview_60s.mp4"
    path.write_bytes(b"baseline")
    before = baseline_manifest(path)
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="baseline"):
        assert_baseline_unchanged(before, path)


def test_v2_output_rejects_baseline_collision_and_escape(base_config):
    with pytest.raises(ValueError, match="baseline"):
        replace(base_config, v2_output_name="preview_60s.mp4").v2_output_path
    with pytest.raises(ValueError, match="output_dir"):
        replace(base_config, v2_output_name="../outside.mp4").v2_output_path


def test_v2_cache_key_requires_recorded_ffmpeg_version():
    with pytest.raises(ValueError, match="FFmpeg version"):
        v2_cache_key({"absolute_path": "C:/clip.mp4"}, "payoff_events", {})


def test_v2_contract_mappings_are_immutable(tmp_path):
    event = make_event("e1", 2.0)
    with pytest.raises(TypeError):
        event.evidence["motion_peak"] = 0.1

    segment = SourceSegment(tmp_path / "clip.mp4", 1.0, 3.0, 2.0)
    variant = CandidateVariant(
        variant_id="variant-1", parent_candidate_id="candidate-1", source_file=segment.source,
        source_segments=(segment,), duration=2.0, human_selection_prior=0.8, payoff_score=0.8,
        combat_intensity=0.8, action_density=0.8, continuity=0.8, visual_novelty=0.8,
        motion=0.8, audio_activity=0.8, danger_score=0.8, uniqueness=1.0, final_score=0.8,
        duplicate_group=None, payoff_events=(event,), primary_anchor=event, secondary_anchors=(),
        anchor_event_time=2.0, anchor_event_type="combat_climax", anchor_event_strength=0.88,
        anchor_event_confidence=0.84, context_integrity_score=1.0,
        penalty_values={"downtime": 0.0}, source_signature="source", environment_signature="env",
        weapon_or_view_signature="view", condense_reason="", rationale="test",
    )
    with pytest.raises(TypeError):
        variant.penalty_values["downtime"] = 1.0
    assert variant.to_dict()["penalty_values"] == {"downtime": 0.0}
