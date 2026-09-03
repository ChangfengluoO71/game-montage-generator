from dataclasses import replace
from pathlib import Path

from montage.models import BoundaryDescriptor, CandidateVariant, MusicAnalysis, PayoffEvent, SourceSegment, VideoAnalysis
from montage.transitions import (
    choose_anchor_music_target,
    choose_v2_transition,
    protect_context_during_sync,
    transition_compatibility_score,
)
from montage.video_analysis import describe_variant_boundary


def event(event_id: str, event_type: str, source_time: float, strength: float = 0.9) -> PayoffEvent:
    return PayoffEvent(event_id, event_type, source_time, 0.9, strength, 0.8, {"impact": strength})


def music() -> MusicAnalysis:
    return MusicAnalysis(
        Path("music.wav"), 30.0, 120.0, [1.0, 2.0, 3.0, 4.0], [2.0, 4.0], [4.0], [1.8, 3.8],
        [{"timestamp": 2.0, "type": "strong_beat"}, {"timestamp": 4.0, "type": "bar"}],
        [], [], [], [], [], {},
    )


def variant(tmp_path: Path, *, anchor: PayoffEvent | None = None, signature: str = "env") -> CandidateVariant:
    source = tmp_path / "clip.mp4"
    segment = SourceSegment(source, 2.0, 10.0, 8.0)
    return CandidateVariant(
        variant_id="v", parent_candidate_id="c", source_file=source, source_segments=(segment,), duration=8.0,
        human_selection_prior=0.8, payoff_score=0.8, combat_intensity=0.8, action_density=0.8,
        continuity=0.8, visual_novelty=0.8, motion=0.8, audio_activity=0.8, danger_score=0.8,
        uniqueness=1.0, final_score=0.8, duplicate_group=None, payoff_events=(anchor,) if anchor else (),
        primary_anchor=anchor, secondary_anchors=(), anchor_event_time=anchor.source_time if anchor else None,
        anchor_event_type=anchor.type if anchor else None, anchor_event_strength=anchor.strength if anchor else None,
        anchor_event_confidence=anchor.confidence if anchor else None, context_integrity_score=1.0,
        penalty_values={}, source_signature="source", environment_signature=signature,
        weapon_or_view_signature="rifle", condense_reason="", rationale="test",
    )


def two_segment_variant(tmp_path: Path, anchor: PayoffEvent) -> CandidateVariant:
    base = variant(tmp_path, anchor=anchor)
    first = SourceSegment(base.source_file, 0.0, 3.0, 3.0)
    second = SourceSegment(base.source_file, 6.0, 10.0, 4.0)
    return replace(base, source_segments=(first, second), duration=7.0, condense_reason="phrase_boundary")


def test_primary_payoff_targets_strong_beat_before_nearby_onset(base_config):
    placement = choose_anchor_music_target(event("p", "kill", 4.0), music(), 3.7, base_config)

    assert placement.target_music_time == 4.0
    assert placement.event_type == "kill"
    assert placement.event_sync_offset == 0.3


def test_primary_anchor_requires_strong_confidence_for_downbeat(base_config):
    weak = replace(event("weak", "kill", 4.0), confidence=0.6, semantic_confidence=0.6)
    placement = choose_anchor_music_target(weak, music(), 3.7, replace(base_config, strong_anchor_threshold=0.8))

    assert placement.target_music_time == 3.8


def test_weak_semantic_event_does_not_force_structural_bar(base_config):
    weak = event("weak", "visual_transient", 4.0, strength=0.6)
    placement = choose_anchor_music_target(weak, music(), 3.7, replace(base_config, strong_anchor_threshold=0.8))

    assert placement.target_music_time == 3.8


def test_context_protection_keeps_both_segments_and_calculates_offset(tmp_path, base_config):
    payoff = event("p", "kill", 7.0)
    placement = protect_context_during_sync(two_segment_variant(tmp_path, payoff), 12.0, base_config)

    assert placement.source_in == 0.0
    assert placement.source_out == 10.0
    assert placement.event_sync_offset == 5.0


def test_weak_context_falls_back_to_original_range(tmp_path, base_config):
    payoff = event("p", "kill", 6.2)
    condensed = replace(two_segment_variant(tmp_path, payoff), context_integrity_score=1.0)
    placement = protect_context_during_sync(condensed, 12.0, base_config)

    assert placement.source_in == 0.0
    assert placement.source_out == 10.0
    assert placement.context_integrity_score == 1.0


def test_context_protection_keeps_setup_and_payoff_tail(tmp_path, base_config):
    payoff = event("p", "kill", 7.0)
    placement = protect_context_during_sync(variant(tmp_path, anchor=payoff), 12.0, base_config)

    assert placement.source_in <= 5.5
    assert placement.source_out >= 8.0
    assert placement.context_integrity_score >= base_config.minimum_context_integrity


def test_matching_motion_and_signatures_score_higher():
    first = describe_variant_boundary(variant(Path(".")), None, "end")
    second = replace(first, motion_direction=first.motion_direction, source_signature="source", environment_signature="env")

    assert transition_compatibility_score(first, second) > 0.5


def test_missing_boundary_evidence_is_conservative():
    descriptor = BoundaryDescriptor()

    assert descriptor.confidence < 0.5
    assert descriptor.motion_direction == "neutral"
    assert transition_compatibility_score(descriptor, descriptor) < 1.0


def test_impact_metadata_requires_natural_impact_evidence(tmp_path):
    previous = variant(tmp_path, anchor=event("a", "kill", 8.0))
    no_impact = replace(previous, primary_anchor=PayoffEvent("b", "kill", 8.0, 0.9, 0.9, 0.8, {}))

    assert choose_v2_transition(previous, no_impact).impact_cut is False


def test_weak_natural_impact_evidence_does_not_mark_impact_cut(tmp_path):
    previous = variant(tmp_path, anchor=event("a", "kill", 8.0))
    weak_impact = replace(previous, primary_anchor=PayoffEvent("b", "kill", 8.0, 0.9, 0.9, 0.8, {"impact": 0.1}))

    assert choose_v2_transition(previous, weak_impact).impact_cut is False


def test_transition_selects_one_audio_overlap_mode(tmp_path):
    previous = variant(tmp_path, anchor=event("a", "kill", 8.0))
    current = variant(tmp_path, anchor=event("b", "multikill", 8.0))
    decision = choose_v2_transition(previous, current)

    assert not (decision.audio_j_cut_ms and decision.audio_l_cut_ms)


def test_v2_transition_is_hard_cut_with_impact_metadata_only(tmp_path):
    previous = variant(tmp_path, anchor=event("a", "kill", 8.0))
    current = variant(tmp_path, anchor=event("b", "multikill", 8.0))

    decision = choose_v2_transition(previous, current)

    assert decision.transition == "hard_cut"
    assert decision.effect in {None, "impact_flash"}
    assert decision.audio_j_cut_ms >= 0
    assert decision.audio_l_cut_ms >= 0
