from dataclasses import replace
from pathlib import Path

import pytest

from montage.models import BoundaryDescriptor, CandidateVariant, MusicAnalysis, PayoffEvent, SourceSegment, VideoAnalysis
from montage.models import EditDecisionList, V2EditDecisionList
import montage.beam_timeline as beam_timeline
from montage.beam_timeline import BeamState, build_v2_preview_edit, expand_beam, validate_v2_edit, _expansion_score
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
    if tmp_path != Path("."):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch(exist_ok=True)
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


def test_boundary_window_uses_configured_milliseconds(tmp_path, base_config):
    candidate = variant(tmp_path)
    configured = replace(base_config, boundary_window_ms=700)
    analysis = VideoAnalysis(
        candidate.source_file, 10.0, [9.4], [0.9], [0.7], [0.2], [0.8], [0.9], [0.6],
    )

    descriptor = describe_variant_boundary(
        candidate, analysis, "end", boundary_window_ms=configured.boundary_window_ms
    )

    assert descriptor.confidence >= 0.5
    assert descriptor.motion_strength == 0.9


def test_non_edge_analysis_samples_keep_boundary_evidence_low_confidence(tmp_path, base_config):
    candidate = variant(tmp_path)
    analysis = VideoAnalysis(
        candidate.source_file, 10.0, [2.0], [0.9], [0.7], [0.2], [0.8], [0.9], [0.6],
    )

    descriptor = describe_variant_boundary(
        candidate, analysis, "end", boundary_window_ms=base_config.boundary_window_ms
    )

    assert descriptor.confidence < 0.5
    assert transition_compatibility_score(descriptor, descriptor) < 1.0


def test_impact_metadata_requires_natural_impact_evidence(tmp_path):
    previous = variant(tmp_path, anchor=event("a", "kill", 8.0))
    no_impact = replace(previous, primary_anchor=PayoffEvent("b", "kill", 8.0, 0.9, 0.9, 0.8, {}))

    assert choose_v2_transition(previous, no_impact).impact_cut is False


def test_weak_natural_impact_evidence_does_not_mark_impact_cut(tmp_path):
    previous = variant(tmp_path, anchor=event("a", "kill", 8.0))
    weak_impact = replace(previous, primary_anchor=PayoffEvent("b", "kill", 8.0, 0.9, 0.9, 0.8, {"impact": 0.1}))

    assert choose_v2_transition(previous, weak_impact).impact_cut is False


def test_audio_impact_is_natural_impact_evidence(tmp_path):
    previous = variant(tmp_path, anchor=event("a", "kill", 8.0))
    current = replace(
        previous,
        primary_anchor=PayoffEvent("b", "combat_climax", 8.0, 0.9, 0.9, 0.4, {"audio_impact": 0.5}),
    )

    assert choose_v2_transition(previous, current).impact_cut is True


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


def timeline_variant(tmp_path: Path, index: int, *, duration: float = 6.0, quality: float = 0.8,
                     duplicate_group: str | None = None, environment: str | None = None,
                     anchor_type: str = "kill") -> CandidateVariant:
    raw = tmp_path / "raw"
    source = raw / f"clip-{index}.mp4"
    raw.mkdir(parents=True, exist_ok=True)
    source.touch(exist_ok=True)
    segment = SourceSegment(source, 0.0, duration, duration)
    payoff = event(f"payoff-{index}", anchor_type, duration - 1.0, strength=quality)
    return replace(
        variant(tmp_path, anchor=payoff, signature=environment or f"env-{index}"),
        variant_id=f"v-{index}", parent_candidate_id=f"c-{index}", source_file=source,
        source_segments=(segment,), duration=duration, human_selection_prior=quality,
        payoff_score=quality, combat_intensity=quality, action_density=quality,
        continuity=quality, visual_novelty=quality, motion=quality, audio_activity=quality,
        danger_score=quality, final_score=quality, duplicate_group=duplicate_group,
        environment_signature=environment or f"env-{index}",
        rationale=f"payoff {anchor_type} setup action payoff tail",
    )


def baseline_edit_for_timeline() -> EditDecisionList:
    return EditDecisionList("preview", Path("music.wav"), 19.0, 74.252, 55.252, "locked", [])


def test_v2_beam_builds_bounded_payoff_aware_edit(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw", work_dir=tmp_path / "work")
    variants = [timeline_variant(tmp_path, index) for index in range(10)]

    edit = build_v2_preview_edit(variants, music(), baseline_edit_for_timeline(), config)

    assert 8 <= len(edit.shots) <= 14
    assert 45.0 <= edit.duration <= 60.0
    assert len({shot.duplicate_group for shot in edit.shots}) == len(edit.shots)
    assert edit.shots[-1].primary_anchor is not None
    assert edit.shots[-1].anchor_event_type == "kill"
    assert (config.preview_v2_edit_path).exists()
    assert (config.preview_v2_timeline_path).exists()


def test_v2_validation_rejects_repeated_duplicate_groups_and_bad_ranges(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw")
    first = timeline_variant(tmp_path, 1)
    shot = first
    from montage.models import V2EditShot
    edit_shot = V2EditShot(
        source=first.source_file, source_in=0.0, source_out=first.duration,
        duration=first.duration, candidate_score=first.final_score,
        duplicate_group="dg-1", timeline_in=0.0, timeline_out=first.duration,
        transition="hard_cut", music_target=20.0, music_event_type="beat",
        sync_offset=0.0, rationale="setup action payoff tail", source_segments=first.source_segments,
        parent_candidate_id=first.parent_candidate_id, variant_id=first.variant_id,
        payoff_events=first.payoff_events, primary_anchor=first.primary_anchor,
        anchor_event_time=first.anchor_event_time, anchor_event_type=first.anchor_event_type,
        anchor_event_strength=first.anchor_event_strength, anchor_event_confidence=first.anchor_event_confidence,
    )
    edit = V2EditDecisionList("preview_v2", Path("music.wav"), 19.0, 74.252, 19.0, 74.252,
                              55.0, "locked", tuple(edit_shot for _ in range(8)))

    with pytest.raises(ValueError, match="duplicate"):
        validate_v2_edit(edit, config)

    bad_segment = SourceSegment(first.source_file, 0.0, first.duration, first.duration)
    bad = replace(edit_shot, source_in=-1.0, source_segments=(bad_segment,))
    invalid = replace(edit, shots=tuple([bad] + [edit_shot] * 7))
    with pytest.raises(ValueError, match="range"):
        validate_v2_edit(invalid, config)


def test_beam_penalizes_recent_source_and_environment_reuse(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw")
    repeated = replace(timeline_variant(tmp_path, 1), source_signature="source-a", environment_signature="env-a")
    fresh = replace(timeline_variant(tmp_path, 2), source_signature="source-b", environment_signature="env-b")
    state = BeamState(recent_sources=("source-a",), recent_environments=("env-a",))

    expanded = expand_beam(state, [repeated, fresh], music(), config)

    assert expanded[0].shots[0].variant_id == "v-2"


def test_finale_prefers_hero_play_when_quality_is_comparable(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw", beam_max_expansions=1000)
    regular = [timeline_variant(tmp_path, index, quality=0.80) for index in range(9)]
    hero = timeline_variant(tmp_path, 99, quality=0.79, anchor_type="hero_play")

    edit = build_v2_preview_edit(regular + [hero], music(), baseline_edit_for_timeline(), config)

    assert edit.shots[-1].anchor_event_type == "hero_play"


def test_music_placement_uses_supplied_baseline_edit_start(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw")
    candidate = timeline_variant(tmp_path, 1)
    state = BeamState()
    from montage.beam_timeline import _make_shot

    shot = _make_shot(candidate, state, music(), config, 55.0, baseline_music_in=31.0)

    assert shot.music_target == 31.0


def test_rapid_multikill_bonus_is_not_added_twice(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw")
    candidate = replace(timeline_variant(tmp_path, 1), final_score=0.8, rapid_multikill_bonus=0.12)
    from montage.beam_timeline import _make_shot, _expansion_score

    shot = _make_shot(candidate, BeamState(), music(), config, 55.0)
    score = _expansion_score(candidate, BeamState(), shot, music(), config)
    control = replace(candidate, rapid_multikill_bonus=0.0)
    control_shot = _make_shot(control, BeamState(), music(), config, 55.0)

    assert score == _expansion_score(control, BeamState(), control_shot, music(), config)


def test_previous_shot_signatures_survive_transition_reconstruction(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw")
    first = replace(timeline_variant(tmp_path, 1), source_signature="src-a", environment_signature="env-a", weapon_or_view_signature="ads")
    second = replace(timeline_variant(tmp_path, 2), source_signature="src-b", environment_signature="env-b")
    expanded = expand_beam(BeamState(), [first], music(), config)
    next_states = expand_beam(expanded[0], [second], music(), config)

    assert next_states[0].shots[0].source_signature == "src-a"
    assert next_states[0].shots[0].environment_signature == "env-a"
    assert next_states[0].shots[0].weapon_or_view_signature == "ads"


def test_validation_rejects_missing_or_directory_sources(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw")
    first = timeline_variant(tmp_path, 1)
    shot = _shot_for_test(first)
    edit = _edit_for_test([shot] * 8)
    first.source_file.unlink()

    with pytest.raises(ValueError, match="regular file"):
        validate_v2_edit(edit, config)

    first.source_file.parent.mkdir(parents=True, exist_ok=True)
    first.source_file.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="regular file"):
        validate_v2_edit(edit, config)


def test_validation_rejects_segment_gap_and_mismatched_shot_range(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw")
    first = timeline_variant(tmp_path, 1)
    first.source_file.parent.mkdir(parents=True, exist_ok=True)
    first.source_file.write_bytes(b"raw")
    gap_segments = (SourceSegment(first.source_file, 0.0, 2.0, 2.0), SourceSegment(first.source_file, 3.0, 7.0, 4.0))
    shot = replace(_shot_for_test(first), source_in=0.0, source_out=2.0, duration=6.0,
                   source_segments=gap_segments, condense_reason="")
    with pytest.raises(ValueError, match="condense_reason"):
        validate_v2_edit(_edit_for_test([replace(shot, duplicate_group=f"d-{i}", variant_id=f"v-{i}",
                                                  timeline_in=i * 6.0, timeline_out=(i + 1) * 6.0) for i in range(8)]), config)


@pytest.mark.parametrize("reason", [
    "downtime_removed", "phrase_boundary", "bar_boundary", "action_phase_change",
    "substantial_spatial_change", "substantial_state_change",
])
def test_validation_allows_documented_condensed_source_gaps(tmp_path, base_config, reason):
    config = replace(base_config, raw_dir=tmp_path / "raw")
    first = timeline_variant(tmp_path, 1)
    first.source_file.parent.mkdir(parents=True, exist_ok=True)
    first.source_file.write_bytes(b"raw")
    condensed = replace(
        first,
        source_segments=(
            SourceSegment(first.source_file, 0.0, 2.0, 2.0),
                SourceSegment(first.source_file, 3.0, 7.0, 4.0),
        ),
        duration=6.0,
        condense_reason=reason,
    )

    shots = [replace(_shot_for_test(condensed), source_in=0.0, source_out=2.0,
                      duplicate_group=f"d-{i}", variant_id=f"v-{i}",
                      timeline_in=i * 6.0, timeline_out=(i + 1) * 6.0) for i in range(8)]
    validate_v2_edit(_edit_for_test(shots), config)


@pytest.mark.parametrize("segments", [
    (SourceSegment(Path("placeholder"), 0.0, 2.0, 2.0), SourceSegment(Path("placeholder"), 1.0, 4.0, 3.0)),
    (SourceSegment(Path("placeholder"), 2.0, 4.0, 2.0), SourceSegment(Path("placeholder"), 0.0, 1.0, 1.0)),
])
def test_validation_rejects_overlapping_or_badly_ordered_segments(tmp_path, base_config, segments):
    config = replace(base_config, raw_dir=tmp_path / "raw")
    first = timeline_variant(tmp_path, 1)
    first.source_file.parent.mkdir(parents=True, exist_ok=True)
    first.source_file.write_bytes(b"raw")
    actual_segments = tuple(SourceSegment(first.source_file, segment.source_in, segment.source_out, segment.duration)
                            for segment in segments)
    shot = replace(
        _shot_for_test(first),
        source_in=actual_segments[0].source_in,
        source_out=actual_segments[0].source_out,
        duration=sum(segment.duration for segment in actual_segments),
        source_segments=actual_segments,
        condense_reason="phrase_boundary",
    )
    shots = [replace(shot, duplicate_group=f"d-{i}", variant_id=f"v-{i}",
                      timeline_in=i * shot.duration, timeline_out=(i + 1) * shot.duration) for i in range(8)]

    with pytest.raises(ValueError, match="overlap|ordering"):
        validate_v2_edit(_edit_for_test(shots), config)


def test_make_shot_legacy_source_range_uses_first_condensed_segment(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw")
    condensed = two_segment_variant(tmp_path, event("p", "kill", 7.0))

    shot = beam_timeline._make_shot(condensed, BeamState(), music(), config, 55.0)

    assert (shot.source_in, shot.source_out) == (0.0, 3.0)
    assert shot.source_segments[-1].source_out == 10.0


def test_validation_requires_contiguous_timeline_from_zero_to_edit_duration(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw")
    first = timeline_variant(tmp_path, 1)
    first.source_file.parent.mkdir(parents=True, exist_ok=True)
    first.source_file.write_bytes(b"raw")
    shots = [_shot_for_test(replace(first, variant_id=f"v-{i}", duplicate_group=f"d-{i}")) for i in range(8)]
    invalid = replace(_edit_for_test(shots), shots=tuple(replace(s, timeline_in=1.0 if i == 0 else s.timeline_in, timeline_out=(s.timeline_out + 1.0 if i == 7 else s.timeline_out)) for i, s in enumerate(shots)))
    with pytest.raises(ValueError, match="timeline"):
        validate_v2_edit(invalid, config)


def test_hero_preference_has_quality_margin(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw", hero_quality_margin=0.02, beam_max_expansions=1000)
    regular = [timeline_variant(tmp_path, index, quality=0.80) for index in range(9)]
    weak_hero = timeline_variant(tmp_path, 99, quality=0.60, anchor_type="hero_play")

    edit = build_v2_preview_edit(regular + [weak_hero], music(), baseline_edit_for_timeline(), config)

    assert edit.shots[-1].anchor_event_type != "hero_play"


def test_duration_limits_allow_documented_context_exception_only(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw")
    candidate = timeline_variant(tmp_path, 1, duration=11.0)
    from montage.beam_timeline import _duration_allowed

    assert not _duration_allowed(11.0, "kill", "", 1.0, config)
    assert _duration_allowed(11.0, "kill", "phrase_boundary", 1.0, config)
    assert not _duration_allowed(13.0, "hero_play", "", 1.0, config)


def test_expansion_budget_is_global_and_independent_of_beam_width(tmp_path, base_config, monkeypatch):
    config = replace(base_config, raw_dir=tmp_path / "raw", work_dir=tmp_path / "work", beam_max_expansions=64, beam_width=2)
    variants = [timeline_variant(tmp_path, i) for i in range(8)]
    expansions = 0
    original_make_shot = beam_timeline._make_shot

    def observing_make_shot(*args, **kwargs):
        nonlocal expansions
        expansions += 1
        return original_make_shot(*args, **kwargs)

    monkeypatch.setattr(beam_timeline, "_make_shot", observing_make_shot)
    build_v2_preview_edit(variants, music(), baseline_edit_for_timeline(), config)
    assert expansions <= config.beam_max_expansions


def test_dynamic_energy_uses_target_duration(tmp_path, base_config):
    config = replace(base_config, raw_dir=tmp_path / "raw", beam_max_expansions=1)
    variants = [timeline_variant(tmp_path, i) for i in range(3)]
    assert len(expand_beam(BeamState(), variants, music(), config)) == 1

    candidate = variants[0]
    from montage.beam_timeline import _make_shot
    short_state = BeamState(target_duration=45.0)
    long_state = BeamState(target_duration=60.0)
    short_score = _expansion_score(candidate, short_state, _make_shot(candidate, short_state, music(), config, 45.0), music(), config)
    long_score = _expansion_score(candidate, long_state, _make_shot(candidate, long_state, music(), config, 60.0), music(), config)
    assert short_score != long_score


def _shot_for_test(variant):
    from montage.models import V2EditShot
    return V2EditShot(
        source=variant.source_file, source_in=variant.source_segments[0].source_in,
        source_out=variant.source_segments[-1].source_out, duration=variant.duration,
        candidate_score=variant.final_score, duplicate_group=variant.duplicate_group or variant.variant_id,
        timeline_in=0.0, timeline_out=variant.duration, transition="hard_cut", music_target=20.0,
        music_event_type="beat", sync_offset=0.0, rationale="setup action payoff tail",
        source_segments=variant.source_segments, parent_candidate_id=variant.parent_candidate_id,
        variant_id=variant.variant_id, payoff_events=variant.payoff_events, primary_anchor=variant.primary_anchor,
        anchor_event_time=variant.anchor_event_time, anchor_event_type=variant.anchor_event_type,
        anchor_event_strength=variant.anchor_event_strength, anchor_event_confidence=variant.anchor_event_confidence,
        condense_reason=variant.condense_reason,
    )


def _edit_for_test(shots):
    return V2EditDecisionList("preview_v2", Path("music.wav"), 19.0, 74.252, 19.0, 74.252, 48.0, "locked", tuple(shots))
