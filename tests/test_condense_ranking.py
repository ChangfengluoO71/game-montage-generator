import csv
from dataclasses import replace
from pathlib import Path

import pytest

from montage.condense import (
    build_condensed_variants,
    context_integrity_score,
    select_anchors,
)
from montage.config import load_config
from montage.models import Candidate, CandidateVariant, MusicAnalysis, PayoffEvent, SourceSegment
from montage.ranking import calculate_penalties, score_variant


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_config():
    return load_config(PROJECT_ROOT / "config.yaml")


test_config.__test__ = False


def make_event(event_id, source_time, confidence, strength=None, event_type="combat_climax"):
    return PayoffEvent(
        event_id=event_id,
        type=event_type,
        source_time=source_time,
        confidence=confidence,
        strength=confidence if strength is None else strength,
        semantic_confidence=0.9 if event_type in {"kill", "multikill", "vehicle_destroy", "objective"} else 0.2,
        evidence={"motion_peak": confidence, "audio_transient": strength or confidence},
    )


def make_music():
    return MusicAnalysis(
        source_file=Path("music.flac"), duration=20.0, tempo=120.0,
        beats=[float(index) for index in range(20)], strong_beats=[4.0, 8.0, 12.0],
        bars=[4.0, 8.0, 12.0], onsets=[2.0, 8.0, 12.0],
        edit_points=[{"timestamp": 8.0, "strength": 1.0, "type": "strong_beat"}],
        energy_times=[0.0, 10.0], rms=[0.2, 0.8], onset_strength=[0.2, 0.9], novelty=[0.1, 0.9],
        structure_regions=[], confidence={"beat": 0.9, "bar": 0.8, "section": 0.0, "onset": 0.8},
    )


def make_long_candidate():
    return Candidate(
        candidate_id="long", source_file=Path("long.mp4"), source_start=0.0, source_end=20.0, duration=20.0,
        human_selection_score=0.42, audio_score=0.7, motion_score=0.7, visual_score=0.7,
        continuity_score=0.9, combat_intensity=0.8, uniqueness=0.8, source_category="short_clip",
    )


def make_variant_with_two_segments(candidate, condense_reason):
    first = SourceSegment(candidate.source_file, 0.0, 4.0, 4.0)
    second = SourceSegment(candidate.source_file, 8.0, 12.0, 4.0)
    return CandidateVariant(
        variant_id="two-segments", parent_candidate_id=candidate.candidate_id, source_file=candidate.source_file,
        source_segments=(first, second), duration=8.0, human_selection_prior=candidate.human_selection_prior,
        payoff_score=0.0, combat_intensity=candidate.combat_intensity, action_density=0.0,
        continuity=candidate.continuity_score, visual_novelty=candidate.visual_score, motion=candidate.motion_score,
        audio_activity=candidate.audio_score, danger_score=0.0, uniqueness=candidate.uniqueness, final_score=0.0,
        duplicate_group=None, payoff_events=(), primary_anchor=None, secondary_anchors=(), anchor_event_time=None,
        anchor_event_type=None, anchor_event_strength=None, anchor_event_confidence=None,
        context_integrity_score=1.0, penalty_values={}, source_signature="source", environment_signature="environment",
        weapon_or_view_signature="view", condense_reason=condense_reason, rationale="test",
    )


def test_late_multikill_can_beat_earlier_isolated_kill():
    primary, secondary = select_anchors(
        [make_event("kill", 2.0, 0.92, 0.55, "kill"),
         make_event("multi", 8.0, 0.84, 0.95, "multikill")],
        0.0, 10.0, make_music(), test_config())
    assert primary.event_id == "multi"
    assert secondary[0].event_id == "kill"


def test_event_below_weak_anchor_threshold_cannot_force_condensation():
    primary, secondary = select_anchors(
        [make_event("weak", 8.0, 0.4, 0.4)],
        0.0, 10.0, make_music(), test_config())

    assert primary is None
    assert secondary == ()


def test_stationary_ads_penalty_is_payoff_aware():
    features = {"stationary_ads": 0.9, "motion": 0.1,
                "visual_novelty": 0.1, "danger_escalation": 0.1,
                "repetitive_fire": 0.7}
    no_event = calculate_penalties(features, [], test_config())
    with_event = calculate_penalties(features, [make_event("kill", 3.0, 0.8)], test_config())
    assert with_event["stationary_ads_penalty"] < no_event["stationary_ads_penalty"]


def test_two_segment_condense_requires_reason():
    with pytest.raises(ValueError, match="condense_reason"):
        make_variant_with_two_segments(make_long_candidate(), "")


def test_condensed_variant_preserves_saved_short_clip_prior_and_anchor_context():
    candidate = make_long_candidate()
    variants = build_condensed_variants(candidate, [make_event("payoff", 14.0, 0.8, 0.9, "kill")], make_music(), test_config())

    assert variants
    variant = variants[0]
    assert variant.human_selection_prior == candidate.human_selection_prior
    assert variant.primary_anchor is not None
    assert variant.source_segments[0].source == candidate.source_file
    assert context_integrity_score(candidate, variant.primary_anchor, variant.source_segments) >= test_config().minimum_context_integrity


def test_variant_scoring_uses_all_configured_v11_components_and_penalties():
    candidate = make_long_candidate()
    segment = SourceSegment(candidate.source_file, 0.0, 8.0, 8.0)
    variant = CandidateVariant(
        variant_id="score", parent_candidate_id=candidate.candidate_id, source_file=candidate.source_file,
        source_segments=(segment,), duration=8.0, human_selection_prior=1.0, payoff_score=1.0,
        combat_intensity=1.0, action_density=1.0, continuity=1.0, visual_novelty=1.0, motion=1.0,
        audio_activity=1.0, danger_score=1.0, uniqueness=1.0, final_score=0.0, duplicate_group=None,
        payoff_events=(make_event("payoff", 4.0, 1.0),), primary_anchor=None, secondary_anchors=(),
        anchor_event_time=None, anchor_event_type=None, anchor_event_strength=None, anchor_event_confidence=None,
        context_integrity_score=1.0, penalty_values={"downtime_penalty": 0.2}, source_signature="source",
        environment_signature="environment", weapon_or_view_signature="view", condense_reason="", rationale="test",
    )

    scored = score_variant(variant, test_config())

    assert scored.final_score == pytest.approx(0.8)
    assert set(test_config().v2_weights) <= set(scored.to_dict()["score_components"])


def test_v2_candidate_writer_serializes_score_components_and_penalties(tmp_path):
    from montage.candidate import write_v2_candidates

    candidate = make_long_candidate()
    segment = SourceSegment(candidate.source_file, 0.0, 6.0, 6.0)
    variant = CandidateVariant(
        variant_id="artifact", parent_candidate_id=candidate.candidate_id, source_file=candidate.source_file,
        source_segments=(segment,), duration=6.0, human_selection_prior=0.42, payoff_score=0.8,
        combat_intensity=0.7, action_density=0.6, continuity=0.9, visual_novelty=0.5, motion=0.4,
        audio_activity=0.3, danger_score=0.2, uniqueness=0.8, final_score=0.5, duplicate_group=None,
        payoff_events=(), primary_anchor=None, secondary_anchors=(), anchor_event_time=None,
        anchor_event_type=None, anchor_event_strength=None, anchor_event_confidence=None,
        context_integrity_score=0.9, penalty_values={"downtime_penalty": 0.1}, source_signature="source",
        environment_signature="environment", weapon_or_view_signature="view", condense_reason="", rationale="test",
        score_components={"payoff_score": 0.2, "downtime_penalty": 0.1},
    )
    json_path = tmp_path / "candidates.json"
    csv_path = tmp_path / "candidates.csv"

    write_v2_candidates([variant], json_path, csv_path)

    assert '"score_components"' in json_path.read_text(encoding="utf-8")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["payoff_score"] == "0.2"
    assert row["downtime_penalty"] == "0.1"
