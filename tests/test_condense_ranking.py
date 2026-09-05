import csv
from dataclasses import replace
from pathlib import Path

import pytest

from montage.condense import (
    _best_rapid_window,
    build_condensed_variants,
    context_integrity_score,
    select_anchors,
)
from montage.candidate import generate_candidates, write_candidates
from montage.config import load_config
from montage.models import Candidate, CandidateVariant, MediaRecord, MusicAnalysis, PayoffEvent, SourceSegment, VideoAnalysis
from montage.ranking import (
    calculate_penalties,
    max_verified_kills_in_window,
    score_variant,
    verified_kill_events,
)


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


def make_verified_event(event_id, source_time, event_type="kill", confidence=0.92, strength=0.92):
    return PayoffEvent(
        event_id=event_id,
        type=event_type,
        source_time=source_time,
        confidence=confidence,
        strength=strength,
        semantic_confidence=0.92,
        evidence={
            "killfeed_change": 0.92,
            "reward_roi_change": 0.92,
            "motion_peak": strength,
            "audio_transient": strength,
        },
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


@pytest.mark.parametrize(("confidence", "strength"), [(0.4, 0.9), (0.9, 0.4)])
def test_anchor_requires_confidence_and_strength_above_weak_threshold(confidence, strength):
    primary, secondary = select_anchors(
        [make_event("half-weak", 8.0, confidence, strength, "kill")],
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


def test_saved_short_clip_keeps_complete_continuous_sequence():
    candidate = replace(
        make_long_candidate(),
        source_category="short_clip",
        source_start=0.0,
        source_end=20.0,
        duration=20.0,
        human_selection_score=0.85,
    )
    events = [
        make_event("saved-1", 4.0, 0.86, 0.86, "kill"),
        make_event("saved-2", 6.0, 0.90, 0.90, "kill"),
        make_event("saved-3", 10.0, 0.92, 0.92, "multikill"),
    ]

    variant = build_condensed_variants(candidate, events, make_music(), test_config())[0]

    assert variant.source_segments == (SourceSegment(candidate.source_file, 0.0, 20.0, 20.0),)
    assert variant.duration == 20.0
    assert variant.human_selection_prior == 0.85
    assert "complete" in variant.rationale.lower()


def test_rapid_kill_condense_keeps_the_complete_short_sequence():
    candidate = replace(make_long_candidate(), source_category="long_clip")
    events = [
        make_event("kill-1", 10.0, 0.86, 0.86, "kill"),
        make_event("kill-2", 11.2, 0.88, 0.88, "kill"),
        make_event("multi", 13.9, 0.94, 0.96, "multikill"),
    ]

    variant = build_condensed_variants(candidate, events, make_music(), test_config())[0]
    segment = variant.source_segments[0]

    assert len(variant.source_segments) == 1
    assert segment.source_in <= events[0].source_time
    assert segment.source_out >= events[-1].source_time
    assert variant.duration <= 6.5
    assert "rapid-kill" in variant.rationale.lower()
    assert {event.event_id for event in variant.payoff_events} == {"kill-1", "kill-2", "multi"}


def test_rapid_multikill_bonus_cap_matches_editorial_priority():
    assert test_config().rapid_multikill_bonus_weight == pytest.approx(0.18)


def test_rapid_window_rounding_keeps_segment_duration_consistent():
    candidate = make_long_candidate()
    events = [
        make_event("round-1", 10.0001, 0.9, 0.9, "kill"),
        make_event("round-2", 11.0026, 0.9, 0.9, "kill"),
    ]

    result = _best_rapid_window(candidate, events, test_config())

    assert result is not None
    segment, _ = result
    assert segment.duration == pytest.approx(segment.source_out - segment.source_in, abs=0.000001)


def test_boundary_anchor_falls_back_to_original_continuous_range():
    candidate = make_long_candidate()
    anchor = make_event("boundary", 0.2, 0.9, 0.9, "kill")
    unsafe_segment = SourceSegment(candidate.source_file, 0.0, 6.0, 6.0)

    assert context_integrity_score(candidate, anchor, (unsafe_segment,)) < test_config().minimum_context_integrity
    variant = build_condensed_variants(candidate, [anchor], make_music(), test_config())[0]
    assert variant.source_segments == (SourceSegment(candidate.source_file, 0.0, 20.0, 20.0),)


def test_quality_rapid_window_accepts_source_end_after_three_decimal_rounding(base_config):
    candidate = replace(
        make_long_candidate(),
        source_category="long_clip",
        source_start=0.0,
        source_end=20.0,
        duration=20.0,
    )
    config = replace(
        base_config,
        v5_profile="quality",
        v5_min_verified_kills_per_shot=1,
        v5_min_rapid_context_tail_s=0.25,
        v5_sequence_context_s=1.0,
        v5_max_sequences_per_candidate=3,
    )
    events = [
        make_verified_event("edge-1", 18.0),
        make_verified_event("edge-2", 19.001),
    ]

    variants = build_condensed_variants(candidate, events, make_music(), config)

    assert variants
    assert len(variants[0].source_segments) == 1
    assert variants[0].source_segments[0].source_out == pytest.approx(20.0)
    assert {event.event_id for event in variants[0].payoff_events} == {"edge-1", "edge-2"}


def test_quality_long_saved_clip_yields_multiple_non_overlapping_rapid_sequences(base_config):
    candidate = replace(
        make_long_candidate(),
        source_category="short_clip",
        source_start=0.0,
        source_end=70.0,
        duration=70.0,
        human_selection_score=0.85,
    )
    config = replace(
        base_config,
        v5_profile="quality",
        v5_min_verified_kills_per_shot=1,
        v5_max_sequences_per_candidate=4,
        v5_kill_density_window_s=6.0,
        v5_sequence_context_s=1.25,
    )
    events = [
        make_verified_event("cluster-a1", 4.0),
        make_verified_event("cluster-a2", 5.2),
        make_verified_event("cluster-a3", 6.4, "multikill"),
        make_verified_event("cluster-b1", 35.0),
        make_verified_event("cluster-b2", 36.1),
        make_verified_event("cluster-b3", 37.4, "multikill"),
    ]

    variants = build_condensed_variants(candidate, events, make_music(), config)

    assert len(variants) >= 2
    assert all(len(variant.source_segments) == 1 for variant in variants)
    ranges = [(variant.source_segments[0].source_in, variant.source_segments[0].source_out) for variant in variants]
    assert all(left[1] <= right[0] or right[1] <= left[0] for index, left in enumerate(ranges) for right in ranges[index + 1:])
    assert all("verified kills" in variant.rationale.lower() for variant in variants)
    assert all("trim" in variant.rationale.lower() for variant in variants)


def test_quality_short_saved_clip_trims_non_event_head_before_rapid_kills(base_config):
    candidate = replace(
        make_long_candidate(),
        source_category="short_clip",
        source_start=0.0,
        source_end=11.4,
        duration=11.4,
        human_selection_score=0.85,
    )
    config = replace(
        base_config,
        v5_profile="quality",
        v5_min_verified_kills_per_shot=1,
        v5_sequence_context_s=1.25,
        v5_min_rapid_context_tail_s=0.20,
    )
    events = [
        make_verified_event("late-1", 7.0),
        make_verified_event("late-2", 8.1),
        make_verified_event("late-3", 9.2, "multikill"),
    ]

    variants = build_condensed_variants(candidate, events, make_music(), config)

    assert variants
    segment = variants[0].source_segments[0]
    assert segment.source_in == pytest.approx(5.75)
    assert segment.source_out == pytest.approx(10.45)
    assert segment.duration < candidate.duration
    assert "trimmed" in variants[0].rationale.lower()


def test_verified_kill_filter_requires_correlated_evidence(base_config):
    generic = make_event("generic", 1.0, 0.99, 0.99, "combat_climax")
    weak_feed = replace(
        make_verified_event("weak-feed", 2.0),
        evidence={"killfeed_change": 0.10, "reward_roi_change": 0.92},
    )
    corroborated = make_verified_event("corroborated", 3.0, "multikill")

    verified = verified_kill_events([generic, weak_feed, corroborated], base_config)

    assert [event.event_id for event in verified] == ["corroborated"]


def test_quality_score_prefers_verified_kill_count_and_density(base_config):
    candidate = make_long_candidate()
    base = build_condensed_variants(candidate, [], make_music(), base_config)[0]
    sparse = replace(
        base,
        variant_id="sparse",
        payoff_events=(make_verified_event("sparse-kill", 5.0),),
    )
    dense = replace(
        base,
        variant_id="dense",
        payoff_events=tuple(make_verified_event(f"dense-{index}", 5.0 + index * 0.75) for index in range(4)),
    )
    quality = replace(
        base_config,
        v5_profile="quality",
        v5_kill_count_weight=0.20,
        v5_kill_density_weight=0.35,
        v5_kill_density_window_s=6.0,
        v5_kill_density_target=4,
        rapid_multikill_bonus_weight=0.0,
    )

    scored_sparse = score_variant(sparse, quality)
    scored_dense = score_variant(dense, quality)

    assert scored_dense.final_score > scored_sparse.final_score
    assert scored_dense.score_components["verified_kill_count"] == 4.0
    assert scored_dense.score_components["kill_density"] == pytest.approx(1.0)
    assert base_config.v5_kill_count_weight == 0.0
    assert base_config.v5_kill_density_weight == 0.0


def test_generated_variant_uses_explicit_feature_runs_for_penalties():
    candidate = replace(
        make_long_candidate(),
        feature_runs={
            "stationary_ads": 0.9,
            "motion": 0.1,
            "visual_novelty": 0.1,
            "danger_escalation": 0.1,
            "downtime": 0.8,
            "same_view": 0.7,
            "repetitive_fire": 0.6,
        },
    )
    no_payoff = build_condensed_variants(candidate, [], make_music(), test_config())[0]
    with_payoff = build_condensed_variants(
        candidate, [make_event("kill", 14.0, 0.9, 0.9, "kill")], make_music(), test_config())[0]

    assert no_payoff.penalty_values["stationary_ads_penalty"] > 0.0
    assert no_payoff.penalty_values["downtime_penalty"] > 0.0
    assert no_payoff.penalty_values["same_view_penalty"] > 0.0
    assert no_payoff.penalty_values["repetitive_fire_penalty"] > 0.0
    assert with_payoff.penalty_values["stationary_ads_penalty"] < no_payoff.penalty_values["stationary_ads_penalty"]
    assert with_payoff.penalty_values["downtime_penalty"] < no_payoff.penalty_values["downtime_penalty"]
    assert with_payoff.penalty_values["repetitive_fire_penalty"] < no_payoff.penalty_values["repetitive_fire_penalty"]


def test_normal_activity_window_derives_runs_that_reach_variant_penalties():
    source = Path("normal-window.mp4")
    record = MediaRecord(
        file_path=source, file_name=source.name, file_size=1, duration=120.0, width=1920, height=1200,
        fps=60.0, codec="h264", bitrate=None, audio_codec=None, audio_channels=None,
        audio_sample_rate=None, creation_time=None, category="long", fingerprint={},
    )
    analysis = VideoAnalysis(
        source_file=source, sample_rate=1.0, times=[float(index) for index in range(12)],
        motion=[0.1] * 9 + [0.9] * 3,
        visual=[0.1] * 9 + [0.9] * 3,
        audio=[0.8] * 3 + [0.1] * 3 + [0.8] * 3 + [0.1] * 3,
        continuity=[0.8] * 12,
        activity=[0.2] * 3 + [0.1] * 3 + [0.3] * 3 + [0.9] * 3,
        candidate_windows=[{"start": 4.0, "end": 10.0}],
    )

    candidate = generate_candidates([record], {str(source): analysis}, test_config())[0]
    variant = build_condensed_variants(candidate, [], make_music(), test_config())[0]

    assert candidate.feature_runs["stationary_ads"] > 0.0
    assert candidate.feature_runs["downtime"] > 0.0
    assert candidate.feature_runs["same_view"] > 0.0
    assert candidate.feature_runs["repetitive_fire"] > 0.0
    assert variant.penalty_values["stationary_ads_penalty"] > 0.0
    assert variant.penalty_values["downtime_penalty"] > 0.0
    assert variant.penalty_values["same_view_penalty"] > 0.0
    assert variant.penalty_values["repetitive_fire_penalty"] > 0.0


def test_expanded_candidate_range_overrides_stale_core_window_diagnostics():
    source = Path("expanded-window.mp4")
    record = MediaRecord(
        file_path=source, file_name=source.name, file_size=1, duration=120.0, width=1920, height=1200,
        fps=60.0, codec="h264", bitrate=None, audio_codec=None, audio_channels=None,
        audio_sample_rate=None, creation_time=None, category="long", fingerprint={},
    )
    analysis = VideoAnalysis(
        source_file=source, sample_rate=1.0, times=[float(index) for index in range(13)],
        motion=[0.1] * 4 + [0.8] * 3 + [0.1] * 6,
        visual=[0.1] * 4 + [0.8] * 3 + [0.1] * 6,
        audio=[0.8] * 4 + [0.8] * 3 + [0.1] * 6,
        continuity=[0.8] * 13,
        activity=[0.3] * 4 + [0.9] * 3 + [0.1] * 6,
        candidate_windows=[{
            "start": 4.0, "end": 6.0,
            "stationary_ads": 0.0, "same_view": 0.0, "downtime": 0.0, "repetitive_fire": 0.0,
            "danger_escalation": 0.0, "motion": 0.8, "visual_novelty": 0.8,
        }],
    )

    candidate = generate_candidates([record], {str(source): analysis}, test_config())[0]
    variant = build_condensed_variants(candidate, [], make_music(), test_config())[0]

    assert (candidate.source_start, candidate.source_end) == (0.0, 12.0)
    assert candidate.feature_runs["stationary_ads"] > 0.0
    assert candidate.feature_runs["downtime"] > 0.0
    assert candidate.feature_runs["same_view"] > 0.0
    assert candidate.feature_runs["repetitive_fire"] > 0.0
    assert variant.penalty_values["stationary_ads_penalty"] > 0.0
    assert variant.penalty_values["downtime_penalty"] > 0.0
    assert variant.penalty_values["same_view_penalty"] > 0.0
    assert variant.penalty_values["repetitive_fire_penalty"] > 0.0


def test_v1_candidate_writer_excludes_v2_feature_runs(tmp_path):
    candidate = replace(make_long_candidate(), feature_runs={"stationary_ads": 0.9})
    json_path = tmp_path / "v1-candidates.json"
    csv_path = tmp_path / "v1-candidates.csv"

    write_candidates([candidate], json_path, csv_path)

    assert "feature_runs" not in candidate.to_dict()
    assert "feature_runs" not in json_path.read_text(encoding="utf-8")
    assert "feature_runs" not in csv_path.read_text(encoding="utf-8-sig")


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


def test_v2_candidate_writer_serializes_score_components_and_penalties_to_configured_work(tmp_path):
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
    config = replace(test_config(), raw_dir=tmp_path / "raw", work_dir=tmp_path / "work", output_dir=tmp_path / "output")
    json_path = config.highlight_candidates_v2_path
    csv_path = config.highlight_candidates_v2_csv_path

    write_v2_candidates([variant], json_path, csv_path, config)

    assert '"score_components"' in json_path.read_text(encoding="utf-8")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["payoff_score"] == "0.2"
    assert row["downtime_penalty"] == "0.1"


@pytest.mark.parametrize("destination", ("raw", "baseline", "outside"))
def test_v2_candidate_writer_rejects_non_work_destinations(tmp_path, destination):
    from montage.candidate import write_v2_candidates

    config = replace(test_config(), raw_dir=tmp_path / "raw", work_dir=tmp_path / "work", output_dir=tmp_path / "output")
    if destination == "raw":
        path = config.raw_dir / "highlight_candidates_v2.json"
    elif destination == "baseline":
        path = config.baseline_output_path
    else:
        path = tmp_path / "outside" / "highlight_candidates_v2.json"

    with pytest.raises(ValueError, match="work_dir"):
        write_v2_candidates([], path, path.with_suffix(".csv"), config)
    assert not path.exists()
