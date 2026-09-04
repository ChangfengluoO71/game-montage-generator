from dataclasses import replace
from pathlib import Path

from montage.config import load_config
from montage.curation import (
    event_is_editorially_excluded,
    excluded_range_reason,
    variant_is_editorially_excluded,
)
from montage.models import CandidateVariant, PayoffEvent, SourceSegment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _event(source_time: float) -> PayoffEvent:
    return PayoffEvent(
        event_id="range-event", type="kill", source_time=source_time,
        confidence=0.9, strength=0.9, semantic_confidence=0.8,
        evidence={"audio_transient": 0.9},
    )


def _variant(source: Path, start: float, end: float) -> CandidateVariant:
    segment = SourceSegment(source, start, end, end - start)
    return CandidateVariant(
        variant_id="range-variant", parent_candidate_id="range-candidate", source_file=source,
        source_segments=(segment,), duration=end - start, human_selection_prior=0.8,
        payoff_score=0.9, combat_intensity=0.8, action_density=0.8, continuity=0.9,
        visual_novelty=0.8, motion=0.8, audio_activity=0.8, danger_score=0.7,
        uniqueness=0.8, final_score=0.8, duplicate_group="range-group", payoff_events=(),
        primary_anchor=None, secondary_anchors=(), anchor_event_time=None,
        anchor_event_type=None, anchor_event_strength=None, anchor_event_confidence=None,
        context_integrity_score=1.0, penalty_values={}, source_signature="source",
        environment_signature="range", weapon_or_view_signature="rifle", condense_reason="",
        rationale="test",
    )


def test_editorial_exclusion_matches_only_overlapping_source_time():
    config = replace(
        load_config(PROJECT_ROOT / "config.yaml"),
        v2_excluded_ranges=(("training.mp4", 49.0, 55.0, "training_range"),),
    )
    source = Path("D:/captures/training.mp4")

    assert excluded_range_reason(source, 50.0, 52.0, config) == "training_range"
    assert excluded_range_reason(source, 55.0, 58.0, config) is None
    assert event_is_editorially_excluded(_event(54.999), source, config)
    assert not event_is_editorially_excluded(_event(55.0), source, config)


def test_editorial_exclusion_rejects_variant_overlapping_range_but_keeps_other_context():
    config = replace(
        load_config(PROJECT_ROOT / "config.yaml"),
        v2_excluded_ranges=(("training.mp4", 49.0, 55.0, "training_range"),),
    )
    source = Path("D:/captures/training.mp4")

    assert variant_is_editorially_excluded(_variant(source, 48.0, 50.0), config)
    assert not variant_is_editorially_excluded(_variant(source, 55.0, 60.0), config)
