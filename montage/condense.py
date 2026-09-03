"""Payoff-anchor selection and context-safe candidate condensation."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Sequence

from .config import PipelineConfig
from .models import Candidate, CandidateVariant, MusicAnalysis, PayoffEvent, SourceSegment
from .ranking import calculate_penalties


_SEMANTIC_IMPORTANCE = {
    "multikill": 1.0,
    "vehicle_destroy": 0.96,
    "objective": 0.90,
    "kill": 0.84,
    "danger_climax": 0.74,
    "combat_climax": 0.70,
    "impact_event": 0.50,
    "visual_transient": 0.32,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _music_opportunity(event: PayoffEvent, music: MusicAnalysis) -> float:
    points = [*music.strong_beats, *music.bars, *music.onsets]
    if not points:
        return 0.0
    return _clamp(1.0 - min(abs(event.source_time - float(point)) for point in points) / 1.0)


def _anchor_utility(
    event: PayoffEvent, candidate_start: float, candidate_end: float, music: MusicAnalysis
) -> float:
    duration = max(candidate_end - candidate_start, 0.001)
    position = _clamp((event.source_time - candidate_start) / duration)
    payoff_position = _clamp(1.0 - abs(position - 0.68) / 0.68)
    setup = _clamp((event.source_time - candidate_start) / 2.0)
    tail = _clamp((candidate_end - event.source_time) / 1.0)
    context = min(setup, tail)
    semantic = _SEMANTIC_IMPORTANCE.get(event.type, 0.45) * max(event.semantic_confidence, 0.35)
    return (
        0.42 * _clamp(event.strength)
        + 0.24 * semantic
        + 0.14 * payoff_position
        + 0.10 * _music_opportunity(event, music)
        + 0.10 * context
    )


def select_anchors(
    events: Sequence[PayoffEvent], candidate_start: float, candidate_end: float,
    music: MusicAnalysis, config: PipelineConfig,
) -> tuple[PayoffEvent | None, tuple[PayoffEvent, ...]]:
    eligible = [
        event for event in events
        if candidate_start <= event.source_time <= candidate_end
        and max(event.confidence, event.strength) >= config.weak_anchor_threshold
    ]
    if not eligible:
        return None, ()
    ordered = sorted(
        eligible,
        key=lambda event: (_anchor_utility(event, candidate_start, candidate_end, music), event.source_time, event.event_id),
        reverse=True,
    )
    primary = ordered[0]
    secondary = tuple(
        event for event in ordered[1:config.max_anchor_count_per_candidate]
        if max(event.confidence, event.strength) >= config.weak_anchor_threshold
    )
    return primary, secondary


def context_integrity_score(
    candidate: Candidate, primary: PayoffEvent | None, segments: Sequence[SourceSegment]
) -> float:
    if not segments or any(segment.source != candidate.source_file for segment in segments):
        return 0.0
    if any(segment.source_in < candidate.source_start or segment.source_out > candidate.source_end for segment in segments):
        return 0.0
    if any(following.source_in < previous.source_out for previous, following in zip(segments, segments[1:])):
        return 0.0
    if primary is None:
        return 1.0 if len(segments) == 1 and segments[0].source_in == candidate.source_start and segments[0].source_out == candidate.source_end else 0.65
    anchored = [segment for segment in segments if segment.source_in <= primary.source_time <= segment.source_out]
    if not anchored:
        return 0.0
    anchor_segment = anchored[0]
    setup = _clamp((primary.source_time - anchor_segment.source_in) / 1.5)
    tail = _clamp((anchor_segment.source_out - primary.source_time) / 1.0)
    duration = _clamp(sum(segment.duration for segment in segments) / 3.0)
    fragmentation = 0.15 * max(0, len(segments) - 1)
    return round(_clamp(0.45 + 0.20 * setup + 0.20 * tail + 0.15 * duration - fragmentation), 6)


def _variant_id(candidate: Candidate, segments: Sequence[SourceSegment]) -> str:
    ranges = "|".join(f"{segment.source_in:.3f}-{segment.source_out:.3f}" for segment in segments)
    digest = hashlib.sha1(f"{candidate.candidate_id}|{ranges}".encode("utf-8")).hexdigest()[:12]
    return f"variant-{digest}"


def _continuous_anchor_window(candidate: Candidate, anchor: PayoffEvent) -> SourceSegment:
    target_duration = 6.0
    start = max(candidate.source_start, anchor.source_time - 3.0)
    end = min(candidate.source_end, anchor.source_time + 3.0)
    if end - start < target_duration:
        if start == candidate.source_start:
            end = min(candidate.source_end, start + target_duration)
        else:
            start = max(candidate.source_start, end - target_duration)
    return SourceSegment(candidate.source_file, round(start, 3), round(end, 3), round(end - start, 3))


def build_condensed_variants(
    candidate: Candidate, events: Sequence[PayoffEvent], music: MusicAnalysis, config: PipelineConfig
) -> list[CandidateVariant]:
    primary, secondary = select_anchors(events, candidate.source_start, candidate.source_end, music, config)
    if candidate.duration > 12.0 and primary is not None:
        segments = (_continuous_anchor_window(candidate, primary),)
        rationale = "Continuous anchor-centered range retains setup, action, payoff, and resolution tail."
    else:
        segments = (SourceSegment(candidate.source_file, candidate.source_start, candidate.source_end, candidate.duration),)
        rationale = "Original continuous source range retained because safe condensation was not proven."
    integrity = context_integrity_score(candidate, primary, segments)
    if integrity < config.minimum_context_integrity:
        return []
    included_events = tuple(event for event in events if any(segment.source_in <= event.source_time <= segment.source_out for segment in segments))
    payoff_score = max((event.strength for event in included_events), default=0.0)
    danger = max((event.evidence.get("damage_border_change", 0.0) for event in included_events), default=0.0)
    action_density = _clamp((candidate.combat_intensity + candidate.motion_score + candidate.audio_score) / 3.0)
    features = {
        "motion": candidate.motion_score,
        "visual_novelty": candidate.visual_score,
        "danger_escalation": danger,
    }
    penalties = calculate_penalties(features, included_events, config)
    variant = CandidateVariant(
        variant_id=_variant_id(candidate, segments), parent_candidate_id=candidate.candidate_id,
        source_file=candidate.source_file, source_segments=segments, duration=round(sum(segment.duration for segment in segments), 3),
        human_selection_prior=candidate.human_selection_prior, payoff_score=payoff_score,
        combat_intensity=candidate.combat_intensity, action_density=action_density,
        continuity=candidate.continuity_score, visual_novelty=candidate.visual_score,
        motion=candidate.motion_score, audio_activity=candidate.audio_score, danger_score=danger,
        uniqueness=candidate.uniqueness, final_score=0.0, duplicate_group=candidate.duplicate_group,
        payoff_events=included_events, primary_anchor=primary, secondary_anchors=secondary,
        anchor_event_time=primary.source_time if primary else None,
        anchor_event_type=primary.type if primary else None,
        anchor_event_strength=primary.strength if primary else None,
        anchor_event_confidence=primary.confidence if primary else None,
        context_integrity_score=integrity, penalty_values=penalties,
        source_signature=str(candidate.source_file.resolve(strict=False)), environment_signature=candidate.source_category,
        weapon_or_view_signature="", condense_reason="", rationale=rationale,
    )
    return [variant]
