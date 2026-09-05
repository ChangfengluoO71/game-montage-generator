"""Payoff-anchor selection and context-safe candidate condensation."""

from __future__ import annotations

import hashlib
from typing import Sequence

from .config import PipelineConfig
from .models import Candidate, CandidateVariant, MusicAnalysis, PayoffEvent, SourceSegment
from .ranking import calculate_penalties, verified_kill_events


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
        and event.confidence >= config.weak_anchor_threshold
        and event.strength >= config.weak_anchor_threshold
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
        if event.confidence >= config.weak_anchor_threshold
        and event.strength >= config.weak_anchor_threshold
    )
    return primary, secondary


def context_integrity_score(
    candidate: Candidate, primary: PayoffEvent | None, segments: Sequence[SourceSegment],
    *, allow_clipped_tail: bool = False, minimum_tail: float = 1.0,
) -> float:
    if not segments or any(segment.source != candidate.source_file for segment in segments):
        return 0.0
    if any(segment.source_in < candidate.source_start or segment.source_out > candidate.source_end for segment in segments):
        return 0.0
    if any(following.source_in < previous.source_out for previous, following in zip(segments, segments[1:])):
        return 0.0
    original_range = (
        len(segments) == 1
        and abs(segments[0].source_in - candidate.source_start) <= 0.001
        and abs(segments[0].source_out - candidate.source_end) <= 0.001
    )
    if primary is None:
        return 1.0 if original_range else 0.65
    anchored = [segment for segment in segments if segment.source_in <= primary.source_time <= segment.source_out]
    if not anchored:
        return 0.0
    anchor_segment = anchored[0]
    if original_range:
        return 1.0
    setup = _clamp((primary.source_time - anchor_segment.source_in) / 1.5)
    tail = _clamp((anchor_segment.source_out - primary.source_time) / 1.0)
    if setup < 1.0 or (tail < 1.0 and not (allow_clipped_tail and tail >= minimum_tail)):
        return 0.0
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


def _best_rapid_window(
    candidate: Candidate, events: Sequence[PayoffEvent], config: PipelineConfig
) -> tuple[SourceSegment, tuple[PayoffEvent, ...]] | None:
    """Find the strongest compact kill sequence and keep it as one continuous range."""
    qualifying = sorted(
        {event.event_id: event for event in events if event.type in {"kill", "multikill"}}.values(),
        key=lambda event: (event.source_time, event.event_id),
    )
    minimum = max(2, int(config.rapid_multikill_min_events))
    window = float(config.rapid_multikill_window_s)
    if window <= 0.0 or len(qualifying) < minimum:
        return None

    clusters: list[tuple[tuple[float, float, float, float, str], tuple[PayoffEvent, ...]]] = []
    for start_index, first in enumerate(qualifying):
        cluster = tuple(
            event for event in qualifying[start_index:]
            if event.source_time - first.source_time <= window + 1e-9
        )
        if len(cluster) < minimum:
            continue
        span = cluster[-1].source_time - cluster[0].source_time
        strength = sum(float(event.strength) for event in cluster)
        # Prefer more kills, then stronger evidence, then a tighter sequence; the
        # final timestamp/event id tie-breaks keep cache output deterministic.
        key = (float(len(cluster)), strength, -span, -cluster[-1].source_time, cluster[0].event_id)
        clusters.append((key, cluster))
    if not clusters:
        return None

    _, cluster = max(clusters, key=lambda item: item[0])
    start = max(candidate.source_start, cluster[0].source_time - 1.0)
    end = min(candidate.source_end, cluster[-1].source_time + 1.0)
    if end <= start:
        return None
    rounded_start = round(start, 3)
    rounded_end = round(end, 3)
    segment = SourceSegment(
        candidate.source_file,
        rounded_start,
        rounded_end,
        round(rounded_end - rounded_start, 3),
    )
    return segment, cluster


def _quality_rapid_windows(
    candidate: Candidate, events: Sequence[PayoffEvent], config: PipelineConfig
) -> list[tuple[SourceSegment, tuple[PayoffEvent, ...]]]:
    """Return bounded, non-overlapping verified-kill sequences for V5."""
    qualifying = list(verified_kill_events(events, config))
    minimum = max(2, int(config.rapid_multikill_min_events))
    window = max(0.0, float(config.v5_kill_density_window_s))
    if window <= 0.0 or len(qualifying) < minimum:
        return []

    clusters: list[tuple[tuple[float, float, float, float, str], tuple[PayoffEvent, ...]]] = []
    for start_index, first in enumerate(qualifying):
        cluster = tuple(
            event for event in qualifying[start_index:]
            if event.source_time - first.source_time <= window + 1e-9
        )
        if len(cluster) < minimum:
            continue
        span = cluster[-1].source_time - cluster[0].source_time
        strength = sum(float(event.strength) for event in cluster)
        key = (float(len(cluster)), strength, -span, -cluster[0].source_time, cluster[0].event_id)
        clusters.append((key, cluster))

    selected: list[tuple[SourceSegment, tuple[PayoffEvent, ...]]] = []
    for _, cluster in sorted(clusters, key=lambda item: item[0], reverse=True):
        context = max(0.1, float(config.v5_sequence_context_s))
        start = max(candidate.source_start, cluster[0].source_time - context)
        end = min(candidate.source_end, cluster[-1].source_time + context)
        if end <= start:
            continue
        segment = SourceSegment(
            candidate.source_file,
            round(start, 3),
            round(end, 3),
            round(round(end, 3) - round(start, 3), 3),
        )
        if any(
            segment.source_in < previous.source_out and previous.source_in < segment.source_out
            for previous, _ in selected
        ):
            continue
        selected.append((segment, cluster))
        if len(selected) >= max(1, int(config.v5_max_sequences_per_candidate)):
            break
    return sorted(selected, key=lambda item: (item[0].source_in, item[0].source_out))


def build_condensed_variants(
    candidate: Candidate, events: Sequence[PayoffEvent], music: MusicAnalysis, config: PipelineConfig
) -> list[CandidateVariant]:
    quality_profile = config.v5_profile == "quality"
    verified = verified_kill_events(events, config) if quality_profile else ()
    anchor_events: Sequence[PayoffEvent] = verified if verified else events
    primary, secondary = select_anchors(anchor_events, candidate.source_start, candidate.source_end, music, config)
    original_segments = (SourceSegment(candidate.source_file, candidate.source_start, candidate.source_end, candidate.duration),)
    saved_short_clip = (
        candidate.source_category == "short_clip"
        and candidate.duration <= config.v2_short_clip_max_duration + 0.001
    )

    def make_variant(
        segments: tuple[SourceSegment, ...],
        selected_primary: PayoffEvent | None,
        selected_secondary: tuple[PayoffEvent, ...],
        rationale: str,
        integrity: float,
    ) -> CandidateVariant:
        included_events = tuple(
            event for event in events
            if any(segment.source_in <= event.source_time <= segment.source_out for segment in segments)
        )
        payoff_score = max((event.strength for event in included_events), default=0.0)
        danger = max((event.evidence.get("damage_border_change", 0.0) for event in included_events), default=0.0)
        action_density = _clamp((candidate.combat_intensity + candidate.motion_score + candidate.audio_score) / 3.0)
        features = dict(candidate.feature_runs)
        features.setdefault("motion", candidate.motion_score)
        features.setdefault("visual_novelty", candidate.visual_score)
        features["danger_escalation"] = max(float(features.get("danger_escalation", 0.0)), danger)
        penalties = calculate_penalties(features, included_events, config)
        human_prior = (
            config.v2_short_clip_prior
            if quality_profile and candidate.source_category == "short_clip"
            else candidate.human_selection_prior
        )
        return CandidateVariant(
            variant_id=_variant_id(candidate, segments), parent_candidate_id=candidate.candidate_id,
            source_file=candidate.source_file, source_segments=segments,
            duration=round(sum(segment.duration for segment in segments), 3),
            human_selection_prior=human_prior, payoff_score=payoff_score,
            combat_intensity=candidate.combat_intensity, action_density=action_density,
            continuity=candidate.continuity_score, visual_novelty=candidate.visual_score,
            motion=candidate.motion_score, audio_activity=candidate.audio_score, danger_score=danger,
            uniqueness=candidate.uniqueness, final_score=0.0, duplicate_group=candidate.duplicate_group,
            payoff_events=included_events, primary_anchor=selected_primary, secondary_anchors=selected_secondary,
            anchor_event_time=selected_primary.source_time if selected_primary else None,
            anchor_event_type=selected_primary.type if selected_primary else None,
            anchor_event_strength=selected_primary.strength if selected_primary else None,
            anchor_event_confidence=selected_primary.confidence if selected_primary else None,
            context_integrity_score=integrity, penalty_values=penalties,
            source_signature=str(candidate.source_file.resolve(strict=False)),
            environment_signature=candidate.source_category, weapon_or_view_signature="",
            condense_reason="", rationale=rationale,
        )

    if quality_profile:
        rapid_windows = _quality_rapid_windows(candidate, events, config)
        if rapid_windows:
            variants: list[CandidateVariant] = []
            for segment, cluster in rapid_windows:
                rapid_primary, rapid_secondary = select_anchors(
                    cluster, segment.source_in, segment.source_out, music, config
                )
                if rapid_primary is None:
                    continue
                integrity = context_integrity_score(
                    candidate,
                    rapid_primary,
                    (segment,),
                    allow_clipped_tail=True,
                    minimum_tail=config.v5_min_rapid_context_tail_s,
                )
                if integrity < config.minimum_context_integrity:
                    continue
                included_verified_count = sum(
                    1 for event in verified
                    if segment.source_in <= event.source_time <= segment.source_out
                )
                variants.append(make_variant(
                    (segment,), rapid_primary, rapid_secondary,
                    f"Verified kills={included_verified_count} retained as one continuous rapid sequence; "
                    "trimmed non-event reload and cover gaps while preserving the transfer and resolution tail.",
                    integrity,
                ))
            if variants:
                return variants

    if saved_short_clip:
        segments = original_segments
        rationale = (
            "Complete human-saved short clip retained as one continuous sequence; "
            "no beat-driven fragmentation is applied."
        )
    elif candidate.duration > 12.0:
        rapid = _best_rapid_window(candidate, events, config)
        if rapid is not None:
            rapid_segment, rapid_events = rapid
            rapid_primary, rapid_secondary = select_anchors(
                rapid_events, rapid_segment.source_in, rapid_segment.source_out, music, config
            )
            if rapid_primary is not None and context_integrity_score(candidate, rapid_primary, (rapid_segment,)) >= config.minimum_context_integrity:
                primary, secondary = rapid_primary, rapid_secondary
                segments = (rapid_segment,)
                rationale = (
                    "Tight continuous rapid-kill sequence retains the transfer, successive kills, "
                    "and short resolution tail."
                )
            else:
                segments = original_segments
                rationale = "Original continuous source range retained because the rapid-kill context could not be proven safe."
        elif primary is not None:
            condensed_segments = (_continuous_anchor_window(candidate, primary),)
            if context_integrity_score(candidate, primary, condensed_segments) >= config.minimum_context_integrity:
                segments = condensed_segments
                rationale = "Continuous anchor-centered range retains setup, action, payoff, and resolution tail."
            else:
                segments = original_segments
                rationale = "Original continuous source range retained because the boundary anchor cannot preserve setup and payoff tail."
        else:
            segments = original_segments
            rationale = "Original continuous source range retained because safe condensation was not proven."
    else:
        segments = original_segments
        rationale = "Original continuous source range retained because safe condensation was not proven."
    integrity = context_integrity_score(candidate, primary, segments)
    if integrity < config.minimum_context_integrity:
        return []
    return [make_variant(segments, primary, secondary, rationale, integrity)]
