"""Bounded payoff-aware beam search for the V2 preview edit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from .cache import atomic_write_json, atomic_write_bytes
from .config import PipelineConfig, is_within
from .models import (
    CandidateVariant,
    EditDecisionList,
    MusicAnalysis,
    V2EditDecisionList,
    V2EditShot,
)
from .transitions import choose_anchor_music_target, choose_v2_transition


@dataclass(frozen=True)
class BeamState:
    shots: tuple[V2EditShot, ...] = ()
    elapsed: float = 0.0
    score: float = 0.0
    used_duplicate_groups: frozenset[str] = frozenset()
    recent_sources: tuple[str, ...] = ()
    recent_environments: tuple[str, ...] = ()
    energy_fit: float = 0.0
    context_total: float = 0.0
    event_sync_offsets: tuple[float, ...] = ()
    cut_sync_offsets: tuple[float, ...] = ()
    expansions: int = 0


def timeline_energy_target(relative_time: float) -> float:
    """Return the intended normalized energy for a preview timeline position."""
    position = max(0.0, min(1.0, float(relative_time)))
    if position < 0.20:
        return 0.30 + position / 0.20 * 0.20
    if position < 0.45:
        return 0.50 + (position - 0.20) / 0.25 * 0.18
    if position < 0.60:
        return 0.68 + (position - 0.45) / 0.15 * 0.14
    if position < 0.85:
        return 0.82 + (position - 0.60) / 0.25 * 0.08
    return 0.90 + (position - 0.85) / 0.15 * 0.10


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _music_energy(music: MusicAnalysis, timestamp: float) -> float:
    if not music.energy_times or not music.rms:
        return 0.5
    pairs = list(zip(music.energy_times, music.rms))
    nearest = min(pairs, key=lambda pair: abs(float(pair[0]) - timestamp))
    high = max(float(value) for _, value in pairs)
    low = min(float(value) for _, value in pairs)
    return 0.5 if high <= low else _clamp((float(nearest[1]) - low) / (high - low))


def _section(elapsed: float, total: float, is_final: bool = False) -> str:
    position = 0.0 if total <= 0 else elapsed / total
    if is_final or position >= 0.85:
        return "finale"
    if position < 0.20:
        return "build"
    if position < 0.45:
        return "rise"
    if position < 0.60:
        return "release"
    return "escalation"


def _anchor_timeline_time(variant: CandidateVariant, timeline_in: float) -> float | None:
    anchor = variant.primary_anchor
    if anchor is None:
        return None
    for segment in variant.source_segments:
        if segment.source_in <= anchor.source_time <= segment.source_out:
            offset = sum(item.duration for item in variant.source_segments if item.source_out <= segment.source_in)
            return timeline_in + offset + anchor.source_time - segment.source_in
    return timeline_in + min(variant.duration, max(0.0, anchor.source_time - variant.source_segments[0].source_in))


def _make_shot(
    variant: CandidateVariant,
    state: BeamState,
    music: MusicAnalysis,
    config: PipelineConfig,
    total_duration: float,
) -> V2EditShot:
    timeline_in = state.elapsed
    hint = config.baseline_music_in + timeline_in
    anchor = variant.primary_anchor
    if anchor is not None:
        placement = choose_anchor_music_target(anchor, music, hint, config)
        music_target = placement.target_music_time
        event_offset = placement.event_sync_offset
        context = placement.context_integrity_score
        event_type = placement.event_type
    else:
        music_target = hint
        event_offset = 0.0
        context = variant.context_integrity_score
        event_type = None
    transition = choose_v2_transition(
        variant if not state.shots else _variant_for_shot(state.shots[-1]), variant, config
    )
    cut_offset = float(music_target - hint)
    return V2EditShot(
        source=variant.source_file,
        source_in=variant.source_segments[0].source_in,
        source_out=variant.source_segments[-1].source_out,
        duration=variant.duration,
        candidate_score=variant.final_score,
        duplicate_group=variant.duplicate_group or f"auto-{variant.variant_id}",
        timeline_in=timeline_in,
        timeline_out=timeline_in + variant.duration,
        transition=transition.transition,
        music_target=music_target,
        music_event_type=event_type,
        sync_offset=cut_offset,
        rationale=variant.rationale or "setup action payoff tail",
        section=_section(timeline_in, total_duration),
        source_duration=variant.duration,
        source_segments=variant.source_segments,
        parent_candidate_id=variant.parent_candidate_id,
        variant_id=variant.variant_id,
        payoff_events=variant.payoff_events,
        anchor_event_time=variant.anchor_event_time,
        anchor_event_type=variant.anchor_event_type,
        anchor_event_strength=variant.anchor_event_strength,
        anchor_event_confidence=variant.anchor_event_confidence,
        primary_anchor=variant.primary_anchor,
        secondary_anchors=variant.secondary_anchors,
        context_integrity_score=context,
        condense_reason=variant.condense_reason,
        event_timeline=_anchor_timeline_time(variant, timeline_in),
        event_sync_offset=event_offset,
        cut_sync_offset=cut_offset,
        transition_compatibility_score=transition.compatibility_score,
        impact_cut=transition.impact_cut,
        audio_j_cut_ms=transition.audio_j_cut_ms,
        audio_l_cut_ms=transition.audio_l_cut_ms,
    )


def _variant_for_shot(shot: V2EditShot) -> CandidateVariant:
    # Transition descriptors only consume the public variant boundary metadata.
    return CandidateVariant(
        variant_id=shot.variant_id, parent_candidate_id=shot.parent_candidate_id,
        source_file=shot.source, source_segments=shot.source_segments, duration=shot.duration,
        human_selection_prior=shot.candidate_score, payoff_score=shot.candidate_score,
        combat_intensity=shot.candidate_score, action_density=shot.candidate_score,
        continuity=shot.context_integrity_score, visual_novelty=shot.candidate_score,
        motion=shot.candidate_score, audio_activity=shot.candidate_score,
        danger_score=shot.candidate_score, uniqueness=1.0, final_score=shot.candidate_score,
        duplicate_group=shot.duplicate_group, payoff_events=shot.payoff_events,
        primary_anchor=shot.primary_anchor, secondary_anchors=shot.secondary_anchors,
        anchor_event_time=shot.anchor_event_time, anchor_event_type=shot.anchor_event_type,
        anchor_event_strength=shot.anchor_event_strength, anchor_event_confidence=shot.anchor_event_confidence,
        context_integrity_score=shot.context_integrity_score, penalty_values={},
        source_signature="", environment_signature="", weapon_or_view_signature="",
        condense_reason=shot.condense_reason, rationale=shot.rationale,
    )


def _expansion_score(variant: CandidateVariant, state: BeamState, shot: V2EditShot,
                     music: MusicAnalysis, config: PipelineConfig) -> float:
    total = max(config.preview_max_duration, state.elapsed + variant.duration, 1.0)
    relative = (state.elapsed + variant.duration) / total
    quality = _clamp(variant.final_score + variant.rapid_multikill_bonus)
    music_fit = 1.0 - abs(_music_energy(music, float(shot.music_target or 0.0)) - timeline_energy_target(relative))
    diversity = 1.0
    if variant.source_signature in state.recent_sources:
        diversity -= 0.5
    if variant.environment_signature in state.recent_environments:
        diversity -= 0.5
    penalty = sum(float(variant.penalty_values.get(key, 0.0)) for key in (
        "downtime_penalty", "no_payoff_penalty", "same_source_recent_penalty", "same_environment_recent_penalty"
    ))
    sync_penalty = min(1.0, abs(shot.event_sync_offset) / 2.0 + abs(shot.cut_sync_offset) / 2.0)
    weights = config.beam_weights
    return (
        float(weights.get("highlight_quality", 0.35)) * quality
        + float(weights.get("music_fit", 0.25)) * _clamp(music_fit)
        + float(weights.get("transition_compatibility", 0.15)) * shot.transition_compatibility_score
        + float(weights.get("diversity", 0.15)) * _clamp(diversity)
        + float(weights.get("energy_curve_fit", 0.10)) * _clamp(music_fit)
        - 0.10 * (1.0 - shot.context_integrity_score)
        - 0.08 * penalty
        - 0.04 * sync_penalty
        + (0.12 if relative >= 0.85 and shot.anchor_event_type in {"hero_play", "hero", "finale"} else 0.0)
    )


def expand_beam(state: BeamState, variants: Sequence[CandidateVariant], music: MusicAnalysis,
                config: PipelineConfig) -> list[BeamState]:
    """Expand a state with a bounded, deterministic set of unused variants."""
    options: list[BeamState] = []
    for variant in sorted(variants, key=lambda item: (-item.final_score, item.variant_id)):
        if len(options) >= max(0, config.beam_max_expansions):
            break
        group = variant.duplicate_group or f"auto-{variant.variant_id}"
        if group in state.used_duplicate_groups:
            continue
        if any(shot.variant_id == variant.variant_id for shot in state.shots):
            continue
        if state.elapsed + variant.duration > config.preview_max_duration + 1e-6:
            continue
        long_run = len(state.shots) >= 2 and all(shot.duration > config.preferred_macro_duration[1] for shot in state.shots[-2:])
        if variant.duration > config.preferred_macro_duration[1] and long_run:
            continue
        shot = _make_shot(variant, state, music, config, config.preview_max_duration)
        score = _expansion_score(variant, state, shot, music, config)
        sources = (state.recent_sources + (variant.source_signature,))[-max(1, config.recent_source_window):]
        environments = (state.recent_environments + (variant.environment_signature,))[-max(1, config.recent_environment_window):]
        options.append(BeamState(
            shots=state.shots + (shot,), elapsed=state.elapsed + variant.duration,
            score=state.score + score, used_duplicate_groups=state.used_duplicate_groups | {group},
            recent_sources=sources, recent_environments=environments,
            energy_fit=state.energy_fit + abs(_music_energy(music, float(shot.music_target or 0.0)) - timeline_energy_target((state.elapsed + variant.duration) / max(config.preview_max_duration, 1.0))),
            context_total=state.context_total + shot.context_integrity_score,
            event_sync_offsets=state.event_sync_offsets + (shot.event_sync_offset,),
            cut_sync_offsets=state.cut_sync_offsets + (shot.cut_sync_offset,),
            expansions=state.expansions + 1,
        ))
    return sorted(options, key=lambda item: item.score, reverse=True)[:max(1, config.beam_width)]


def build_v2_preview_edit(variants: Sequence[CandidateVariant], music: MusicAnalysis,
                          baseline_edit: EditDecisionList, config: PipelineConfig) -> V2EditDecisionList:
    """Build the best bounded preview state and write its EDL before rendering."""
    if not variants:
        raise ValueError("V2 preview requires candidate variants")
    states = [BeamState()]
    music_duration = max(0.0, baseline_edit.music_out - baseline_edit.music_in)
    target_max = min(config.preview_max_duration, music_duration) if music_duration else config.preview_max_duration
    completed: list[BeamState] = []
    for _ in range(14):
        next_states: list[BeamState] = []
        for state in states:
            if len(state.shots) >= 8 and config.preview_min_duration <= state.elapsed <= target_max:
                completed.append(state)
            if len(state.shots) < 14 and state.elapsed < config.preview_max_duration - 1e-6:
                next_states.extend(expand_beam(state, variants, music, config))
        if not next_states:
            break
        states = sorted(next_states, key=lambda item: item.score, reverse=True)[:max(1, config.beam_width)]
    completed.extend(state for state in states if len(state.shots) >= 8 and config.preview_min_duration <= state.elapsed <= target_max)
    if not completed:
        raise ValueError("unable to build an 8-14 shot V2 preview within 45-60 seconds")
    best = max(completed, key=lambda state: (
        state.score + (best_finale_quality(state) if state.shots else 0.0),
        sum(shot.context_integrity_score for shot in state.shots),
        -abs(state.elapsed - target_max),
    ))
    shots = tuple(replace(shot, section=_section(shot.timeline_in, best.elapsed, shot is best.shots[-1])) for shot in best.shots)
    edit = V2EditDecisionList(
        kind="preview_v2", music_source=baseline_edit.music_source,
        baseline_music_in=baseline_edit.music_in, baseline_music_out=baseline_edit.music_out,
        music_in=baseline_edit.music_in, music_out=baseline_edit.music_out,
        duration=best.elapsed, music_reason="retained locked V1 baseline interval",
        shots=shots,
    )
    validate_v2_edit(edit, config)
    atomic_write_json(config.preview_v2_edit_path, edit.to_dict())
    timeline = "\n".join(
        f"{shot.timeline_in:07.3f}-{shot.timeline_out:07.3f} {shot.section:10s} {shot.variant_id} "
        f"{shot.anchor_event_type or 'no-payoff'} event={shot.event_sync_offset:+.3f}s cut={shot.cut_sync_offset:+.3f}s"
        for shot in shots
    ) + "\n"
    atomic_write_bytes(config.preview_v2_timeline_path, timeline.encode("utf-8"))
    return edit


def best_finale_quality(state: BeamState) -> float:
    final = state.shots[-1]
    anchor_type = final.anchor_event_type or ""
    hero_bonus = 0.12 if anchor_type in {"hero_play", "hero", "finale"} else 0.0
    return (final.candidate_score + (final.primary_anchor.strength if final.primary_anchor else 0.0)) * 0.05 + hero_bonus


def validate_v2_edit(edit: V2EditDecisionList, config: PipelineConfig) -> None:
    """Raise ValueError for edits that cannot safely proceed to a future renderer."""
    if not 8 <= len(edit.shots) <= 14:
        raise ValueError("V2 edit must contain 8-14 macro shots")
    if not config.preview_min_duration <= edit.duration <= config.preview_max_duration:
        raise ValueError("V2 edit duration must be in the 45-60 second range")
    if edit.music_out <= edit.music_in or edit.music_out - edit.music_in + 0.01 < edit.duration:
        raise ValueError("V2 music range is invalid")
    groups: set[str] = set()
    long_count = 0
    previous_out = 0.0
    for shot in edit.shots:
        if shot.duplicate_group in groups:
            raise ValueError("duplicate group is repeated in V2 edit")
        groups.add(shot.duplicate_group or f"auto-{shot.variant_id}")
        if shot.source_in < 0 or shot.source_out <= shot.source_in or shot.duration <= 0:
            raise ValueError("source range is invalid")
        if shot.timeline_in < previous_out - 0.001 or abs(shot.timeline_out - shot.timeline_in - shot.duration) > 0.01:
            raise ValueError("timeline range is invalid")
        if not is_within(shot.source, config.raw_dir):
            raise ValueError("source path must remain below raw")
        if not shot.source_segments:
            raise ValueError("source segments are required")
        if any(segment.source != shot.source for segment in shot.source_segments):
            raise ValueError("source segment path does not match shot source")
        if any(segment.source_in < 0 or segment.source_out <= segment.source_in or segment.duration <= 0 for segment in shot.source_segments):
            raise ValueError("source range is invalid")
        if any(not is_within(segment.source, config.raw_dir) for segment in shot.source_segments):
            raise ValueError("source path must remain below raw")
        if abs(sum(segment.duration for segment in shot.source_segments) - shot.duration) > 0.01:
            raise ValueError("source segment durations do not match shot duration")
        if not shot.rationale.strip():
            raise ValueError("rationale is required")
        if shot.primary_anchor is None:
            raise ValueError("payoff anchor is required")
        if shot.transition != "hard_cut":
            raise ValueError("V2 selection must use hard cuts")
        long_count = long_count + 1 if shot.duration > config.preferred_macro_duration[1] else 0
        if long_count >= 3:
            raise ValueError("three long segments are not permitted")
        previous_out = shot.timeline_out
