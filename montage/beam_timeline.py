"""Bounded payoff-aware beam search for the V2 preview edit."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from .ranking import verified_kill_events


_CONDENSE_GAP_REASONS = frozenset({
    "downtime_removed",
    "phrase_boundary",
    "bar_boundary",
    "action_phase_change",
    "substantial_spatial_change",
    "substantial_state_change",
})


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
    target_duration: float = 0.0


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


def _preview_music_interval(
    music: MusicAnalysis, baseline_edit: EditDecisionList, config: PipelineConfig
) -> tuple[float, float, str]:
    """Return the interval that must cover the complete selected edit."""
    if (
        config.v2_music_window_policy == "representative"
        and music.preview_music_in is not None
        and music.preview_music_out is not None
        and float(music.preview_music_out) > float(music.preview_music_in)
    ):
        return (
            float(music.preview_music_in),
            float(music.preview_music_out),
            music.preview_reason or "representative music structure window",
        )
    return (
        float(baseline_edit.music_in),
        float(baseline_edit.music_out),
        "retained locked V1 baseline interval",
    )


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
    baseline_music_in: float | None = None,
) -> V2EditShot:
    timeline_in = state.elapsed
    hint = (config.baseline_music_in if baseline_music_in is None else baseline_music_in) + timeline_in
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
    section = _section(timeline_in, total_duration)
    base_rationale = variant.rationale.strip() or "setup action payoff tail"
    anchor_label = event_type or "none"
    rationale = (
        f"{base_rationale}; placement section={section}; anchor={anchor_label}; "
        f"music_target={music_target:.3f}s; event_sync={event_offset:+.3f}s; "
        f"cut_sync={cut_offset:+.3f}s"
    )
    return V2EditShot(
        source=variant.source_file,
        source_in=variant.source_segments[0].source_in,
        source_out=variant.source_segments[0].source_out,
        duration=variant.duration,
        candidate_score=variant.final_score,
        duplicate_group=variant.duplicate_group or f"auto-{variant.variant_id}",
        timeline_in=timeline_in,
        timeline_out=timeline_in + variant.duration,
        transition=transition.transition,
        music_target=music_target,
        music_event_type=event_type,
        sync_offset=cut_offset,
        rationale=rationale,
        section=section,
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
        source_signature=variant.source_signature,
        environment_signature=variant.environment_signature,
        weapon_or_view_signature=variant.weapon_or_view_signature,
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
        source_signature=shot.source_signature, environment_signature=shot.environment_signature,
        weapon_or_view_signature=shot.weapon_or_view_signature,
        condense_reason=shot.condense_reason, rationale=shot.rationale,
    )


def _expansion_score(variant: CandidateVariant, state: BeamState, shot: V2EditShot,
                     music: MusicAnalysis, config: PipelineConfig,
                     hero_quality_floor: float = 0.0) -> float:
    total = max(state.target_duration or config.preview_max_duration, state.elapsed + variant.duration, 1.0)
    relative = (state.elapsed + variant.duration) / total
    quality = _clamp(variant.final_score)
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
        + (0.12 if relative >= 0.85 and _is_hero(shot) and variant.final_score >= hero_quality_floor else 0.0)
    )


def expand_beam(state: BeamState, variants: Sequence[CandidateVariant], music: MusicAnalysis,
                config: PipelineConfig, *, baseline_music_in: float | None = None,
                expansion_budget: int | None = None, hero_quality_floor: float = 0.0,
                expansion_counter: list[int] | None = None) -> list[BeamState]:
    """Expand one state with a bounded, deterministic set of unused variants.

    ``expansion_budget`` caps this call. When ``expansion_counter`` is shared
    by the caller, the cap applies to the whole search rather than each state.
    """
    options: list[BeamState] = []
    for variant in sorted(
        variants,
        key=lambda item: (
            -item.final_score,
            -float(item.score_components.get("verified_kill_count", 0.0)),
            -float(item.score_components.get("kill_density", 0.0)),
            -item.human_selection_prior,
            item.variant_id,
        ),
    ):
        limit = config.beam_max_expansions if expansion_budget is None else expansion_budget
        if expansion_counter is not None:
            limit -= expansion_counter[0]
        if len(options) >= max(0, limit):
            break
        group = variant.duplicate_group or f"auto-{variant.variant_id}"
        if group in state.used_duplicate_groups:
            continue
        if any(shot.variant_id == variant.variant_id for shot in state.shots):
            continue
        if (
            config.v5_min_verified_kills_per_shot > 0
            and len(verified_kill_events(variant.payoff_events, config)) < config.v5_min_verified_kills_per_shot
        ):
            continue
        if state.elapsed + variant.duration > config.preview_max_duration + 1e-6:
            continue
        long_run = len(state.shots) >= 2 and all(
            shot.duration > config.preferred_macro_duration[1]
            and not _is_preserved_short_shot(shot, config)
            for shot in state.shots[-2:]
        )
        if (
            variant.duration > config.preferred_macro_duration[1]
            and long_run
            and not _is_preserved_short_variant(variant, config)
        ):
            continue
        if not _duration_allowed(
            variant.duration,
            variant.anchor_event_type,
            variant.condense_reason,
            variant.context_integrity_score,
            config,
            preserved_short_clip=_is_preserved_short_variant(variant, config),
        ):
            continue
        if expansion_counter is not None:
            expansion_counter[0] += 1
        shot = _make_shot(variant, state, music, config, state.target_duration or config.preview_max_duration, baseline_music_in)
        score = _expansion_score(variant, state, shot, music, config, hero_quality_floor)
        sources = (state.recent_sources + (variant.source_signature,))[-max(1, config.recent_source_window):]
        environments = (state.recent_environments + (variant.environment_signature,))[-max(1, config.recent_environment_window):]
        options.append(BeamState(
            shots=state.shots + (shot,), elapsed=state.elapsed + variant.duration,
            score=state.score + score, used_duplicate_groups=state.used_duplicate_groups | {group},
            recent_sources=sources, recent_environments=environments,
            energy_fit=state.energy_fit + abs(_music_energy(music, float(shot.music_target or 0.0)) - timeline_energy_target((state.elapsed + variant.duration) / max(state.target_duration or config.preview_max_duration, 1.0))),
            context_total=state.context_total + shot.context_integrity_score,
            event_sync_offsets=state.event_sync_offsets + (shot.event_sync_offset,),
            cut_sync_offsets=state.cut_sync_offsets + (shot.cut_sync_offset,),
            expansions=state.expansions + 1,
        ))
    bounded = sorted(
        options,
        key=lambda item: (
            -item.score,
            -sum(float(shot.candidate_score) for shot in item.shots[-1:]),
            item.shots[-1].variant_id,
        ),
    )[:max(1, config.beam_width)]
    # Reserve a beam lane for a quality-comparable Hero candidate so the explicit
    # closing-shot comparison can still be made after later expansions.
    if bounded:
        ordinary_quality = max((shot.candidate_score for item in options for shot in item.shots[-1:] if not _is_hero(shot)), default=0.0)
        hero_options = [item for item in options if _is_hero(item.shots[-1]) and
                        _closing_quality(item.shots[-1]) >= ordinary_quality - config.hero_quality_margin]
        if hero_options and not any(_is_hero(item.shots[-1]) for item in bounded):
            bounded[-1] = max(hero_options, key=lambda item: item.score)
    return bounded


def build_v2_preview_edit(variants: Sequence[CandidateVariant], music: MusicAnalysis,
                          baseline_edit: EditDecisionList, config: PipelineConfig) -> V2EditDecisionList:
    """Build the best bounded preview state and write its EDL before rendering."""
    if not variants:
        raise ValueError("V2 preview requires candidate variants")
    selected_music_in, selected_music_out, music_reason = _preview_music_interval(music, baseline_edit, config)
    music_duration = max(0.0, selected_music_out - selected_music_in)
    target_max = min(config.preview_max_duration, music_duration) if music_duration else config.preview_max_duration
    states = [BeamState(target_duration=target_max)]
    completed: list[BeamState] = []
    ordinary_quality = max((variant.final_score for variant in variants if variant.anchor_event_type not in {"hero_play", "hero", "finale"}), default=0.0)
    hero_quality_floor = ordinary_quality - config.hero_quality_margin
    expansion_counter = [0]
    for _ in range(max(1, config.v2_max_shots)):
        next_states: list[BeamState] = []
        for state in states:
            if len(state.shots) >= config.v2_min_shots and config.preview_min_duration <= state.elapsed <= target_max:
                completed.append(state)
            if len(state.shots) < config.v2_max_shots and state.elapsed < target_max - 1e-6:
                next_states.extend(expand_beam(state, variants, music, config, baseline_music_in=selected_music_in,
                                               hero_quality_floor=hero_quality_floor,
                                               expansion_budget=config.beam_max_expansions,
                                               expansion_counter=expansion_counter))
                if expansion_counter[0] >= max(0, config.beam_max_expansions):
                    break
        if not next_states:
            break
        states = sorted(
            next_states,
            key=lambda item: (-item.score, -item.elapsed, tuple(shot.variant_id for shot in item.shots)),
        )[:max(1, config.beam_width)]
    completed.extend(
        state for state in states
        if len(state.shots) >= config.v2_min_shots and config.preview_min_duration <= state.elapsed <= target_max
    )
    if not completed:
        raise ValueError(
            f"unable to build a {config.v2_min_shots}-{config.v2_max_shots} shot V2 preview within "
            f"{config.preview_min_duration:g}-{config.preview_max_duration:g} seconds"
        )
    best = max(completed, key=lambda state: (state.score, sum(shot.context_integrity_score for shot in state.shots),
                                             -abs(state.elapsed - target_max)))
    non_hero = [state for state in completed if not _is_hero(state.shots[-1])]
    if non_hero:
        best_non_hero = max(non_hero, key=lambda state: state.score)
        comparable_heroes = [state for state in completed if _is_hero(state.shots[-1]) and
                             _closing_quality(state.shots[-1]) >= _closing_quality(best_non_hero.shots[-1]) - config.hero_quality_margin]
        if comparable_heroes:
            best = max(comparable_heroes, key=lambda state: (state.score, _closing_quality(state.shots[-1])))
    shots = tuple(replace(shot, section=_section(shot.timeline_in, best.elapsed, shot is best.shots[-1])) for shot in best.shots)
    edit = V2EditDecisionList(
        kind="preview_v5" if config.v5_profile == "quality" else "preview_v2", music_source=baseline_edit.music_source,
        baseline_music_in=baseline_edit.music_in, baseline_music_out=baseline_edit.music_out,
        music_in=selected_music_in, music_out=selected_music_out,
        duration=best.elapsed, music_reason=music_reason,
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


def _is_hero(shot: V2EditShot) -> bool:
    return shot.anchor_event_type in {"hero_play", "hero", "finale"}


def _closing_quality(shot: V2EditShot) -> float:
    """Quality used for the explicit hero-vs-ordinary closing-shot comparison."""
    return _clamp(shot.candidate_score)


def _duration_allowed(duration: float, anchor_type: str | None, condense_reason: str,
                      context_integrity: float, config: PipelineConfig,
                      *, preserved_short_clip: bool = False) -> bool:
    """Apply preferred macro bounds, with an explicit context-safe long-shot exception."""
    low, high = config.preferred_macro_duration
    if duration < low - 0.001:
        return False
    if preserved_short_clip:
        return duration <= config.v2_short_clip_max_duration + 0.001
    if duration > config.hero_max_duration + 0.001:
        return False
    if duration <= high + 0.001:
        return True
    is_hero = anchor_type in {"hero_play", "hero", "finale"}
    return is_hero or (bool(condense_reason.strip()) and context_integrity >= config.minimum_context_integrity)


def validate_v2_edit(edit: V2EditDecisionList, config: PipelineConfig) -> None:
    """Raise ValueError for edits that cannot safely proceed to a future renderer."""
    if not config.v2_min_shots <= len(edit.shots) <= config.v2_max_shots:
        raise ValueError(f"V2 edit must contain {config.v2_min_shots}-{config.v2_max_shots} macro shots")
    if not config.preview_min_duration <= edit.duration <= config.preview_max_duration:
        raise ValueError(
            f"V2 edit duration must be in the {config.preview_min_duration:g}-"
            f"{config.preview_max_duration:g} second range"
        )
    if edit.music_out <= edit.music_in or edit.music_out - edit.music_in + 0.01 < edit.duration:
        raise ValueError("V2 music range is invalid")
    groups: set[str] = set()
    long_count = 0
    previous_out = 0.0
    for index, shot in enumerate(edit.shots):
        if shot.duplicate_group in groups:
            raise ValueError("duplicate group is repeated in V2 edit")
        groups.add(shot.duplicate_group or f"auto-{shot.variant_id}")
        if shot.source_in < 0 or shot.source_out <= shot.source_in or shot.duration <= 0:
            raise ValueError("source range is invalid")
        if (index == 0 and abs(shot.timeline_in) > 0.001) or (index and abs(shot.timeline_in - previous_out) > 0.001) or abs(shot.timeline_out - shot.timeline_in - shot.duration) > 0.01:
            raise ValueError("timeline range is invalid")
        resolved_source = shot.source.resolve(strict=False)
        if not is_within(resolved_source, config.raw_dir) or not shot.source.is_file():
            raise ValueError("source path must be an existing regular file below raw")
        if not shot.source_segments:
            raise ValueError("source segments are required")
        if any(segment.source != shot.source for segment in shot.source_segments):
            raise ValueError("source segment path does not match shot source")
        if any(segment.source_in < 0 or segment.source_out <= segment.source_in or segment.duration <= 0 for segment in shot.source_segments):
            raise ValueError("source range is invalid")
        if any(not is_within(segment.source.resolve(strict=False), config.raw_dir) or not segment.source.is_file() for segment in shot.source_segments):
            raise ValueError("source segment source must be an existing regular file below raw")
        if abs(shot.source_in - shot.source_segments[0].source_in) > 0.001 or abs(shot.source_out - shot.source_segments[0].source_out) > 0.001:
            raise ValueError("shot source range does not match first source segment")
        if len(shot.source_segments) == 2 and shot.condense_reason not in _CONDENSE_GAP_REASONS:
            raise ValueError("two source segments require an explicit valid condense_reason")
        if len(shot.source_segments) == 1 and shot.condense_reason:
            raise ValueError("continuous source segment cannot have a condense_reason")
        for previous_segment, next_segment in zip(shot.source_segments, shot.source_segments[1:]):
            gap = next_segment.source_in - previous_segment.source_out
            if gap < -0.001 or next_segment.source_in < previous_segment.source_in - 0.001:
                raise ValueError("source segments contain an overlap or bad ordering")
            if gap > 0.001 and not (len(shot.source_segments) == 2 and shot.condense_reason in _CONDENSE_GAP_REASONS):
                raise ValueError("source segments contain an unexplained gap")
        if abs(sum(segment.duration for segment in shot.source_segments) - shot.duration) > 0.01:
            raise ValueError("source segment durations do not match shot duration")
        if not shot.rationale.strip():
            raise ValueError("rationale is required")
        if shot.primary_anchor is None:
            raise ValueError("payoff anchor is required")
        if config.v5_min_verified_kills_per_shot > 0:
            verified = {event.event_id for event in verified_kill_events(shot.payoff_events, config)}
            if (
                len(verified) < config.v5_min_verified_kills_per_shot
                or shot.primary_anchor.event_id not in verified
            ):
                raise ValueError("verified kill anchor is required for the quality profile")
        if shot.transition != "hard_cut":
            raise ValueError("V2 selection must use hard cuts")
        if not _duration_allowed(
            shot.duration,
            shot.anchor_event_type,
            shot.condense_reason,
            shot.context_integrity_score,
            config,
            preserved_short_clip=_is_preserved_short_shot(shot, config),
        ):
            raise ValueError("shot duration is outside preferred macro or hero limits")
        long_count = long_count + 1 if (
            shot.duration > config.preferred_macro_duration[1]
            and not _is_preserved_short_shot(shot, config)
        ) else 0
        if long_count >= 3:
            raise ValueError("three long segments are not permitted")
        previous_out = shot.timeline_out
    if abs(previous_out - edit.duration) > 0.01:
        raise ValueError("timeline must end at edit duration")


def _is_preserved_short_variant(variant: CandidateVariant, config: PipelineConfig) -> bool:
    return (
        variant.environment_signature == "short_clip"
        and len(variant.source_segments) == 1
        and not variant.condense_reason
        and variant.duration <= config.v2_short_clip_max_duration + 0.001
    )


def _is_preserved_short_shot(shot: V2EditShot, config: PipelineConfig) -> bool:
    return (
        shot.environment_signature == "short_clip"
        and len(shot.source_segments) == 1
        and not shot.condense_reason
        and shot.duration <= config.v2_short_clip_max_duration + 0.001
    )
