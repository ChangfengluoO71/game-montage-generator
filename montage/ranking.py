"""Configurable candidate scoring."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .config import PipelineConfig
from .models import Candidate, CandidateVariant, PayoffEvent


RAPID_MULTIKILL_MAX_BONUS = 0.18


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def verified_kill_events(events: Sequence[PayoffEvent], config: PipelineConfig) -> tuple[PayoffEvent, ...]:
    """Return only kill labels with the corroboration required by the quality profile.

    A generic combat climax or a lone kill-feed change is not sufficient.  The
    event must retain both the kill-feed and reward-ROI evidence so the V5 gate
    cannot be satisfied by a noisy visual transient.
    """
    unique: dict[str, PayoffEvent] = {}
    for event in events:
        if event.type not in {"kill", "multikill"}:
            continue
        if event.semantic_confidence < config.v5_min_kill_semantic_confidence:
            continue
        if event.evidence.get("killfeed_change", 0.0) < config.v5_min_killfeed_evidence:
            continue
        if event.evidence.get("reward_roi_change", 0.0) < config.v5_min_reward_evidence:
            continue
        unique[event.event_id] = event
    return tuple(sorted(unique.values(), key=lambda event: (event.source_time, event.event_id)))


def max_verified_kills_in_window(
    events: Sequence[PayoffEvent], window_s: float, config: PipelineConfig
) -> int:
    """Return the largest corroborated kill count inside one compact time window."""
    qualifying = verified_kill_events(events, config)
    window = max(0.0, float(window_s))
    if not qualifying or window <= 0.0:
        return 0
    best = 0
    for index, first in enumerate(qualifying):
        count = sum(
            1 for event in qualifying[index:]
            if event.source_time - first.source_time <= window + 1e-9
        )
        best = max(best, count)
    return best


def score_candidates(candidates: Sequence[Candidate], config: PipelineConfig) -> list[Candidate]:
    weights = config.weights
    scored: list[Candidate] = []
    for candidate in candidates:
        values = {
            "human_selection_prior": _clamp(candidate.human_selection_score),
            "combat_intensity": _clamp(candidate.combat_intensity),
            "motion": _clamp(candidate.motion_score),
            "audio_activity": _clamp(candidate.audio_score),
            "visual_activity": _clamp(candidate.visual_score),
            "continuity": _clamp(candidate.continuity_score),
            "uniqueness": _clamp(candidate.uniqueness),
        }
        final_score = sum(float(weights.get(key, 0.0)) * value for key, value in values.items())
        scored.append(replace(candidate, final_score=round(final_score, 6)))
    return scored


_PENALTY_FEATURES = {
    "stationary_ads_penalty": "stationary_ads",
    "same_view_penalty": "same_view",
    "downtime_penalty": "downtime",
    "no_payoff_penalty": "no_payoff",
    "repetitive_fire_penalty": "repetitive_fire",
    "same_source_recent_penalty": "same_source_recent",
    "same_environment_recent_penalty": "same_environment_recent",
}


def _feature(features: dict[str, float], *names: str) -> float:
    return _clamp(next((features[name] for name in names if name in features), 0.0))


def calculate_penalties(
    features: dict[str, float], events: Sequence[PayoffEvent], config: PipelineConfig
) -> dict[str, float]:
    """Return auditable, payoff-aware editorial penalties for one candidate variant."""
    payoff_present = bool(events)
    stationary_ads = _feature(features, "stationary_ads")
    low_motion = 1.0 - _feature(features, "motion")
    low_novelty = 1.0 - _feature(features, "visual_novelty", "visual_activity")
    low_escalation = 1.0 - _feature(features, "danger_escalation", "danger_score")
    stationary = stationary_ads * low_motion * low_novelty * low_escalation if not payoff_present else 0.0
    repetitive_fire = _feature(features, "repetitive_fire", "repetitive_fire_run")
    downtime = _feature(features, "downtime", "downtime_run")
    same_view = _feature(features, "same_view", "same_view_run")
    penalties = {
        "stationary_ads_penalty": stationary,
        "same_view_penalty": same_view * (1.0 - 0.5 * _feature(features, "motion")),
        "downtime_penalty": downtime * (1.0 - 0.65 * float(payoff_present)),
        "no_payoff_penalty": 0.12 if not payoff_present else 0.0,
        "repetitive_fire_penalty": repetitive_fire * (1.0 - 0.75 * float(payoff_present)),
        "same_source_recent_penalty": _feature(features, "same_source_recent"),
        "same_environment_recent_penalty": _feature(features, "same_environment_recent"),
    }
    return {
        name: round(_clamp(value) * float(config.penalty_weights.get(_PENALTY_FEATURES[name], 1.0)), 6)
        for name, value in penalties.items()
    }


def rapid_multikill_score(events: Sequence[PayoffEvent], window_s: float, minimum_events: int) -> float:
    """Return the capped payoff score for distinct rapid kill/multikill events."""
    qualifying = sorted(
        {event.event_id: event for event in events if event.type in {"kill", "multikill"}}.values(),
        key=lambda event: event.source_time,
    )
    minimum = max(2, int(minimum_events))
    window = float(window_s)
    if window <= 0 or len(qualifying) < minimum:
        return 0.0
    best_count = 0
    for start, event in enumerate(qualifying):
        count = sum(candidate.source_time - event.source_time <= window for candidate in qualifying[start:])
        best_count = max(best_count, count)
    return round(_clamp(best_count / minimum) if best_count >= minimum else 0.0, 6)


def score_variant(variant: CandidateVariant, config: PipelineConfig) -> CandidateVariant:
    """Apply the configured V1.1 weighted score and subtract explicit diagnostics."""
    values = {
        "payoff_score": variant.payoff_score,
        "human_selection_prior": variant.human_selection_prior,
        "combat_intensity": variant.combat_intensity,
        "action_density": variant.action_density,
        "continuity": variant.continuity,
        "visual_novelty": variant.visual_novelty,
        "motion": variant.motion,
        "audio_activity": variant.audio_activity,
        "danger_score": variant.danger_score,
        "uniqueness": variant.uniqueness,
    }
    weighted = {name: round(_clamp(value) * float(config.v2_weights.get(name, 0.0)), 6) for name, value in values.items()}
    penalties = {name: _clamp(value) for name, value in variant.penalty_values.items()}
    rapid_score = rapid_multikill_score(
        variant.payoff_events, config.rapid_multikill_window_s, config.rapid_multikill_min_events
    )
    configured_bonus = max(0.0, min(RAPID_MULTIKILL_MAX_BONUS, float(config.rapid_multikill_bonus_weight)))
    rapid_bonus = round(min(configured_bonus, rapid_score * configured_bonus), 6)
    verified = verified_kill_events(variant.payoff_events, config)
    verified_count = float(len(verified))
    density_count = float(max_verified_kills_in_window(
        variant.payoff_events, config.v5_kill_density_window_s, config
    ))
    kill_count_score = _clamp(verified_count / max(1.0, float(config.v5_kill_density_target)))
    density_score = _clamp(density_count / max(1.0, float(config.v5_kill_density_target)))
    components = {
        **weighted,
        **penalties,
        "rapid_multikill_score": rapid_score,
        "rapid_multikill_bonus": rapid_bonus,
        "verified_kill_count": verified_count,
        "verified_kill_score": round(kill_count_score, 6),
        "rapid_kill_count": density_count,
        "kill_density": round(density_score, 6),
    }
    base_score = _clamp(sum(weighted.values()) + rapid_bonus - sum(penalties.values()))
    if config.v5_profile == "quality":
        # The quality weights are editorial rebalancing weights, not an extra
        # bonus stacked on top of the already-normalized V2 score.  Normalize
        # the base contribution first; otherwise every strong candidate can
        # saturate at 1.0 and the beam loses the kill-density ordering.
        kill_weight = min(
            1.0,
            max(0.0, float(config.v5_kill_count_weight))
            + max(0.0, float(config.v5_kill_density_weight)),
        )
        final_score = _clamp(
            base_score * (1.0 - kill_weight)
            + float(config.v5_kill_count_weight) * kill_count_score
            + float(config.v5_kill_density_weight) * density_score
        )
    else:
        final_score = base_score
    components["base_score"] = round(base_score, 6)
    return replace(
        variant,
        final_score=round(final_score, 6),
        score_components=components,
        rapid_multikill_score=rapid_score,
        rapid_multikill_bonus=rapid_bonus,
    )
