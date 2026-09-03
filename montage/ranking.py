"""Configurable candidate scoring."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .config import PipelineConfig
from .models import Candidate


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
