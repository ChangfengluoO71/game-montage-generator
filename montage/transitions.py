"""Conservative transition and music synchronization decisions."""

from __future__ import annotations

from typing import Mapping

from .config import PipelineConfig
from .models import AnchorPlacement, BoundaryDescriptor, Candidate, CandidateVariant, MusicAnalysis, PayoffEvent, TransitionDecision, VideoAnalysis
from .video_analysis import describe_variant_boundary


_PRIORITY = {
    "section_change": 6,
    "phrase": 5,
    "bar": 4,
    "strong_beat": 3,
    "beat": 2,
    "onset": 1,
}


def choose_sync_point(
    shot: Candidate,
    music: MusicAnalysis,
    target: float,
    tolerance: float = 0.20,
) -> tuple[float, str, float]:
    del shot
    points = [point for point in music.edit_points if abs(float(point["timestamp"]) - target) <= max(0.75, tolerance * 4.0)]
    if not points:
        return target, "none", 0.0
    selected = min(
        points,
        key=lambda point: (
            abs(float(point["timestamp"]) - target),
            -_PRIORITY.get(str(point.get("type")), 0),
        ),
    )
    timestamp = float(selected["timestamp"])
    return timestamp, str(selected.get("type", "beat")), timestamp - target


def choose_transition(
    previous: Candidate | None,
    current: Candidate,
    analyses: Mapping[str, VideoAnalysis],
) -> str:
    del current, analyses
    if previous is None:
        return "hard_cut"
    # Direction metadata is intentionally not guessed in V1. A hard cut is safer than a false match.
    return "hard_cut"


def _music_points(anchor: PayoffEvent, music: MusicAnalysis, timeline_hint: float) -> tuple[float, str]:
    primary = anchor.type in {"multikill", "vehicle_destroy", "objective", "kill", "combat_climax", "danger_climax"}
    sources = [(float(point), "bar") for point in music.bars]
    sources += [(float(point), "strong_beat") for point in music.strong_beats]
    sources += [(float(point), "beat") for point in music.beats]
    sources += [(float(point), "onset") for point in music.onsets]
    sources += [(float(point.get("timestamp", 0.0)), str(point.get("type", "beat"))) for point in music.edit_points]
    if not sources:
        return timeline_hint, "none"
    rank = {"phrase": 5, "bar": 4, "strong_beat": 3, "beat": 2, "onset": 1}
    if primary:
        # A strong musical boundary wins over a closer ordinary onset; this keeps
        # payoff anchors on phrase/bar/downbeat structure instead of micro-edits.
        best = min(sources, key=lambda item: (-rank.get(item[1], 0), abs(item[0] - timeline_hint)))
    else:
        ordinary = [item for item in sources if item[1] in {"beat", "onset"}] or sources
        best = min(ordinary, key=lambda item: abs(item[0] - timeline_hint))
    return best


def choose_anchor_music_target(anchor: PayoffEvent, music: MusicAnalysis, timeline_hint: float, config: PipelineConfig) -> AnchorPlacement:
    target, _ = _music_points(anchor, music, timeline_hint)
    return AnchorPlacement(anchor.source_time, target, anchor.type, round(target - timeline_hint, 6), 1.0, anchor.source_time, anchor.source_time)


def protect_context_during_sync(variant: CandidateVariant, target_music_time: float, config: PipelineConfig) -> AnchorPlacement:
    anchor = variant.primary_anchor
    if anchor is None:
        start = variant.source_segments[0].source_in
        end = variant.source_segments[-1].source_out
        return AnchorPlacement((start + end) / 2.0, target_music_time, "none", 0.0, variant.context_integrity_score, start, end)
    anchored = next((segment for segment in variant.source_segments if segment.source_in <= anchor.source_time <= segment.source_out), None)
    if anchored is None:
        return AnchorPlacement(anchor.source_time, target_music_time, anchor.type, 0.0, 0.0, variant.source_segments[0].source_in, variant.source_segments[-1].source_out)
    setup = max(0.0, min(1.0, (anchor.source_time - anchored.source_in) / 1.5))
    tail = max(0.0, min(1.0, (anchored.source_out - anchor.source_time) / 1.0))
    context = min(float(variant.context_integrity_score), setup, tail)
    # The safe fallback retains the complete anchored segment. It may have a
    # larger sync error, but it cannot delete the encounter or payoff tail.
    return AnchorPlacement(anchor.source_time, target_music_time, anchor.type, 0.0, round(context, 6), anchored.source_in, anchored.source_out)


def transition_compatibility_score(previous: BoundaryDescriptor, current: BoundaryDescriptor) -> float:
    direction = 1.0 if previous.motion_direction == current.motion_direction else 0.35
    numeric = 1.0 - sum(abs(getattr(previous, name) - getattr(current, name)) for name in ("motion_strength", "luminance", "visual_tone", "weapon_motion", "impact_strength")) / 5.0
    state = 1.0 if previous.ads_state == current.ads_state else 0.45
    signatures = 1.0 if previous.environment_signature == current.environment_signature else 0.55
    source = 1.0 if previous.source_signature == current.source_signature else 0.7
    return round(max(0.0, min(1.0, 0.28 * direction + 0.38 * numeric + 0.12 * state + 0.12 * signatures + 0.10 * source)), 6)


def choose_v2_transition(previous: CandidateVariant, current: CandidateVariant) -> TransitionDecision:
    before = describe_variant_boundary(previous, None, "end")
    after = describe_variant_boundary(current, None, "start")
    compatibility = transition_compatibility_score(before, after)
    impact = bool(current.primary_anchor and current.primary_anchor.type in {"kill", "multikill", "vehicle_destroy", "impact_event"})
    return TransitionDecision("hard_cut", "impact_flash" if impact else None, compatibility, impact, 100 if impact else 0, 250 if impact else 0)
