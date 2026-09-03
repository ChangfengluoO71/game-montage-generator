"""Gameplay-first preview timeline and human-readable EDL output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .cache import atomic_write_bytes
from .config import PipelineConfig
from .models import Candidate, EditDecisionList, EditShot, MusicAnalysis
from .music_analysis import choose_preview_music_window
from .transitions import choose_sync_point, choose_transition


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


def _section_for(music: MusicAnalysis, timestamp: float) -> str:
    for region in music.structure_regions:
        if float(region["start"]) <= timestamp < float(region["end"]):
            return f"{region.get('role', 'section')} / {region.get('energy', 'unknown')}"
    return "unknown / unknown"


def _candidate_order(candidates: Sequence[Candidate], preferred_max_duration: float) -> list[Candidate]:
    # Ascending score creates a rise into the finale; it is deliberately not a high-to-low ranking.
    ordered = sorted(candidates, key=lambda candidate: (candidate.final_score, candidate.continuity_score, candidate.candidate_id))
    saved = [candidate for candidate in ordered if candidate.source_category == "short_clip" or candidate.human_selection_score >= 0.30]
    saved_montage = [candidate for candidate in saved if candidate.duration <= preferred_max_duration]
    saved_long = [candidate for candidate in saved if candidate.duration > preferred_max_duration]
    fallback = [candidate for candidate in ordered if candidate not in saved]
    return saved_montage + fallback + saved_long


def build_preview_edit(
    candidates: Sequence[Candidate],
    music: MusicAnalysis,
    config: PipelineConfig,
) -> EditDecisionList:
    if music.preview_music_in is None or music.preview_music_out is None:
        music_in, music_out, music_reason = choose_preview_music_window(
            {"regions": music.structure_regions},
            music.duration,
            config.preview_min_duration,
            config.preview_max_duration,
        )
    else:
        music_in = float(music.preview_music_in)
        music_out = float(music.preview_music_out)
        music_reason = music.preview_reason
    maximum = min(config.preview_max_duration, max(config.preview_min_duration, music_out - music_in))
    minimum = min(config.preview_min_duration, maximum)
    ordered = _candidate_order(candidates, config.preview_preferred_shot_max_duration)
    selected: list[Candidate] = []
    used_groups: set[str] = set()
    total = 0.0
    for candidate in ordered:
        group = candidate.duplicate_group or candidate.candidate_id
        if group in used_groups:
            continue
        if total >= maximum:
            break
        remaining = maximum - total
        chosen = candidate
        if candidate.duration > remaining:
            if candidate.source_category == "short_clip" or candidate.human_selection_score >= 0.30:
                continue
            if total >= minimum or remaining < 3.0:
                continue
            chosen = Candidate(**{**candidate.__dict__, "source_end": candidate.source_start + remaining, "duration": remaining})
        selected.append(chosen)
        used_groups.add(group)
        total += chosen.duration
    if total < minimum:
        for candidate in reversed(ordered):
            group = candidate.duplicate_group or candidate.candidate_id
            if group in used_groups:
                continue
            remaining = maximum - total
            if remaining < 1.0:
                break
            chosen = candidate
            if chosen.duration > remaining:
                if chosen.source_category == "short_clip" or chosen.human_selection_score >= 0.30:
                    continue
                chosen = Candidate(**{**chosen.__dict__, "source_end": chosen.source_start + remaining, "duration": remaining})
            selected.append(chosen)
            used_groups.add(group)
            total += chosen.duration
            if total >= minimum:
                break
    shots: list[EditShot] = []
    timeline = 0.0
    previous: Candidate | None = None
    for index, candidate in enumerate(selected):
        target = music_in + timeline
        music_target, event_type, offset = choose_sync_point(candidate, music, target)
        transition = choose_transition(previous, candidate, {})
        section = _section_for(music, music_target)
        if index == 0:
            rationale = f"Opening establishing combat at medium energy; {candidate.rationale}"
        elif index == len(selected) - 1:
            rationale = f"Reserve a high-quality closing action for the preview finale; {candidate.rationale}"
        else:
            rationale = f"Increase narrative energy toward the detected high-energy section; {candidate.rationale}"
        shots.append(
            EditShot(
                source=candidate.source_file,
                source_in=candidate.source_start,
                source_out=candidate.source_end,
                duration=candidate.duration,
                candidate_score=candidate.final_score,
                duplicate_group=candidate.duplicate_group,
                timeline_in=round(timeline, 3),
                timeline_out=round(timeline + candidate.duration, 3),
                transition=transition,
                music_target=round(music_target, 3),
                music_event_type=event_type,
                sync_offset=round(offset, 6),
                rationale=rationale,
                section=section,
            )
        )
        timeline += candidate.duration
        previous = candidate
    music_out = min(music.duration, music_in + timeline)
    return EditDecisionList(
        kind="preview",
        music_source=music.source_file,
        music_in=round(music_in, 3),
        music_out=round(music_out, 3),
        duration=round(timeline, 3),
        music_reason=music_reason,
        shots=shots,
    )


def write_edit_list(edit: EditDecisionList, json_path: Path, timeline_path: Path) -> None:
    atomic_write_bytes(json_path, json.dumps(edit.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"))
    lines = [
        f"Music: {_format_time(edit.music_in)} - {_format_time(edit.music_out)}",
        f"Reason: {edit.music_reason}",
        f"Duration: {edit.duration:.3f}s",
        "",
    ]
    for shot in edit.shots:
        lines.extend(
            [
                f"{_format_time(shot.timeline_in)} - {_format_time(shot.timeline_out)}",
                f"Source: {shot.source.name}",
                f"Source range: {_format_time(shot.source_in)} - {_format_time(shot.source_out)}",
                f"Score: {shot.candidate_score:.3f}",
                f"Music: {shot.section}",
                f"Sync: {shot.music_event_type} {shot.sync_offset * 1000:+.0f}ms",
                f"Transition: {shot.transition}",
                f"Reason: {shot.rationale}",
                "",
            ]
        )
    atomic_write_bytes(timeline_path, "\n".join(lines).encode("utf-8"))
