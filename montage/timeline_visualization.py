"""Auditable, readable visualization of the V2 music/edit timeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import MusicAnalysis, V2EditDecisionList, V2EditShot


def _markers(values: Iterable[float], label: str, color: str, axis: plt.Axes) -> None:
    values = [float(value) for value in values]
    if not values:
        return
    axis.vlines(values, 0.0, 1.0, color=color, alpha=0.45, linewidth=0.8, label=label)


def _edit_times(values: Iterable[float], music_in: float, duration: float) -> list[float]:
    """Convert source/music-clock evidence to the EDL's zero-based edit clock."""
    upper = max(float(duration), 0.0)
    return [float(value) - float(music_in) for value in values
            if -1e-6 <= float(value) - float(music_in) <= upper + 1e-6]


def _event_label(shot: V2EditShot) -> str:
    event = shot.primary_anchor
    if event is None:
        return shot.music_event_type or "shot"
    return f"{event.type} ({event.confidence:.2f})"


def render_v2_timeline_plot(edit: V2EditDecisionList, music: MusicAnalysis, path: Path) -> None:
    """Write a timeline plot; labels are attached to the edit, not rendered media."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (energy_axis, shot_axis) = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                                                   gridspec_kw={"height_ratios": [2, 1]})
    raw_times = np.asarray(music.energy_times or [], dtype=float)
    energy = np.asarray(music.rms or [], dtype=float)
    if raw_times.size and energy.size:
        size = min(raw_times.size, energy.size)
        edit_times = raw_times[:size] - float(edit.music_in)
        mask = (edit_times >= -1e-6) & (edit_times <= float(edit.duration) + 1e-6)
        if np.any(mask):
            energy_axis.plot(edit_times[mask], energy[:size][mask], color="#243b53", linewidth=1.4, label="RMS / energy")
    _markers(_edit_times(music.beats, edit.music_in, edit.duration), "beat", "#94a3b8", energy_axis)
    _markers(_edit_times(music.strong_beats, edit.music_in, edit.duration), "strong beat", "#f59e0b", energy_axis)
    _markers(_edit_times(music.bars, edit.music_in, edit.duration), "downbeat / bar", "#ef4444", energy_axis)
    phrase = _edit_times([float(item.get("start", item.get("time", 0.0))) for item in music.structure_regions],
                         edit.music_in, edit.duration)
    _markers(phrase, "phrase / section", "#8b5cf6", energy_axis)
    energy_axis.axvspan(0.0, edit.duration, color="#22c55e", alpha=0.08, label="V2 music range")
    energy_axis.set_ylabel("normalized energy")
    energy_axis.set_ylim(bottom=0.0)
    energy_axis.legend(loc="upper right", ncol=4, fontsize=8)
    energy_axis.set_title("V2 timeline: music evidence, anchors, cuts, and diagnostic confidence")

    for index, shot in enumerate(edit.shots):
        color = "#2563eb" if index % 2 == 0 else "#60a5fa"
        shot_axis.barh(0, shot.duration, left=shot.timeline_in, height=0.5, color=color, alpha=0.85)
        label = f"{index + 1}:{_event_label(shot)}" if shot.duration >= 2.0 else str(index + 1)
        label_y = 0.34 + (0.14 if index % 2 else 0.0)
        shot_axis.text(shot.timeline_in + shot.duration / 2, label_y, label, ha="center", va="bottom",
                       fontsize=8, rotation=0, clip_on=True)
        if index:
            shot_axis.axvline(shot.timeline_in, color="#111827", linewidth=1.0)
        if shot.primary_anchor is not None:
            anchor_time = shot.timeline_in + shot.event_timeline if shot.event_timeline is not None else shot.timeline_in
            shot_axis.scatter(anchor_time, 0, marker="*", s=70, color="#dc2626", zorder=4,
                              label="primary anchor" if index == 0 else None)
        for anchor in shot.secondary_anchors:
            relative = max(0.0, min(shot.duration, anchor.source_time - shot.source_in))
            shot_axis.scatter(shot.timeline_in + relative, -0.04, marker="|", s=70, color="#f97316",
                              label="secondary anchor" if index == 0 else None)
    shot_axis.set_yticks([])
    shot_axis.set_ylabel("V2 shots")
    shot_axis.set_xlabel("seconds")
    shot_axis.legend(loc="upper right", fontsize=8)
    shot_axis.set_xlim(left=0.0, right=max(edit.duration, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
