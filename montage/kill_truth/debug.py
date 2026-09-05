"""Debug visualizations for skull state timelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from .models import OwnKillEvent, SkullRowState


def render_skull_state_timeline(
    states: Iterable[SkullRowState],
    events: Iterable[OwnKillEvent],
    output: Path,
    *,
    title: str = "V6 Skull Row State Timeline",
) -> Path:
    materialized = list(states)
    event_list = list(events)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    if materialized:
        times = [state.timestamp for state in materialized]
        counts = [state.skull_count if state.panel_present else 0 for state in materialized]
        confidences = [state.confidence for state in materialized]
        headshots = [state.headshot_count for state in materialized]
        axes[0].step(times, counts, where="post", color="#4b82c5", label="skull_count")
        axes[0].scatter(times, counts, s=7, color="#4b82c5", alpha=0.4)
        axes[1].plot(times, confidences, color="#6aa84f", label="panel confidence")
        axes[1].plot(times, headshots, color="#e69138", alpha=0.8, label="headshot_count")
    for index, event in enumerate(event_list, start=1):
        axes[0].axvline(event.confirmation_time, color="#cc0000", alpha=0.55, linewidth=0.9)
        axes[0].text(event.confirmation_time, max(1, event.skull_count_after) + 0.12, f"K{index}", color="#cc0000", fontsize=8)
    axes[0].set_ylabel("count")
    axes[1].set_ylabel("confidence")
    axes[1].set_xlabel("source seconds")
    axes[0].set_title(title)
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output
