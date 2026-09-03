"""V2 comparison metrics and report serialization."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from .cache import atomic_write_json
from .models import CandidateVariant, EditDecisionList, V2EditDecisionList


def _stats(values: Sequence[float]) -> dict[str, float]:
    numbers = sorted(abs(float(value)) for value in values)
    if not numbers:
        return {"mean": 0.0, "median": 0.0, "P90": 0.0, "P95": 0.0}
    def percentile(percent: float) -> float:
        index = max(0, min(len(numbers) - 1, int((len(numbers) * percent + 99) // 100) - 1))
        return numbers[index]
    return {"mean": statistics.fmean(numbers), "median": statistics.median(numbers),
            "P90": percentile(90), "P95": percentile(95)}


def _shot_durations(edit: Any) -> list[float]:
    return [float(shot.duration) for shot in getattr(edit, "shots", ())]


def _sync_values(edit: V2EditDecisionList, field: str) -> list[float]:
    return [float(getattr(shot, field, 0.0)) for shot in edit.shots]


def _ratio(variants: Sequence[CandidateVariant], key: str) -> float:
    values = [float(variant.penalty_values.get(key, variant.penalty_values.get(f"{key}_penalty", 0.0)))
              for variant in variants]
    return statistics.fmean(values) if values else 0.0


def build_v2_sync_report(edit: V2EditDecisionList, baseline: EditDecisionList,
                         variants: Sequence[CandidateVariant], rejected: dict[str, int]) -> dict[str, object]:
    durations = _shot_durations(edit)
    event = _stats(_sync_values(edit, "event_sync_offset"))
    cut = _stats(_sync_values(edit, "cut_sync_offset"))
    anchors = [shot.primary_anchor for shot in edit.shots if shot.primary_anchor is not None]
    unique_sources = {str(shot.source.resolve(strict=False)) for shot in edit.shots}
    hero_count = sum(1 for shot in edit.shots if (shot.anchor_event_type or "").lower() in {"hero", "hero_play"})
    strong_count = sum(1 for shot in edit.shots if (shot.music_event_type or "") in {"strong_beat", "downbeat"}
                       or (shot.primary_anchor is not None and shot.primary_anchor.strength >= 0.75))
    same_env = sum(1 for left, right in zip(edit.shots, edit.shots[1:])
                   if left.environment_signature and left.environment_signature == right.environment_signature)
    same_source_recent = sum(1 for left, right in zip(edit.shots, edit.shots[1:])
                             if left.source_signature and left.source_signature == right.source_signature)
    rapid_scores = [float(getattr(variant, "rapid_multikill_score", 0.0)) for variant in variants]
    report: dict[str, object] = {
        "report_version": "v2-sync-1",
        "diagnostic_evidence": True,
        "baseline_music_range": [float(baseline.music_in), float(baseline.music_out)],
        "v2_music_range": [float(edit.music_in), float(edit.music_out)],
        "macro_shot_count": len(edit.shots),
        "average_shot_duration": statistics.fmean(durations) if durations else 0.0,
        "median_shot_duration": statistics.median(durations) if durations else 0.0,
        "unique_source_count": len(unique_sources),
        "hero_shot_count": hero_count,
        "payoff_anchor_count": len(anchors),
        "strong_anchor_count": strong_count,
        "event_sync_mean": event["mean"], "event_sync_median": event["median"],
        "event_sync_P90": event["P90"], "event_sync_P95": event["P95"],
        "cut_sync_mean": cut["mean"], "cut_sync_median": cut["median"],
        "cut_sync_P90": cut["P90"], "cut_sync_P95": cut["P95"],
        "transition_compatibility_mean": statistics.fmean(
            [float(getattr(shot, "transition_compatibility_score", 0.0)) for shot in edit.shots]
        ) if edit.shots else 0.0,
        "stationary_ads_duration_ratio": _ratio(variants, "stationary_ads"),
        "estimated_downtime_ratio": _ratio(variants, "downtime"),
        "same_environment_consecutive_count": same_env,
        "same_source_recent_penalty_count": same_source_recent,
        "rejected_by_stationary_ads": int(rejected.get("stationary_ads", rejected.get("rejected_by_stationary_ads", 0))),
        "rejected_by_downtime": int(rejected.get("downtime", rejected.get("rejected_by_downtime", 0))),
        "rejected_by_no_payoff": int(rejected.get("no_payoff", rejected.get("rejected_by_no_payoff", 0))),
        "rejections": {str(key): int(value) for key, value in rejected.items()},
        "dedupe": {"unique_duplicate_groups": len({shot.duplicate_group for shot in edit.shots if shot.duplicate_group}),
                    "selected_shots": len(edit.shots)},
        "rapid_multikill": {"candidate_count": sum(1 for value in rapid_scores if value > 0),
                             "score_sum": sum(rapid_scores),
                             "bonus_sum": sum(float(getattr(variant, "rapid_multikill_bonus", 0.0)) for variant in variants)},
        "audio_strategy": "continuous music; gameplay audio retained with conservative J/L overlap",
        "transition_strategy": "hard cuts; compatibility is selection metadata only",
        "caveats": ["Metrics are diagnostic evidence, not semantic ground truth.",
                    "Phrase and section labels are heuristic.",
                    "Event and cut sync statistics are reported separately."],
    }
    return report


def write_v2_report(report: dict[str, object], path: Path) -> None:
    atomic_write_json(Path(path), report)


def render_v2_markdown_report(report: dict[str, object], path: Path, *, toolchain: Any = None,
                              output: Path | None = None) -> None:
    lines = ["# Battlefield Montage V2 Preview Report", "", "Diagnostic evidence for V1/V2 A/B review.", ""]
    for key, value in report.items():
        if key == "caveats":
            continue
        lines.append(f"- **{key}**: {json.dumps(value, ensure_ascii=False)}")
    if toolchain is not None:
        lines.extend(["", "## Encoder/toolchain", f"- FFmpeg: `{toolchain.ffmpeg}` ({toolchain.ffmpeg_version})",
                      f"- ffprobe: `{toolchain.ffprobe}` ({toolchain.ffprobe_version})"])
    if output is not None:
        lines.extend(["", "## Output", f"- `{output}`"])
    lines.extend(["", "## Caveats", *[f"- {item}" for item in report.get("caveats", [])],
                  "", "Stop rule: V2 preview only; do not render Fast Montage or Full Highlights."])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
