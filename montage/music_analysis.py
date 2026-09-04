"""Confidence-aware music structure and edit-point analysis."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cache import atomic_write_bytes, cache_key, file_fingerprint, read_cached_json, v2_cache_key, write_cached_json
from .config import PipelineConfig
from .models import EditDecisionList, MusicAnalysis
from .toolchain import Toolchain


_POINT_PRIORITY = {
    "downbeat": 7,
    "section_change": 6,
    "phrase": 5,
    "bar": 4,
    "strong_beat": 3,
    "beat": 2,
    "onset": 1,
}


@dataclass(frozen=True)
class MusicWindowDecision:
    baseline_music_in: float
    baseline_music_out: float
    v2_music_in: float
    v2_music_out: float
    changed: bool
    reason: str


def _beat_stability(beat_times: Sequence[float]) -> float:
    beats = np.asarray(list(beat_times), dtype=float)
    if beats.size < 3:
        return 0.0
    intervals = np.diff(beats)
    mean_interval = float(np.mean(intervals))
    if mean_interval <= 1e-9:
        return 0.0
    return float(np.clip(1.0 - np.std(intervals) / mean_interval * 2.0, 0.0, 1.0))


def _phrase_boundaries(beat_times: Sequence[float]) -> tuple[list[float], float]:
    del beat_times
    # Beat regularity only supplies a grid; it cannot establish musical phrase semantics.
    return [], 0.0


def _downbeat_attempts(beat_times: Sequence[float]) -> tuple[list[float], float]:
    """Expose a beat-grid-derived downbeat attempt without claiming meter knowledge."""
    beats = np.asarray(list(beat_times), dtype=float)
    return beats[::4].astype(float).tolist(), 0.20


def choose_v2_music_window(
    structure: dict[str, object], baseline_in: float, baseline_out: float, duration: float, max_shift: float
) -> MusicWindowDecision:
    """Keep V1's interval unless strong heuristic boundary evidence justifies a bounded endpoint move."""
    baseline_in = round(float(baseline_in), 3)
    baseline_out = round(float(baseline_out), 3)
    max_shift = max(0.0, float(max_shift))
    candidates: list[tuple[float, float, str, bool]] = []
    for item in structure.get("phrase_boundaries", []) if isinstance(structure.get("phrase_boundaries"), list) else []:
        if isinstance(item, dict):
            timestamp = float(item.get("timestamp", -1.0))
            confidence = float(item.get("confidence", 0.0))
            independently_supported = item.get("support") not in {None, "beat_grid_only", "unsupported"}
        else:
            timestamp, confidence = float(item), float(structure.get("phrase_confidence", 0.0))
            independently_supported = False
        candidates.append((timestamp, confidence, "phrase", independently_supported))
    regions = structure.get("regions")
    if isinstance(regions, list):
        for region in regions[1:]:
            if isinstance(region, dict):
                candidates.append((float(region.get("start", -1.0)), float(region.get("confidence", 0.0)), "section", True))
    for item in structure.get("boundaries", []) if isinstance(structure.get("boundaries"), list) else []:
        if isinstance(item, dict):
            candidates.append((float(item.get("timestamp", -1.0)), float(item.get("confidence", 0.0)), str(item.get("type", "section")), item.get("support") not in {None, "beat_grid_only", "unsupported"}))

    # A boundary must carry enough evidence to be materially better than an exact V1 lock.
    eligible = [candidate for candidate in candidates if candidate[1] >= 0.70 and candidate[3] and 0.0 <= candidate[0] <= duration]
    endpoint_options: list[tuple[float, float, float, str]] = []
    for timestamp, confidence, kind, _ in eligible:
        if 0.0 < abs(timestamp - baseline_in) <= max_shift:
            endpoint_options.append((confidence, -abs(timestamp - baseline_in), timestamp, f"Adjusted music in to a high-confidence heuristic {kind} boundary."))
        if 0.0 < abs(timestamp - baseline_out) <= max_shift:
            endpoint_options.append((confidence, -abs(timestamp - baseline_out), timestamp, f"Adjusted music out to a high-confidence heuristic {kind} boundary."))
    if endpoint_options:
        _, _, timestamp, reason = max(endpoint_options)
        if " in " in reason:
            return MusicWindowDecision(baseline_in, baseline_out, round(timestamp, 3), baseline_out, True, reason)
        return MusicWindowDecision(baseline_in, baseline_out, baseline_in, round(timestamp, 3), True, reason)
    return MusicWindowDecision(
        baseline_in,
        baseline_out,
        baseline_in,
        baseline_out,
        False,
        "Retained the exact V1 baseline interval; no materially better high-confidence phrase/section boundary was within the permitted shift.",
    )


def _normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return array
    low = float(np.percentile(array, 5))
    high = float(np.percentile(array, 95))
    if high <= low + 1e-12:
        low = float(array.min())
        high = float(array.max())
    if high <= low + 1e-12:
        return np.zeros(array.shape, dtype=float)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _energy_at(timestamp: float, times: np.ndarray, energy: np.ndarray) -> float:
    if times.size == 0 or energy.size == 0:
        return 0.5
    size = min(times.size, energy.size)
    return float(np.interp(timestamp, times[:size], energy[:size], left=energy[0], right=energy[size - 1]))


def build_edit_points(
    tempo: float,
    beat_times: Sequence[float],
    onset_times: Sequence[float],
    energy: Sequence[float],
    times: Sequence[float],
) -> list[dict[str, Any]]:
    del tempo
    beat_array = np.asarray(list(beat_times), dtype=float)
    time_array = np.asarray(list(times), dtype=float)
    energy_array = np.asarray(list(energy), dtype=float)
    points: dict[tuple[float, str], dict[str, Any]] = {}
    for index, timestamp in enumerate(beat_array):
        if index % 4 == 0:
            point_type = "bar"
            base_strength = 0.95
        elif index % 2 == 0:
            point_type = "strong_beat"
            base_strength = 0.78
        else:
            point_type = "beat"
            base_strength = 0.52
        strength = max(base_strength, _energy_at(float(timestamp), time_array, energy_array) * 0.85)
        points[(round(float(timestamp), 3), point_type)] = {
            "timestamp": round(float(timestamp), 3),
            "strength": round(min(strength, 1.0), 4),
            "type": point_type,
            "confidence": 0.8 if point_type != "beat" else 0.65,
        }
    beat_array_for_match = beat_array.tolist()
    for timestamp in onset_times:
        onset = float(timestamp)
        if any(abs(onset - beat) <= 0.04 for beat in beat_array_for_match):
            continue
        points[(round(onset, 3), "onset")] = {
            "timestamp": round(onset, 3),
            "strength": round(max(0.35, _energy_at(onset, time_array, energy_array) * 0.75), 4),
            "type": "onset",
            "confidence": 0.55,
        }
    return sorted(
        points.values(),
        key=lambda point: (float(point["timestamp"]), -_POINT_PRIORITY.get(str(point["type"]), 0)),
    )


def infer_music_structure(
    times: Sequence[float],
    energy: Sequence[float],
    beat_times: Sequence[float],
    onset_strength: Sequence[float],
) -> dict[str, Any]:
    clock = np.asarray(list(times), dtype=float)
    values = np.asarray(list(energy), dtype=float)
    if values.size == 0 or clock.size == 0:
        return {"regions": [], "boundaries": [], "confidence": 0.0}
    size = min(clock.size, values.size)
    clock = clock[:size]
    values = np.clip(values[:size], 0.0, 1.0)
    raw_steps = np.diff(clock)
    raw_step = float(np.median(raw_steps[raw_steps > 0])) if np.any(raw_steps > 0) else 1.0
    bin_size = max(2.0, raw_step * 4.0)
    if clock[-1] - clock[0] < 12.0:
        bin_size = max(1.0, raw_step * 2.0)
    bin_starts = np.arange(float(clock[0]), float(clock[-1]) + 1e-9, bin_size)
    binned_clock: list[float] = []
    binned_values: list[float] = []
    for start in bin_starts:
        end = start + bin_size
        mask = (clock >= start) & (clock < end if end <= clock[-1] else clock <= end)
        if np.any(mask):
            binned_clock.append(float(start))
            binned_values.append(float(np.mean(values[mask])))
    clock = np.asarray(binned_clock, dtype=float)
    values = np.asarray(binned_values, dtype=float)
    if values.size >= 3:
        values = np.convolve(values, np.ones(3, dtype=float) / 3.0, mode="same")
    size = values.size
    q33, q66 = np.quantile(values, [0.33, 0.66])
    if q66 - q33 < 0.05:
        labels = np.array(["medium_energy"] * size, dtype=object)
        separation = 0.0
    else:
        labels = np.where(values <= q33, "low_energy", np.where(values >= q66, "high_energy", "medium_energy"))
        separation = float(q66 - q33)
    step_values = np.diff(clock)
    step = float(np.median(step_values[step_values > 0])) if np.any(step_values > 0) else 1.0
    regions: list[dict[str, Any]] = []
    start_index = 0
    for index in range(1, size + 1):
        if index == size or labels[index] != labels[start_index]:
            start = float(clock[start_index])
            end = float(clock[index - 1] + step)
            energy_name = str(labels[start_index])
            if index < size and labels[index] == "high_energy" and energy_name != "high_energy":
                role = "build_up"
            elif energy_name == "high_energy":
                role = "chorus"
            elif start < 15.0:
                role = "intro"
            else:
                role = "verse"
            confidence = float(np.clip(0.35 + separation * 4.0, 0.35, 0.95))
            if size < 8:
                confidence *= 0.7
            regions.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "energy": energy_name,
                    "role": role,
                    "confidence": round(confidence, 4),
                }
            )
            start_index = index
    beat_array = np.asarray(list(beat_times), dtype=float)
    beat_confidence = 0.2
    if beat_array.size >= 3:
        intervals = np.diff(beat_array)
        mean_interval = float(np.mean(intervals))
        coefficient = float(np.std(intervals) / mean_interval) if mean_interval else 1.0
        beat_confidence = float(np.clip(1.0 - coefficient * 2.0, 0.2, 0.98))
    onset_array = np.asarray(list(onset_strength), dtype=float)
    onset_confidence = 0.5 if onset_array.size >= 4 else 0.2
    section_confidence = float(np.mean([region["confidence"] for region in regions])) if regions else 0.2
    boundaries = [region["start"] for region in regions[1:]]
    return {
        "regions": regions,
        "boundaries": boundaries,
        "confidence": round(section_confidence, 4),
        "beat_confidence": round(beat_confidence, 4),
        "onset_confidence": round(onset_confidence, 4),
    }


def choose_preview_music_window(
    structure: dict[str, Any],
    duration: float,
    minimum: float,
    maximum: float,
) -> tuple[float, float, str]:
    regions = list(structure.get("regions") or [])
    window = min(float(maximum), float(duration))
    if duration <= minimum:
        return 0.0, float(duration), "The song is shorter than the requested preview minimum."
    for index in range(len(regions) - 1):
        current = regions[index]
        following = regions[index + 1]
        boundary = float(following["start"])
        current_duration = float(current["end"]) - float(current["start"])
        following_duration = float(following["end"]) - float(following["start"])
        if following.get("energy") == "high_energy" and (
            current.get("role") == "build_up" or current.get("energy") == "medium_energy"
        ) and boundary >= 20.0 and current_duration >= 8.0 and following_duration >= 12.0:
            start = max(0.0, boundary - window * 0.45)
            end = start + window
            if end > duration:
                end = float(duration)
                start = max(0.0, end - window)
            if end - start >= minimum:
                return round(start, 3), round(end, 3), "Representative build-up into a high-energy section transition."
    candidates: list[tuple[float, float, float]] = []
    step = 5.0
    start = 0.0
    while start + minimum <= duration + 1e-6:
        end = min(duration, start + window)
        score = 0.0
        for region in regions:
            overlap = max(0.0, min(end, float(region["end"])) - max(start, float(region["start"])))
            score += overlap * (1.0 if region.get("energy") == "high_energy" else 0.4 if region.get("energy") == "medium_energy" else 0.1)
        candidates.append((score, start, end))
        start += step
    if candidates:
        _, start, end = max(candidates)
        return round(start, 3), round(end, 3), "Highest-energy contiguous music window available from the inferred regions."
    return 0.0, round(min(duration, window), 3), "Fallback preview window because structure evidence was insufficient."


def _from_dict(data: dict[str, Any]) -> MusicAnalysis:
    return MusicAnalysis(
        source_file=Path(data["source_file"]),
        duration=float(data["duration"]),
        tempo=float(data["tempo"]),
        beats=[float(value) for value in data.get("beats", [])],
        strong_beats=[float(value) for value in data.get("strong_beats", [])],
        bars=[float(value) for value in data.get("bars", [])],
        onsets=[float(value) for value in data.get("onsets", [])],
        edit_points=list(data.get("edit_points", [])),
        energy_times=[float(value) for value in data.get("energy_times", [])],
        rms=[float(value) for value in data.get("rms", [])],
        onset_strength=[float(value) for value in data.get("onset_strength", [])],
        novelty=[float(value) for value in data.get("novelty", [])],
        structure_regions=list(data.get("structure_regions", [])),
        confidence={str(key): float(value) for key, value in (data.get("confidence") or {}).items()},
        preview_music_in=data.get("preview_music_in"),
        preview_music_out=data.get("preview_music_out"),
        preview_reason=str(data.get("preview_reason", "")),
    )


def _write_music_artifacts(analysis: MusicAnalysis, config: PipelineConfig) -> None:
    directory = config.music_analysis_dir
    directory.mkdir(parents=True, exist_ok=True)
    beat_map = {
        "source_file": str(analysis.source_file),
        "duration": analysis.duration,
        "tempo": analysis.tempo,
        "beats": analysis.beats,
        "strong_beats": analysis.strong_beats,
        "bars": analysis.bars,
        "onsets": analysis.onsets,
        "edit_points": analysis.edit_points,
        "confidence": analysis.confidence,
    }
    structure = {
        "source_file": str(analysis.source_file),
        "duration": analysis.duration,
        "regions": analysis.structure_regions,
        "confidence": analysis.confidence,
        "preview_music_in": analysis.preview_music_in,
        "preview_music_out": analysis.preview_music_out,
        "preview_reason": analysis.preview_reason,
    }
    from .cache import atomic_write_json

    atomic_write_json(directory / "beat_map.json", beat_map)
    atomic_write_json(directory / "music_structure.json", structure)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["time", "rms", "onset_strength", "novelty", "energy"])
    energy = _normalize(analysis.rms)
    onset = np.asarray(analysis.onset_strength, dtype=float)
    novelty = np.asarray(analysis.novelty, dtype=float)
    size = len(analysis.energy_times)
    for index in range(size):
        writer.writerow(
            [
                f"{analysis.energy_times[index]:.6f}",
                f"{analysis.rms[index] if index < len(analysis.rms) else 0.0:.8f}",
                f"{onset[index] if index < onset.size else 0.0:.8f}",
                f"{novelty[index] if index < novelty.size else 0.0:.8f}",
                f"{energy[index] if index < energy.size else 0.0:.8f}",
            ]
        )
    atomic_write_bytes(directory / "energy_curve.csv", buffer.getvalue().encode("utf-8-sig"))
    figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    y = np.asarray(analysis.rms, dtype=float)
    if y.size:
        axes[0].plot(analysis.energy_times[: y.size], _normalize(y), label="RMS", color="#2563eb")
    onset_values = np.asarray(analysis.onset_strength, dtype=float)
    if onset_values.size:
        axes[0].plot(analysis.energy_times[: onset_values.size], _normalize(onset_values), label="onset", color="#dc2626", alpha=0.75)
    for beat in analysis.beats:
        axes[0].axvline(beat, color="#64748b", alpha=0.08, linewidth=0.5)
    for beat in analysis.strong_beats:
        axes[0].axvline(beat, color="#f59e0b", alpha=0.35, linewidth=0.8)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("normalized signal")
    for region in analysis.structure_regions:
        color = {"low_energy": "#cbd5e1", "medium_energy": "#fde68a", "high_energy": "#fecaca"}.get(region.get("energy"), "#e2e8f0")
        axes[1].axvspan(region["start"], region["end"], color=color, alpha=0.55)
        axes[1].text((region["start"] + region["end"]) / 2, 0.8, str(region.get("role", "")), ha="center", va="center", fontsize=8)
    axes[1].plot(analysis.energy_times[: len(analysis.novelty)], _normalize(analysis.novelty), label="novelty", color="#16a34a")
    axes[1].set_xlabel("music time (s)")
    axes[1].set_ylabel("structure")
    figure.tight_layout()
    temporary = directory / ".music_analysis.tmp.png"
    figure.savefig(temporary, dpi=120)
    plt.close(figure)
    temporary.replace(directory / "music_analysis.png")


def analyze_music(config: PipelineConfig, toolchain: Toolchain) -> MusicAnalysis:
    if not config.music_file.exists():
        raise FileNotFoundError(f"Music file does not exist: {config.music_file}")
    parameters = {
        "sample_rate": config.analysis_sample_rate,
        "ffmpeg_version": toolchain.ffmpeg_version,
        "analysis_version": "coarse-energy-sections-v2",
    }
    key = cache_key(file_fingerprint(config.music_file), "music_analysis", parameters)
    cache_path = config.music_analysis_dir / "music_analysis_cache.json"
    cached = read_cached_json(cache_path, key)
    if isinstance(cached, dict) and (config.music_analysis_dir / "music_analysis.png").exists():
        return _from_dict(cached)
    samples, sample_rate = librosa.load(str(config.music_file), sr=config.analysis_sample_rate, mono=True)
    duration = float(len(samples) / sample_rate)
    tempo_value, beat_frames = librosa.beat.beat_track(y=samples, sr=sample_rate, hop_length=512)
    tempo_array = np.asarray(tempo_value).reshape(-1)
    tempo = float(tempo_array[0]) if tempo_array.size else 0.0
    beats = librosa.frames_to_time(np.asarray(beat_frames), sr=sample_rate, hop_length=512).astype(float)
    onset_values = librosa.onset.onset_strength(y=samples, sr=sample_rate, hop_length=512)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_values, sr=sample_rate, hop_length=512, units="frames")
    onsets = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=512).astype(float)
    rms_values = librosa.feature.rms(y=samples, frame_length=2048, hop_length=512)[0]
    energy_times = librosa.frames_to_time(np.arange(rms_values.size), sr=sample_rate, hop_length=512).astype(float)
    normalized_rms = _normalize(rms_values)
    normalized_onset = _normalize(onset_values)
    onset_at_energy = np.interp(energy_times, librosa.frames_to_time(np.arange(onset_values.size), sr=sample_rate, hop_length=512), normalized_onset, left=0.0, right=0.0)
    energy = _normalize(0.68 * normalized_rms + 0.32 * onset_at_energy)
    novelty = _normalize(np.abs(np.diff(energy, prepend=energy[0])))
    structure = infer_music_structure(energy_times, energy, beats, onset_values)
    points = build_edit_points(tempo, beats, onsets, energy, energy_times)
    for region in structure["regions"][1:]:
        points.append(
            {
                "timestamp": region["start"],
                "strength": region["confidence"],
                "type": "section_change",
                "confidence": region["confidence"],
            }
        )
    points.sort(key=lambda point: (point["timestamp"], -_POINT_PRIORITY.get(point["type"], 0)))
    strong_beats = beats[::2].tolist()
    bars = beats[::4].tolist()
    preview_in, preview_out, preview_reason = choose_preview_music_window(structure, duration, config.preview_min_duration, config.preview_max_duration)
    confidence = {
        "beat": float(structure.get("beat_confidence", 0.2)),
        "bar": float(structure.get("beat_confidence", 0.2) * 0.9),
        "section": float(structure.get("confidence", 0.2)),
        "onset": float(structure.get("onset_confidence", 0.2)),
    }
    analysis = MusicAnalysis(
        source_file=config.music_file,
        duration=duration,
        tempo=tempo,
        beats=beats.tolist(),
        strong_beats=strong_beats,
        bars=bars,
        onsets=onsets.tolist(),
        edit_points=points,
        energy_times=energy_times.tolist(),
        rms=rms_values.tolist(),
        onset_strength=onset_values.tolist(),
        novelty=novelty.tolist(),
        structure_regions=structure["regions"],
        confidence=confidence,
        preview_music_in=preview_in,
        preview_music_out=preview_out,
        preview_reason=preview_reason,
    )
    _write_music_artifacts(analysis, config)
    write_cached_json(cache_path, key, analysis.to_dict())
    return analysis


def _write_v2_music_artifacts(
    analysis: MusicAnalysis,
    decision: MusicWindowDecision,
    structure: dict[str, Any],
    percussive_beats: Sequence[float],
    percussive_onsets: Sequence[float],
    config: PipelineConfig,
) -> None:
    from .cache import atomic_write_json

    directory = config.music_v2_analysis_dir
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        config.music_v2_beat_map_path,
        {
            "source_file": str(analysis.source_file),
            "tempo": analysis.tempo,
            "beats": analysis.beats,
            "strong_beats": analysis.strong_beats,
            "downbeat_attempts": analysis.bars,
            "onsets": analysis.onsets,
            "percussive_beats": list(percussive_beats),
            "percussive_onsets": list(percussive_onsets),
            "edit_points": analysis.edit_points,
            "confidence": analysis.confidence,
        },
    )
    atomic_write_json(
        config.music_v2_structure_path,
        {
            "source_file": str(analysis.source_file),
            "duration": analysis.duration,
            "regions": analysis.structure_regions,
            "high_energy_regions": [region for region in analysis.structure_regions if region.get("energy") == "high_energy"],
            "low_energy_regions": [region for region in analysis.structure_regions if region.get("energy") == "low_energy"],
            "phrase_boundaries": structure["phrase_boundaries"],
            "section_boundaries": structure["boundaries"],
            "section_semantics": "heuristic energy segmentation; labels are not asserted musical form.",
            "confidence": analysis.confidence,
            "music_window": decision.__dict__,
        },
    )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["time", "rms", "onset_strength", "percussive_onset_strength", "novelty", "energy"])
    energy = _normalize(analysis.rms)
    for index, timestamp in enumerate(analysis.energy_times):
        writer.writerow([
            f"{timestamp:.6f}",
            f"{analysis.rms[index] if index < len(analysis.rms) else 0.0:.8f}",
            f"{analysis.onset_strength[index] if index < len(analysis.onset_strength) else 0.0:.8f}",
            f"{structure['percussive_onset_at_energy'][index] if index < len(structure['percussive_onset_at_energy']) else 0.0:.8f}",
            f"{analysis.novelty[index] if index < len(analysis.novelty) else 0.0:.8f}",
            f"{energy[index] if index < len(energy) else 0.0:.8f}",
        ])
    atomic_write_bytes(config.music_v2_energy_curve_path, buffer.getvalue().encode("utf-8-sig"))

    figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(analysis.energy_times[: len(analysis.rms)], _normalize(analysis.rms), label="RMS", color="#2563eb")
    axes[0].plot(analysis.energy_times[: len(analysis.onset_strength)], _normalize(analysis.onset_strength), label="onset", color="#dc2626", alpha=0.7)
    axes[0].plot(analysis.energy_times[: len(structure["percussive_onset_at_energy"])], structure["percussive_onset_at_energy"], label="percussive onset", color="#7c3aed", alpha=0.7)
    for timestamp in analysis.beats:
        axes[0].axvline(timestamp, color="#64748b", alpha=0.10, linewidth=0.5)
    for timestamp in analysis.strong_beats:
        axes[0].axvline(timestamp, color="#f59e0b", alpha=0.45, linewidth=0.8)
    for index, timestamp in enumerate(analysis.bars):
        axes[0].axvline(timestamp, color="#0f172a", alpha=0.55, linewidth=1.0, label="downbeat attempt" if index == 0 else None)
    axes[0].axvspan(decision.v2_music_in, decision.v2_music_out, color="#22c55e", alpha=0.10, label="locked V2 range")
    axes[0].set_ylabel("normalized signal")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    for region in analysis.structure_regions:
        color = {"low_energy": "#cbd5e1", "medium_energy": "#fde68a", "high_energy": "#fecaca"}.get(region.get("energy"), "#e2e8f0")
        axes[1].axvspan(region["start"], region["end"], color=color, alpha=0.55)
    for boundary in structure["phrase_boundaries"]:
        axes[1].axvline(boundary["timestamp"], color="#7c3aed", alpha=0.55, linestyle="--", label="phrase")
    for boundary in structure["boundaries"]:
        axes[1].axvline(boundary, color="#ef4444", alpha=0.45, linestyle=":", label="heuristic section")
    axes[1].set_xlabel("music time (s)")
    axes[1].set_ylabel("heuristic structure")
    figure.tight_layout()
    temporary = directory / ".music_analysis_v2.tmp.png"
    figure.savefig(temporary, dpi=120)
    plt.close(figure)
    temporary.replace(config.music_v2_analysis_image_path)


def analyze_music_v2(
    config: PipelineConfig, toolchain: Toolchain, baseline_edit: EditDecisionList
) -> tuple[MusicAnalysis, MusicWindowDecision]:
    """Analyze music with HPSS while preserving the V1 edit interval by default."""
    if not config.music_file.exists():
        raise FileNotFoundError(f"Music file does not exist: {config.music_file}")
    parameters = {
        "sample_rate": config.analysis_sample_rate,
        "ffmpeg_version": toolchain.ffmpeg_version,
        "analysis_version": "hpss-aware-v2",
        "hop_length": 512,
        "baseline_music_in": baseline_edit.music_in,
        "baseline_music_out": baseline_edit.music_out,
        "baseline_music_max_shift": config.baseline_music_max_shift,
        "window_policy_version": "confidence-supported-boundaries-v3",
        "window_policy": config.v2_music_window_policy,
        "preview_min_duration": config.preview_min_duration,
        "preview_max_duration": config.preview_max_duration,
    }
    key = v2_cache_key(file_fingerprint(config.music_file), "music_analysis_v2", parameters)
    artifact_paths = (
        config.music_v2_beat_map_path,
        config.music_v2_structure_path,
        config.music_v2_energy_curve_path,
        config.music_v2_analysis_image_path,
    )
    cached = read_cached_json(config.music_v2_cache_path, key)
    if isinstance(cached, dict) and isinstance(cached.get("analysis"), dict) and isinstance(cached.get("decision"), dict) and all(path.exists() for path in artifact_paths):
        return _from_dict(cached["analysis"]), MusicWindowDecision(**cached["decision"])

    samples, sample_rate = librosa.load(str(config.music_file), sr=config.analysis_sample_rate, mono=True)
    duration = float(len(samples) / sample_rate)
    harmonic, percussive = librosa.effects.hpss(samples)
    del harmonic
    tempo_value, full_frames = librosa.beat.beat_track(y=samples, sr=sample_rate, hop_length=512)
    percussive_tempo_value, percussive_frames = librosa.beat.beat_track(y=percussive, sr=sample_rate, hop_length=512)
    full_beats = librosa.frames_to_time(np.asarray(full_frames), sr=sample_rate, hop_length=512).astype(float)
    percussive_beats = librosa.frames_to_time(np.asarray(percussive_frames), sr=sample_rate, hop_length=512).astype(float)
    full_stability = _beat_stability(full_beats)
    percussive_stability = _beat_stability(percussive_beats)
    beats = percussive_beats if percussive_stability > full_stability and percussive_beats.size >= 3 else full_beats
    tempo_array = np.asarray(percussive_tempo_value if beats is percussive_beats else tempo_value).reshape(-1)
    tempo = float(tempo_array[0]) if tempo_array.size else 0.0
    onset_values = librosa.onset.onset_strength(y=samples, sr=sample_rate, hop_length=512)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_values, sr=sample_rate, hop_length=512, units="frames")
    onsets = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=512).astype(float)
    percussive_onset_values = librosa.onset.onset_strength(y=percussive, sr=sample_rate, hop_length=512)
    percussive_onset_frames = librosa.onset.onset_detect(onset_envelope=percussive_onset_values, sr=sample_rate, hop_length=512, units="frames")
    percussive_onsets = librosa.frames_to_time(percussive_onset_frames, sr=sample_rate, hop_length=512).astype(float)
    rms_values = librosa.feature.rms(y=samples, frame_length=2048, hop_length=512)[0]
    energy_times = librosa.frames_to_time(np.arange(rms_values.size), sr=sample_rate, hop_length=512).astype(float)
    onset_times = librosa.frames_to_time(np.arange(onset_values.size), sr=sample_rate, hop_length=512)
    percussive_onset_at_energy = np.interp(energy_times, onset_times, _normalize(percussive_onset_values), left=0.0, right=0.0)
    energy = _normalize(0.60 * _normalize(rms_values) + 0.22 * np.interp(energy_times, onset_times, _normalize(onset_values), left=0.0, right=0.0) + 0.18 * percussive_onset_at_energy)
    novelty = _normalize(np.abs(np.diff(energy, prepend=energy[0] if energy.size else 0.0)))
    structure = infer_music_structure(energy_times, energy, beats, percussive_onset_values)
    phrase_times, phrase_confidence = _phrase_boundaries(beats)
    structure["phrase_boundaries"] = [{"timestamp": round(timestamp, 3), "confidence": round(phrase_confidence, 4), "support": "beat_grid_only"} for timestamp in phrase_times]
    structure["phrase_confidence"] = round(phrase_confidence, 4)
    structure["percussive_onset_at_energy"] = percussive_onset_at_energy.tolist()
    points = build_edit_points(tempo, beats, onsets, energy, energy_times)
    strong_beats = beats[::2].tolist()
    downbeats, downbeat_confidence = _downbeat_attempts(beats)
    for timestamp in downbeats:
        points.append({"timestamp": round(timestamp, 3), "strength": downbeat_confidence, "type": "downbeat_attempt", "confidence": downbeat_confidence})
    for boundary in structure["phrase_boundaries"]:
        points.append({"timestamp": boundary["timestamp"], "strength": boundary["confidence"], "type": "phrase", "confidence": boundary["confidence"]})
    for region in structure["regions"][1:]:
        points.append({"timestamp": region["start"], "strength": region["confidence"], "type": "section_change", "confidence": region["confidence"]})
    points.sort(key=lambda point: (point["timestamp"], -_POINT_PRIORITY.get(point["type"], 0)))
    confidence = {
        "beat": full_stability if beats is full_beats else percussive_stability,
        "strong_beat": (full_stability if beats is full_beats else percussive_stability) * 0.9,
        "bar": (full_stability if beats is full_beats else percussive_stability) * 0.7,
        "downbeat": downbeat_confidence,
        "phrase": phrase_confidence,
        "section": float(structure.get("confidence", 0.0)),
        "onset": float(structure.get("onset_confidence", 0.0)),
        "percussive": percussive_stability,
    }
    if config.v2_music_window_policy == "representative":
        preview_in, preview_out, preview_reason = choose_preview_music_window(
            structure, duration, config.preview_min_duration, config.preview_max_duration
        )
        decision = MusicWindowDecision(
            baseline_music_in=float(baseline_edit.music_in),
            baseline_music_out=float(baseline_edit.music_out),
            v2_music_in=preview_in,
            v2_music_out=preview_out,
            changed=True,
            reason=preview_reason,
        )
    else:
        decision = choose_v2_music_window(
            structure, baseline_edit.music_in, baseline_edit.music_out, duration, config.baseline_music_max_shift
        )
    analysis = MusicAnalysis(
        source_file=config.music_file,
        duration=duration,
        tempo=tempo,
        beats=beats.tolist(),
        strong_beats=strong_beats,
        bars=downbeats,
        onsets=onsets.tolist(),
        edit_points=points,
        energy_times=energy_times.tolist(),
        rms=rms_values.tolist(),
        onset_strength=onset_values.tolist(),
        novelty=novelty.tolist(),
        structure_regions=structure["regions"],
        confidence={key: round(float(value), 4) for key, value in confidence.items()},
        preview_music_in=decision.v2_music_in,
        preview_music_out=decision.v2_music_out,
        preview_reason=decision.reason,
    )
    _write_v2_music_artifacts(analysis, decision, structure, percussive_beats, percussive_onsets, config)
    write_cached_json(config.music_v2_cache_path, key, {"analysis": analysis.to_dict(), "decision": decision.__dict__})
    return analysis, decision
