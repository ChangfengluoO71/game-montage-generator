"""Low-rate video activity analysis without a heavyweight CV model."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cache import atomic_write_json
from .config import PipelineConfig
from .models import MediaRecord, VideoAnalysis
from .toolchain import Toolchain


def normalize_signal(values: Sequence[float]) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return []
    low = float(array.min())
    high = float(array.max())
    if high <= low + 1e-12:
        return [0.0] * int(array.size)
    return np.clip((array - low) / (high - low), 0.0, 1.0).tolist()


def merge_activity_peaks(
    activity: Sequence[float],
    times: Sequence[float],
    threshold: float,
    gap: float,
    min_duration: float,
    max_duration: float,
) -> list[tuple[float, float]]:
    values = np.asarray(list(activity), dtype=float)
    clock = np.asarray(list(times), dtype=float)
    if values.size == 0 or clock.size != values.size:
        return []
    active = np.flatnonzero(values >= threshold)
    if active.size == 0:
        return []
    steps = np.diff(clock)
    step = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 1.0
    groups: list[list[int]] = [[int(active[0])]]
    for index in active[1:]:
        previous = groups[-1][-1]
        if clock[int(index)] - clock[previous] <= gap + step:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    result: list[tuple[float, float]] = []
    for group in groups:
        start = float(clock[group[0]])
        end = float(clock[group[-1]] + (2.0 * step))
        while end - start > max_duration:
            chunk_end = start + max_duration
            if chunk_end - start >= min_duration:
                result.append((round(start, 3), round(chunk_end, 3)))
            start = chunk_end
        if end - start >= min_duration:
            result.append((round(start, 3), round(end, 3)))
    return result


def _entropy(frame: np.ndarray) -> float:
    histogram = np.bincount(frame.reshape(-1), minlength=256).astype(float)
    probabilities = histogram[histogram > 0] / frame.size
    if probabilities.size == 0:
        return 0.0
    return float(-(probabilities * np.log2(probabilities)).sum() / 8.0)


def _decode_grayscale_frames(source: Path, record: MediaRecord, fps: float, toolchain: Toolchain) -> tuple[list[np.ndarray], float]:
    width = 320
    height = max(2, int(round((record.height / max(record.width, 1)) * width / 2.0) * 2))
    frame_size = width * height
    argv = [
        str(toolchain.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"fps={fps},scale={width}:{height}:flags=fast_bilinear",
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(
        argv,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frames: list[np.ndarray] = []
    assert process.stdout is not None
    while True:
        chunk = process.stdout.read(frame_size)
        if len(chunk) != frame_size:
            break
        frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(height, width).copy())
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Video decode failed for {source}: {stderr[-500:]}")
    return frames, fps


def _align_feature(times: np.ndarray, feature: dict[str, Any], key: str) -> np.ndarray:
    source_times = np.asarray(feature.get("times") or [], dtype=float)
    source_values = np.asarray(feature.get(key) or [], dtype=float)
    if source_times.size == 0 or source_values.size == 0:
        return np.zeros(times.size, dtype=float)
    size = min(source_times.size, source_values.size)
    return np.interp(times, source_times[:size], source_values[:size], left=source_values[0], right=source_values[size - 1])


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if values.size == 0 or width <= 1:
        return values
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, kernel, mode="same")


def analyze_video_activity(
    record: MediaRecord,
    source_for_analysis: Path,
    audio_features: dict[str, Any],
    config: PipelineConfig,
    toolchain: Toolchain,
) -> VideoAnalysis:
    frames, sample_rate = _decode_grayscale_frames(source_for_analysis, record, config.analysis_fps, toolchain)
    if not frames:
        return VideoAnalysis(record.file_path, config.analysis_fps, [], [], [], [], [], [])
    brightness: list[float] = []
    black_ratio: list[float] = []
    entropy: list[float] = []
    differences: list[float] = []
    previous: np.ndarray | None = None
    for frame in frames:
        brightness.append(float(frame.mean()) / 255.0)
        black_ratio.append(float(np.mean(frame < 8)))
        entropy.append(_entropy(frame))
        if previous is None:
            differences.append(0.0)
        else:
            differences.append(float(np.mean(np.abs(frame.astype(np.int16) - previous.astype(np.int16)))) / 255.0)
        previous = frame
    times = np.arange(len(frames), dtype=float) / max(sample_rate, 1e-6)
    motion = np.asarray(normalize_signal(differences), dtype=float)
    entropy_norm = np.asarray(normalize_signal(entropy), dtype=float)
    entropy_change = np.asarray(normalize_signal(np.abs(np.diff(entropy, prepend=entropy[0]))), dtype=float)
    visual = np.clip(0.45 * motion + 0.35 * entropy_norm + 0.20 * entropy_change, 0.0, 1.0)
    audio = _align_feature(times, audio_features, "audio_activity")
    audio = np.asarray(normalize_signal(audio.tolist()), dtype=float)
    raw_activity = (0.35 * motion) + (0.25 * visual) + (0.25 * audio) + (0.15 * (1.0 - np.asarray(black_ratio)))
    activity = np.asarray(normalize_signal(_smooth(raw_activity, max(1, int(round(sample_rate * 2.0))))), dtype=float)
    continuity = np.asarray(normalize_signal(_smooth(activity, max(1, int(round(sample_rate * 3.0))))), dtype=float)
    threshold = float(np.percentile(activity, config.activity_threshold_percentile)) if activity.size else 1.0
    windows = merge_activity_peaks(
        activity.tolist(),
        times.tolist(),
        threshold,
        config.activity_merge_gap,
        config.highlight_min_duration,
        config.highlight_max_duration,
    )
    return VideoAnalysis(
        source_file=record.file_path,
        sample_rate=sample_rate,
        times=times.tolist(),
        motion=motion.tolist(),
        visual=visual.tolist(),
        audio=audio.tolist(),
        continuity=continuity.tolist(),
        activity=activity.tolist(),
        brightness=brightness,
        black_ratio=black_ratio,
        entropy=entropy,
        candidate_windows=[{"start": start, "end": end} for start, end in windows],
    )


def write_video_analysis(analysis: VideoAnalysis, json_path: Path, plot_path: Path) -> None:
    atomic_write_json(json_path, analysis.to_dict())
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].plot(analysis.times, analysis.activity, label="activity", color="#dc2626")
    axes[0].plot(analysis.times, analysis.audio, label="audio", alpha=0.65)
    axes[0].plot(analysis.times, analysis.motion, label="motion", alpha=0.65)
    for window in analysis.candidate_windows:
        axes[0].axvspan(window["start"], window["end"], color="#f59e0b", alpha=0.18)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("normalized")
    axes[1].plot(analysis.times, analysis.visual, label="visual", color="#2563eb")
    axes[1].plot(analysis.times, analysis.continuity, label="continuity", color="#16a34a")
    axes[1].plot(analysis.times, analysis.black_ratio, label="black ratio", color="#111827", alpha=0.6)
    axes[1].legend(loc="upper right")
    axes[1].set_xlabel("source time (s)")
    figure.tight_layout()
    temporary = plot_path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=120)
    plt.close(figure)
    temporary.replace(plot_path)
