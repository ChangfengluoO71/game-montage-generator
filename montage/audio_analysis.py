"""Small, dependency-light audio feature extraction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .toolchain import Toolchain, run_command


def extract_analysis_audio(
    source: Path,
    destination: Path,
    toolchain: Toolchain,
    sample_rate: int = 22050,
) -> Path:
    if destination.exists() and destination.stat().st_size > 44:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [
            toolchain.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            source,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            destination,
        ],
        check=True,
    )
    if result.returncode != 0 or not destination.exists():
        raise RuntimeError(f"Audio extraction failed: {source}")
    return destination


def _normalize_array(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    if high <= low + 1e-12:
        low = float(values.min())
        high = float(values.max())
    if high <= low + 1e-12:
        return np.zeros(values.shape, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def analyze_audio_waveform(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    signal = np.asarray(samples, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        empty: list[float] = []
        return {"sample_rate": sample_rate, "times": empty, "rms": empty, "peak": empty, "spectral_flux": empty, "onset_strength": empty, "transient_density": empty, "audio_activity": empty}
    frame_length = min(2048, max(256, int(sample_rate * 0.05)))
    hop = max(128, frame_length // 2)
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))
    padded = np.pad(signal, (0, frame_length), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop]
    window = np.hanning(frame_length).astype(np.float32)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    peak = np.max(np.abs(frames), axis=1)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    flux = np.zeros(spectrum.shape[0], dtype=np.float32)
    if spectrum.shape[0] > 1:
        positive_delta = np.maximum(spectrum[1:] - spectrum[:-1], 0.0)
        flux[1:] = np.sqrt(np.mean(positive_delta * positive_delta, axis=1))
    rms_norm = _normalize_array(rms)
    flux_norm = _normalize_array(flux)
    peak_norm = _normalize_array(peak)
    activity = _normalize_array(0.45 * rms_norm + 0.40 * flux_norm + 0.15 * peak_norm)
    frame_times = (np.arange(frames.shape[0], dtype=float) * hop) / float(sample_rate)
    density = (flux_norm >= np.percentile(flux_norm, 75)).astype(float)
    return {
        "sample_rate": sample_rate,
        "times": frame_times.tolist(),
        "rms": rms.tolist(),
        "peak": peak.tolist(),
        "spectral_flux": flux.tolist(),
        "onset_strength": flux_norm.tolist(),
        "transient_density": density.tolist(),
        "audio_activity": activity.tolist(),
    }


def nearest_audio_evidence(audio: Mapping[str, Sequence[float]], source_time: float) -> dict[str, float]:
    """Return normalized transient, loudness, and impact evidence nearest a source time."""
    times = np.asarray(audio.get("times") or [], dtype=float)
    if times.size == 0:
        return {"audio_transient": 0.0, "audio_rms": 0.0, "audio_impact": 0.0}
    index = int(np.argmin(np.abs(times - float(source_time))))

    def value_for(*keys: str) -> float:
        for key in keys:
            values = np.asarray(audio.get(key) or [], dtype=float)
            if values.size:
                value = float(values[min(index, values.size - 1)])
                low, high = np.percentile(values, [5, 95])
                if high > low + 1e-12:
                    value = (value - float(low)) / float(high - low)
                return float(np.clip(value, 0.0, 1.0))
        return 0.0

    return {
        "audio_transient": value_for("onset_strength", "transient_density"),
        "audio_rms": value_for("rms", "audio_activity"),
        "audio_impact": value_for("impact", "spectral_flux", "peak"),
    }
