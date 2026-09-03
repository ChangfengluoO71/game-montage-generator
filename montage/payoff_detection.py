"""Evidence-safe local payoff detection and temporal event fusion."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .audio_analysis import nearest_audio_evidence
from .cache import file_fingerprint, read_cached_json, v2_cache_key, write_cached_json
from .config import PipelineConfig, assert_source_read_only, is_within
from .models import PayoffEvent, VideoAnalysis
from .toolchain import Toolchain, run_command
from .video_analysis import decode_color_frames


SEMANTIC_TYPES = {"kill", "multikill", "vehicle_destroy", "objective"}
SEMANTIC_RANK = {
    "visual_transient": 0,
    "impact_event": 1,
    "danger_climax": 2,
    "combat_climax": 3,
    "objective": 4,
    "kill": 5,
    "vehicle_destroy": 6,
    "multikill": 7,
}
FAMILIES = {
    "reward_roi_change": "reward",
    "killfeed_change": "killfeed",
    "crosshair_event": "crosshair",
    "objective_roi_change": "objective",
    "damage_border_change": "danger",
    "motion_peak": "action",
    "luminance_change": "action",
    "visual_novelty": "action",
    "audio_transient": "audio",
    "audio_rms": "audio",
    "audio_impact": "audio",
    "multikill_indicator": "multikill_indicator",
    "vehicle_indicator": "vehicle_indicator",
    "objective_indicator": "objective_indicator",
}


def _clamp(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass(frozen=True)
class EvidenceSample:
    source_time: float
    reward_roi_change: float = 0.0
    killfeed_change: float = 0.0
    crosshair_event: float = 0.0
    objective_roi_change: float = 0.0
    damage_border_change: float = 0.0
    motion_peak: float = 0.0
    luminance_change: float = 0.0
    visual_novelty: float = 0.0
    audio_transient: float = 0.0
    audio_rms: float = 0.0
    audio_impact: float = 0.0
    multikill_indicator: float = 0.0
    vehicle_indicator: float = 0.0
    objective_indicator: float = 0.0

    def evidence(self) -> dict[str, float]:
        return {
            field.name: _clamp(getattr(self, field.name))
            for field in fields(self)
            if field.name != "source_time" and _clamp(getattr(self, field.name)) > 0.0
        }


def _family_scores(evidence: Mapping[str, float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for channel, value in evidence.items():
        family = FAMILIES.get(channel, channel)
        scores[family] = max(scores.get(family, 0.0), _clamp(value))
    return scores


def _diminishing_fusion(values: Mapping[str, float]) -> float:
    """Noisy-or across independent families; same-family inputs are pre-collapsed."""
    remaining = 1.0
    for value in values.values():
        remaining *= 1.0 - _clamp(value)
    return _clamp(1.0 - remaining)


def _semantic_confidence(*values: float) -> float:
    return _diminishing_fusion({str(index): value for index, value in enumerate(values)})


def _fallback_type(evidence: Mapping[str, float]) -> str:
    if evidence.get("damage_border_change", 0.0) >= 0.5 and (
        evidence.get("motion_peak", 0.0) >= 0.35 or evidence.get("audio_transient", 0.0) >= 0.35
    ):
        return "danger_climax"
    if max(evidence.get("crosshair_event", 0.0), evidence.get("audio_impact", 0.0)) >= 0.5:
        return "impact_event"
    if max(
        evidence.get("reward_roi_change", 0.0), evidence.get("killfeed_change", 0.0),
        evidence.get("objective_roi_change", 0.0), evidence.get("luminance_change", 0.0),
        evidence.get("visual_novelty", 0.0),
    ) >= 0.5:
        return "visual_transient"
    return "combat_climax"


def classify_semantics(evidence: dict[str, float], config: PipelineConfig) -> tuple[str, float]:
    """Classify only labels supported by two genuinely independent evidence families."""
    values = {key: _clamp(value) for key, value in evidence.items()}
    threshold = config.payoff_evidence_threshold

    def corroborated(indicator: str, companions: tuple[str, ...]) -> tuple[bool, float]:
        indicator_value = values.get(indicator, 0.0)
        companion_values = [values.get(key, 0.0) for key in companions if values.get(key, 0.0) >= threshold]
        if indicator_value < threshold or not companion_values:
            return False, 0.0
        return True, _semantic_confidence(indicator_value, max(companion_values))

    supported, score = corroborated("multikill_indicator", ("killfeed_change", "reward_roi_change", "crosshair_event", "audio_transient"))
    if supported and score >= config.weak_anchor_threshold:
        return "multikill", score
    supported, score = corroborated("vehicle_indicator", ("reward_roi_change", "audio_impact", "crosshair_event", "killfeed_change"))
    if supported and score >= config.weak_anchor_threshold:
        return "vehicle_destroy", score
    supported, score = corroborated("objective_indicator", ("objective_roi_change", "crosshair_event", "audio_transient", "motion_peak"))
    if supported and score >= config.weak_anchor_threshold:
        return "objective", score
    supported, score = corroborated("killfeed_change", ("reward_roi_change", "crosshair_event", "audio_transient", "audio_impact", "motion_peak"))
    if supported and score >= config.weak_anchor_threshold:
        return "kill", score
    return _fallback_type(values), 0.0


def fuse_evidence(sample: EvidenceSample, config: PipelineConfig) -> PayoffEvent | None:
    evidence = sample.evidence()
    if not evidence:
        return None
    confidence = _diminishing_fusion(_family_scores(evidence))
    event_type, semantic_confidence = classify_semantics(evidence, config)
    strength = _clamp((0.70 * confidence) + (0.30 * max(evidence.values())))
    flags = tuple(sorted(channel for channel, value in evidence.items() if value >= config.payoff_evidence_threshold))
    return PayoffEvent(
        event_id=f"evt-{round(sample.source_time * 1000):08d}",
        type=event_type,
        source_time=round(float(sample.source_time), 6),
        confidence=confidence,
        strength=strength,
        semantic_confidence=semantic_confidence,
        evidence=evidence,
        detector_flags=flags,
    )


def merge_event_peaks(events: Sequence[PayoffEvent], window_ms: int) -> list[PayoffEvent]:
    if window_ms < 0:
        raise ValueError("event merge window must not be negative")
    window = window_ms / 1000.0
    groups: list[list[PayoffEvent]] = []
    for event in sorted(events, key=lambda item: item.source_time):
        if groups and event.source_time - groups[-1][-1].source_time <= window + 1e-9:
            groups[-1].append(event)
        else:
            groups.append([event])
    merged: list[PayoffEvent] = []
    for group in groups:
        peak = max(group, key=lambda item: (item.confidence, item.strength, item.semantic_confidence))
        evidence: dict[str, float] = {}
        flags: list[str] = []
        for event in group:
            for channel, value in event.evidence.items():
                evidence[channel] = max(evidence.get(channel, 0.0), _clamp(value))
            for flag in event.detector_flags:
                if flag not in flags:
                    flags.append(flag)
        event_type, semantic_confidence = classify_semantics(evidence, _MergeConfig())
        if event_type not in SEMANTIC_TYPES:
            event_type = max(group, key=lambda item: SEMANTIC_RANK.get(item.type, -1)).type
        merged.append(PayoffEvent(
            event_id=peak.event_id,
            type=event_type,
            source_time=peak.source_time,
            confidence=_diminishing_fusion({str(index): event.confidence for index, event in enumerate(group)}),
            strength=max(event.strength for event in group),
            semantic_confidence=max(semantic_confidence, max(event.semantic_confidence for event in group)),
            evidence=evidence,
            detector_flags=tuple(flags),
            merged_peak_count=sum(event.merged_peak_count for event in group),
        ))
    return merged


class _MergeConfig:
    """The merge classifier needs only the stable safety thresholds."""
    payoff_evidence_threshold = 0.35
    weak_anchor_threshold = 0.55


def _roi(frame: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = [float(value) for value in bounds]
    left, right = sorted((int(round(np.clip(x0, 0.0, 1.0) * width)), int(round(np.clip(x1, 0.0, 1.0) * width))))
    top, bottom = sorted((int(round(np.clip(y0, 0.0, 1.0) * height)), int(round(np.clip(y1, 0.0, 1.0) * height))))
    return frame[top:max(bottom, top + 1), left:max(right, left + 1)]


def _roi_change(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None or previous.shape != current.shape or current.size == 0:
        return 0.0
    return _clamp(float(np.mean(np.abs(current.astype(np.int16) - previous.astype(np.int16)))) / 255.0 * 3.0)


def _border_red_change(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None or previous.shape != current.shape or current.size == 0:
        return 0.0
    def red_excess(frame: np.ndarray) -> float:
        border = np.concatenate((frame[:2], frame[-2:], frame[:, :2], frame[:, -2:]), axis=None).reshape(-1, 3)
        return float(np.mean(np.maximum(border[:, 0].astype(float) - np.maximum(border[:, 1], border[:, 2]), 0.0)) / 255.0)
    return _clamp(abs(red_excess(current) - red_excess(previous)) * 4.0)


def _analysis_value(analysis: VideoAnalysis, channel: str, source_time: float, source_start: float) -> float:
    times = np.asarray(analysis.times, dtype=float)
    values = np.asarray(getattr(analysis, channel, []) or [], dtype=float)
    if times.size == 0 or values.size == 0:
        return 0.0
    size = min(times.size, values.size)
    # Analyses of a whole source use source timestamps; refined-window analyses start at zero.
    query_time = source_time if float(np.max(times[:size])) >= source_start else source_time - source_start
    return _clamp(float(np.interp(query_time, times[:size], values[:size])))


def ocr_available() -> bool:
    try:
        import pytesseract  # type: ignore # Optional only; this detector never requires it.
        return bool(pytesseract.get_tesseract_version())
    except (ImportError, OSError, RuntimeError):
        return False


def _decode_color_frames(source: Path, fps: float, toolchain: Toolchain, source_start: float, source_end: float) -> list[np.ndarray]:
    return decode_color_frames(source, fps, toolchain, source_start, source_end)


def _build_long_window_proxy(
    source: Path,
    source_start: float,
    source_end: float,
    config: PipelineConfig,
    toolchain: Toolchain,
) -> Path:
    """Create a cached, long-only candidate proxy below work before original-source refinement."""
    if not is_within(config.proxy_dir, config.work_dir):
        raise ValueError("payoff proxy must remain below work")
    parameters: dict[str, object] = {
        "stage_version": config.payoff_detector_version,
        "ffmpeg_version": toolchain.ffmpeg_version,
        "source_start": source_start,
        "source_end": source_end,
        "resolution": list(config.proxy_resolution),
        "payoff_analysis_fps": config.payoff_analysis_fps,
    }
    key = v2_cache_key(file_fingerprint(source), "payoff_window_proxy", parameters)
    destination = config.proxy_dir / f"payoff_{key[:20]}.mp4"
    metadata_path = destination.with_suffix(".json")
    if destination.exists() and read_cached_json(metadata_path, key) is not None:
        return destination
    duration = max(0.0, source_end - source_start)
    width, height = config.proxy_resolution
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoder = "h264_nvenc" if toolchain.nvenc_h264 else "libx264"
    argv: list[str | Path] = [
        toolchain.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{source_start:.6f}",
        "-i", source, "-t", f"{duration:.6f}", "-map", "0:v:0",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=fast_bilinear,pad={width}:{height}:(ow-iw)/2:(oh-ih)",
        "-an", "-c:v", encoder,
    ]
    if encoder == "h264_nvenc":
        argv.extend(["-preset", "p4", "-rc", "vbr", "-cq", "31", "-b:v", "0"])
    else:
        argv.extend(["-preset", "veryfast", "-crf", "30"])
    argv.append(destination)
    result = run_command(argv, check=True)
    if result.returncode != 0 or not destination.exists():
        raise RuntimeError(f"Payoff proxy creation failed: {source}")
    write_cached_json(metadata_path, key, {
        "source": str(source.resolve()), "source_start": source_start, "source_end": source_end,
        "destination": str(destination.resolve()),
    })
    return destination


def _event_from_payload(payload: Mapping[str, object]) -> PayoffEvent:
    return PayoffEvent(
        event_id=str(payload["event_id"]), type=str(payload["type"]), source_time=float(payload["source_time"]),
        confidence=float(payload["confidence"]), strength=float(payload["strength"]),
        semantic_confidence=float(payload["semantic_confidence"]),
        evidence={str(key): float(value) for key, value in dict(payload.get("evidence") or {}).items()},
        detector_flags=tuple(str(value) for value in payload.get("detector_flags") or ()),
        merged_peak_count=int(payload.get("merged_peak_count", 1)),
    )


def detect_payoff_events(
    source: Path,
    source_start: float,
    source_end: float,
    analysis: VideoAnalysis,
    audio: Mapping[str, Sequence[float]],
    config: PipelineConfig,
    toolchain: Toolchain,
) -> list[PayoffEvent]:
    """Detect events in a source-time range, caching raw and refractory-merged evidence below work."""
    assert_source_read_only(source, config.raw_dir)
    if source_end < source_start:
        raise ValueError("source_end must not precede source_start")
    if not is_within(config.cache_dir, config.work_dir):
        raise ValueError("payoff detector cache must remain below work")
    parameters: dict[str, object] = {
        "stage_version": config.payoff_detector_version,
        "ffmpeg_version": toolchain.ffmpeg_version,
        "payoff_analysis_fps": config.payoff_analysis_fps,
        "roi_profile_version": config.roi_profile.get("version", 1),
        "payoff_evidence_threshold": config.payoff_evidence_threshold,
        "weak_anchor_threshold": config.weak_anchor_threshold,
        "strong_anchor_threshold": config.strong_anchor_threshold,
        "event_merge_window_ms": config.event_merge_window_ms,
        "source_start": source_start,
        "source_end": source_end,
    }
    key = v2_cache_key(file_fingerprint(source), "payoff_events", parameters)
    cache_path = config.cache_dir / f"payoff_events_{key}.json"
    cached = read_cached_json(cache_path, key)
    if isinstance(cached, dict) and isinstance(cached.get("merged_events"), list):
        return [_event_from_payload(item) for item in cached["merged_events"]]

    duration = max(0.0, source_end - source_start)
    if duration > config.long_clip_threshold:
        proxy = _build_long_window_proxy(source, source_start, source_end, config, toolchain)
        coarse_frames = _decode_color_frames(proxy, config.payoff_analysis_fps, toolchain, 0.0, duration)
        frames = _decode_color_frames(source, config.payoff_analysis_fps, toolchain, source_start, source_end) if coarse_frames else []
    else:
        frames = _decode_color_frames(source, config.payoff_analysis_fps, toolchain, source_start, source_end)
    raw_events: list[PayoffEvent] = []
    previous: np.ndarray | None = None
    for index, frame in enumerate(frames):
        local_time = min(duration, index / max(config.payoff_analysis_fps, 1e-6))
        frame = np.asarray(frame, dtype=np.uint8)
        reward = _roi(frame, config.roi_profile.get("reward_score", [0.0, 0.0, 1.0, 1.0]))
        killfeed = _roi(frame, config.roi_profile.get("kill_feed", [0.0, 0.0, 1.0, 0.25]))
        crosshair = _roi(frame, config.roi_profile.get("crosshair", [0.4, 0.3, 0.6, 0.7]))
        objective = _roi(frame, config.roi_profile.get("objective", [0.0, 0.0, 1.0, 0.25]))
        previous_frame = previous
        sample = EvidenceSample(
            source_time=source_start + local_time,
            reward_roi_change=_roi_change(_roi(previous_frame, config.roi_profile.get("reward_score", [0, 0, 1, 1])) if previous_frame is not None else None, reward),
            killfeed_change=_roi_change(_roi(previous_frame, config.roi_profile.get("kill_feed", [0, 0, 1, 0.25])) if previous_frame is not None else None, killfeed),
            crosshair_event=_roi_change(_roi(previous_frame, config.roi_profile.get("crosshair", [0.4, 0.3, 0.6, 0.7])) if previous_frame is not None else None, crosshair),
            objective_roi_change=_roi_change(_roi(previous_frame, config.roi_profile.get("objective", [0, 0, 1, 0.25])) if previous_frame is not None else None, objective),
            damage_border_change=_border_red_change(previous_frame, frame),
            motion_peak=_analysis_value(analysis, "motion", source_start + local_time, source_start),
            luminance_change=_clamp(abs(float(frame.mean()) - float(previous_frame.mean())) / 255.0 * 3.0) if previous_frame is not None else 0.0,
            visual_novelty=_analysis_value(analysis, "visual", source_start + local_time, source_start),
            **nearest_audio_evidence(
                audio,
                source_start + local_time if max(audio.get("times") or [0.0]) >= source_start else local_time,
            ),
        )
        event = fuse_evidence(sample, config)
        if event is not None:
            raw_events.append(event)
        previous = frame
    merged_events = merge_event_peaks(raw_events, config.event_merge_window_ms)
    histogram, edges = np.histogram([event.confidence for event in merged_events], bins=[0.0, 0.25, 0.5, 0.75, 1.000001])
    write_cached_json(cache_path, key, {
        "detector_version": config.payoff_detector_version,
        "roi_profile_version": config.roi_profile.get("version", 1),
        "thresholds": {"evidence": config.payoff_evidence_threshold, "weak_anchor": config.weak_anchor_threshold, "strong_anchor": config.strong_anchor_threshold},
        "source_fingerprint": file_fingerprint(source),
        "source_start": source_start,
        "source_end": source_end,
        "ocr_available": ocr_available(),
        "raw_events": [event.to_dict() for event in raw_events],
        "merged_events": [event.to_dict() for event in merged_events],
        "confidence_histogram": {f"{edges[index]:.2f}-{edges[index + 1]:.2f}": int(count) for index, count in enumerate(histogram)},
    })
    return merged_events


def write_payoff_events(events: Sequence[PayoffEvent], path: Path) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError("payoff events must be written as JSON")
    from .cache import atomic_write_json
    atomic_write_json(path, {"events": [event.to_dict() for event in events], "event_count": len(events)})
