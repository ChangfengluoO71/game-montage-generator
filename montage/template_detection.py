"""Small, auditable grayscale template detector used by desktop generation."""

from __future__ import annotations

import subprocess
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

from .cache import cache_key, file_fingerprint, read_cached_json, write_cached_json


@dataclass(frozen=True)
class TemplateEvent:
    timestamp: float
    confidence: float
    source_id: str
    rule_id: str
    label: str
    evidence: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": float(self.timestamp),
            "confidence": float(self.confidence),
            "source_id": self.source_id,
            "rule_id": self.rule_id,
            "label": self.label,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TemplateEvent":
        return cls(
            float(data["timestamp"]),
            float(data["confidence"]),
            str(data["source_id"]),
            str(data["rule_id"]),
            str(data["label"]),
            dict(data.get("evidence") or {}),
        )


def load_image_unicode(path: Path) -> np.ndarray:
    """Load an image through bytes so Windows paths remain Unicode safe."""
    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(f"template or sample is missing: {resolved}")
    try:
        payload = np.fromfile(str(resolved), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"unable to read template or sample: {resolved}: {exc}") from exc
    image = cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE) if payload.size else None
    if image is None or image.size == 0:
        raise ValueError(f"template or sample is not a decodable image: {resolved}")
    if image.ndim != 2 or image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError(f"template is too small: {resolved}")
    if float(np.std(image)) <= 1e-6:
        raise ValueError(f"template is constant: {resolved}")
    return image


def _path_value(value: object) -> Path:
    if isinstance(value, dict):
        value = value.get("file", value.get("path", ""))
    if not value:
        raise ValueError("template entry requires a file path")
    return Path(str(value))


def _roi_pixels(roi: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(roi) != 4:
        raise ValueError("template ROI must contain four normalized values")
    x1, y1, x2, y2 = (float(item) for item in roi)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("template ROI must be a normalized rectangle")
    return round(width * x1), round(height * y1), round(width * x2), round(height * y2)


def _best_match(region: np.ndarray, templates: Sequence[np.ndarray], scales: Sequence[float]) -> tuple[float, str, float]:
    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    best = (-1.0, "", 1.0)
    for index, template in enumerate(templates):
        for scale in scales:
            width = max(2, int(round(template.shape[1] * scale)))
            height = max(2, int(round(template.shape[0] * scale)))
            if width > region_gray.shape[1] or height > region_gray.shape[0]:
                continue
            scaled = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA) if scale != 1 else template
            response = cv2.matchTemplate(region_gray, scaled, cv2.TM_CCOEFF_NORMED)
            response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
            score = float(response.max()) if response.size else -1.0
            if score > best[0]:
                best = (score, str(index), float(scale))
    return best


def scan_template_frames(
    frames: Iterable[tuple[float, np.ndarray]],
    roi: Sequence[float],
    template_paths: Sequence[Path],
    *,
    threshold: float,
    source_id: str,
    rule_id: str,
    label: str,
    reference_height: int | None = None,
    duration: float | None = None,
    release_seconds: float = 0.10,
    progress: Callable[[int, str], None] | None = None,
) -> list[TemplateEvent]:
    """Scan decoded frames and emit debounced threshold rising edges.

    A single dropped frame must not re-arm a HUD marker that is still visible.
    ``release_seconds`` is measured from the most recent positive match, so a
    new event is emitted only after the marker has been absent for a bounded
    interval.
    """
    templates = [load_image_unicode(Path(path)) for path in template_paths]
    if not templates:
        raise ValueError("template rule has no templates")
    if not 0 <= float(threshold) <= 1:
        raise ValueError("template threshold must be between 0 and 1")
    reference_height = int(reference_height or 0)
    active = False
    below_since: float | None = None
    events: list[TemplateEvent] = []
    last_report = 0.0
    seen_frames = 0
    for timestamp, frame in frames:
        if frame is None or not getattr(frame, "size", 0):
            raise RuntimeError(f"decoder returned an empty frame at {timestamp:.3f}s")
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = _roi_pixels(roi, width, height)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            raise ValueError("template ROI is outside decoded frame")
        scale_base = (height / reference_height) if reference_height > 0 else 1.0
        scales = tuple(scale_base * value for value in (0.94, 0.97, 1.0, 1.03, 1.06))
        if all(int(round(template.shape[1] * min(scales))) > region.shape[1] or int(round(template.shape[0] * min(scales))) > region.shape[0] for template in templates):
            raise ValueError("template is larger than the configured ROI")
        score, template_index, scale = _best_match(region, templates, scales)
        matched = score >= float(threshold)
        release_window = max(0.0, float(release_seconds))
        if matched and (not active or (below_since is not None and timestamp - below_since + 1e-6 >= release_window)):
            event_id = f"{source_id}-{rule_id}-{len(events) + 1:04d}"
            events.append(TemplateEvent(timestamp, score, source_id, rule_id, label, {
                "detector": "template_match",
                "template_index": int(template_index) if template_index else None,
                "scale": scale,
                "threshold": float(threshold),
                "positive_samples_are_calibration_references": True,
            }))
        if matched:
            active = True
            below_since = None
        elif below_since is None:
            below_since = timestamp
        elif timestamp - below_since + 1e-6 >= release_window:
            active = False
        seen_frames += 1
        now = time.monotonic()
        if progress and (now - last_report >= 0.2 or seen_frames == 1):
            percent = min(99, int((float(timestamp) / duration) * 100)) if duration and duration > 0 else min(99, seen_frames)
            progress(percent, f"template frames scanned: {seen_frames}")
            last_report = now
    if seen_frames == 0:
        raise RuntimeError("decoder returned no video frames")
    return events


def _path_fingerprint(path: Path) -> dict[str, object]:
    try:
        return file_fingerprint(path)
    except (FileNotFoundError, OSError):
        return {"absolute_path": str(path.resolve(strict=False))}


def _template_cache_key(
    record: object,
    rule: object,
    toolchain: object,
    *,
    fps_cap: float,
    decode_height: int | None,
) -> str:
    detector = rule.detector
    source = Path(record.file_path)
    template_values = [str(_path_value(value).resolve(strict=False)) for value in detector.templates]
    positive_values = [str(Path(value).resolve(strict=False)) for value in detector.positive_samples]
    detector_parameters = {
        "templates": [{"path": value, "fingerprint": _path_fingerprint(Path(value))} for value in template_values],
        "positive_samples": [{"path": value, "fingerprint": _path_fingerprint(Path(value))} for value in positive_values],
        "roi": {key: float(detector.roi[key]) for key in ("x1", "y1", "x2", "y2")},
        "threshold": float(detector.thresholds.get("template", 0.65)),
        "rule_id": str(rule.id),
    }
    parameters = {
        "stage_version": "template-match-v2",
        "detector": detector_parameters,
        "fps_cap": float(fps_cap),
        "decode_height": int(decode_height) if decode_height is not None else None,
        "source_dimensions": [int(record.width), int(record.height)],
        "ffmpeg_version": str(getattr(toolchain, "ffmpeg_version", "unknown")),
    }
    return cache_key(_path_fingerprint(source), "template_scan", parameters)


def _decode_dimensions(width: int, height: int, decode_height: int | None) -> tuple[int, int]:
    if decode_height is None or decode_height <= 0 or height <= decode_height:
        return width, height
    output_height = max(2, int(decode_height))
    output_width = max(2, int(round(width * output_height / height)))
    if output_width % 2:
        output_width -= 1
    return max(2, output_width), output_height


def iter_full_frames(
    toolchain: object,
    source: Path,
    *,
    width: int,
    height: int,
    fps: float,
    scale_width: int | None = None,
    scale_height: int | None = None,
) -> Iterable[tuple[float, np.ndarray]]:
    """Yield full BGR frames from FFmpeg without creating frame files."""
    if width < 1 or height < 1 or fps <= 0:
        raise ValueError("invalid video decode dimensions")
    ffmpeg = getattr(toolchain, "ffmpeg", toolchain)
    output_width = int(scale_width or width)
    output_height = int(scale_height or height)
    if output_width < 1 or output_height < 1:
        raise ValueError("invalid scaled video dimensions")
    filters = [f"fps={fps:.6f}"]
    if (output_width, output_height) != (width, height):
        filters.append(f"scale={output_width}:{output_height}:flags=fast_bilinear")
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(source), "-map", "0:v:0", "-an", "-vf", ",".join(filters), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    size = output_width * output_height * 3
    index = 0
    try:
        assert process.stdout is not None
        while True:
            chunks: list[bytes] = []
            remaining = size
            while remaining:
                chunk = process.stdout.read(remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if not payload:
                break
            if len(payload) != size:
                raise RuntimeError(f"decoder returned a partial frame ({len(payload)} of {size} bytes)")
            yield index / fps, np.frombuffer(payload, dtype=np.uint8).reshape((output_height, output_width, 3)).copy()
            index += 1
    finally:
        if process.stdout is not None:
            process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.stderr is not None:
            process.stderr.close()
        code = process.wait()
        if code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg video decode failed ({code}): {detail}")


def scan_template_source(
    record: object,
    rule: object,
    toolchain: object,
    *,
    progress: Callable[[int, str], None] | None = None,
    fps_cap: float = 30.0,
    decode_height: int | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> list[TemplateEvent]:
    from .kill_truth.scanner import source_id_for_path

    detector = rule.detector
    template_paths = [_path_value(value) for value in detector.templates]
    positive_paths = [Path(value) for value in detector.positive_samples if isinstance(value, (str, Path))]
    cache_path: Path | None = None
    expected_cache_key: str | None = None
    if cache_dir is not None and use_cache:
        expected_cache_key = _template_cache_key(
            record,
            rule,
            toolchain,
            fps_cap=fps_cap,
            decode_height=decode_height,
        )
        cache_path = Path(cache_dir) / f"{expected_cache_key}.json"
        cached = read_cached_json(cache_path, expected_cache_key)
        if isinstance(cached, list):
            events = [TemplateEvent.from_dict(item) for item in cached if isinstance(item, dict)]
            if progress:
                progress(100, f"template cache hit: {len(events)} events")
            return events

    reference_height = None
    if positive_paths:
        positive_path = positive_paths[0]
        if positive_path.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}:
            probe = subprocess.run([str(getattr(toolchain, "ffprobe", "ffprobe")), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height", "-of", "json", str(positive_path)], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
            try:
                reference_height = int((json.loads(probe.stdout).get("streams") or [{}])[0].get("height") or 0)
            except (ValueError, TypeError, json.JSONDecodeError):
                reference_height = 0
            if reference_height <= 0:
                raise ValueError(f"positive sample video has no valid dimensions: {positive_path}")
        else:
            positive = load_image_unicode(positive_path)
            reference_height = positive.shape[0]
    roi = tuple(float(detector.roi[key]) for key in ("x1", "y1", "x2", "y2"))
    threshold = float(detector.thresholds.get("template", 0.65))
    fps_limit = max(1.0, float(fps_cap))
    fps = min(max(float(getattr(record, "fps", 30.0) or 30.0), 1.0), fps_limit)
    source_width = int(record.width)
    source_height = int(record.height)
    decode_width, decode_height_value = _decode_dimensions(source_width, source_height, decode_height)
    frames = iter_full_frames(
        toolchain,
        Path(record.file_path),
        width=source_width,
        height=source_height,
        fps=fps,
        scale_width=decode_width,
        scale_height=decode_height_value,
    )
    events = scan_template_frames(
        frames,
        roi,
        template_paths,
        threshold=threshold,
        source_id=source_id_for_path(Path(record.file_path)),
        rule_id=rule.id,
        label=rule.label,
        reference_height=reference_height,
        duration=float(getattr(record, "duration", 0.0) or 0.0),
        progress=progress,
    )
    if cache_path is not None and expected_cache_key is not None:
        write_cached_json(cache_path, expected_cache_key, [event.to_dict() for event in events])
    return events
