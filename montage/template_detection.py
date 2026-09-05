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


@dataclass(frozen=True)
class TemplateEvent:
    timestamp: float
    confidence: float
    source_id: str
    rule_id: str
    label: str
    evidence: dict[str, object]


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
    progress: Callable[[int, str], None] | None = None,
) -> list[TemplateEvent]:
    """Scan decoded frames and emit only threshold rising edges."""
    templates = [load_image_unicode(Path(path)) for path in template_paths]
    if not templates:
        raise ValueError("template rule has no templates")
    if not 0 <= float(threshold) <= 1:
        raise ValueError("template threshold must be between 0 and 1")
    reference_height = int(reference_height or 0)
    active = False
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
        if matched and not active:
            event_id = f"{source_id}-{rule_id}-{len(events) + 1:04d}"
            events.append(TemplateEvent(timestamp, score, source_id, rule_id, label, {
                "detector": "template_match",
                "template_index": int(template_index) if template_index else None,
                "scale": scale,
                "threshold": float(threshold),
                "positive_samples_are_calibration_references": True,
            }))
        active = matched
        seen_frames += 1
        now = time.monotonic()
        if progress and (now - last_report >= 0.2 or seen_frames == 1):
            percent = min(99, int((float(timestamp) / duration) * 100)) if duration and duration > 0 else min(99, seen_frames)
            progress(percent, f"template frames scanned: {seen_frames}")
            last_report = now
    if seen_frames == 0:
        raise RuntimeError("decoder returned no video frames")
    return events


def iter_full_frames(toolchain: object, source: Path, *, width: int, height: int, fps: float) -> Iterable[tuple[float, np.ndarray]]:
    """Yield full BGR frames from FFmpeg without creating frame files."""
    if width < 1 or height < 1 or fps <= 0:
        raise ValueError("invalid video decode dimensions")
    ffmpeg = getattr(toolchain, "ffmpeg", toolchain)
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(source), "-map", "0:v:0", "-an", "-vf", f"fps={fps:.6f}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    size = width * height * 3
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
            yield index / fps, np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3)).copy()
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


def scan_template_source(record: object, rule: object, toolchain: object, *, progress: Callable[[int, str], None] | None = None) -> list[TemplateEvent]:
    from .kill_truth.scanner import source_id_for_path

    detector = rule.detector
    template_paths = [Path(value.get("file", value.get("path", "")) if isinstance(value, dict) else value) for value in detector.templates]
    positive_paths = [Path(value) for value in detector.positive_samples if isinstance(value, (str, Path))]
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
    fps = min(max(float(getattr(record, "fps", 30.0) or 30.0), 1.0), 30.0)
    frames = iter_full_frames(toolchain, Path(record.file_path), width=int(record.width), height=int(record.height), fps=fps)
    return scan_template_frames(frames, roi, template_paths, threshold=threshold, source_id=source_id_for_path(Path(record.file_path)), rule_id=rule.id, label=rule.label, reference_height=reference_height, duration=float(getattr(record, "duration", 0.0) or 0.0), progress=progress)
