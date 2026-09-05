"""Direct-RAW two-pass scanner for V6 skull-row truth."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import cv2
import numpy as np

from ..cache import atomic_write_json, file_fingerprint, read_cached_json
from ..config import assert_source_read_only
from ..models import MediaRecord
from ..toolchain import Toolchain
from .models import KillSequence, OwnKillEvent, SkullRowState
from .panel_state import TemporalStateMachine
from .profile import HudProfile, template_bank_fingerprint, v6_cache_key
from .sequence import build_sequences
from .skull_detector import SkullDetector


SCANNER_VERSION = "v6-scanner-even-crop-2-cache-rebuild"
TRUTH_LOGIC_VERSION = "v6-temporal-fsm-3-refinement-terminal-tolerant"


@dataclass(frozen=True)
class V6ScanConfig:
    coarse_fps: float = 12.0
    dense_fps: float = 60.0
    panel_disappear_s: float = 0.75
    refinement_radius_s: float = 1.0
    keep_all_states: bool = True


@dataclass
class V6ScanResult:
    source_id: str
    source_path: Path
    width: int
    height: int
    duration: float
    hud_profile_id: str
    status: str
    cache_key: str
    cache_hit: bool = False
    frames_scanned: int = 0
    dense_frames_scanned: int = 0
    coarse_states: list[SkullRowState] = field(default_factory=list)
    refinement_requests: list[dict[str, Any]] = field(default_factory=list)
    events: list[OwnKillEvent] = field(default_factory=list)
    sequences: list[KillSequence] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "v6-scan-result-v1",
            "source_id": self.source_id,
            "source_path": str(self.source_path.resolve(strict=False)),
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "hud_profile_id": self.hud_profile_id,
            "status": self.status,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "frames_scanned": self.frames_scanned,
            "dense_frames_scanned": self.dense_frames_scanned,
            "coarse_states": [state.to_dict() for state in self.coarse_states],
            "refinement_requests": self.refinement_requests,
            "events": [event.to_dict() for event in self.events],
            "sequences": [sequence.to_dict() for sequence in self.sequences],
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "V6ScanResult":
        return cls(
            source_id=str(data["source_id"]),
            source_path=Path(data["source_path"]),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            duration=float(data.get("duration", 0.0)),
            hud_profile_id=str(data.get("hud_profile_id", "")),
            status=str(data.get("status", "UNKNOWN")),
            cache_key=str(data.get("cache_key", "")),
            cache_hit=bool(data.get("cache_hit", False)),
            frames_scanned=int(data.get("frames_scanned", 0)),
            dense_frames_scanned=int(data.get("dense_frames_scanned", 0)),
            coarse_states=[SkullRowState.from_dict(item) for item in data.get("coarse_states", [])],
            refinement_requests=[dict(item) for item in data.get("refinement_requests", [])],
            events=[OwnKillEvent.from_dict(item) for item in data.get("events", [])],
            sequences=[KillSequence.from_dict(item) for item in data.get("sequences", [])],
            errors=[str(item) for item in data.get("errors", [])],
        )


def source_id_for_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:10]
    return f"{resolved.stem}-{digest}"


def _raw_pipe_command(
    toolchain: Toolchain,
    source: Path,
    *,
    fps: float,
    crop_bounds: tuple[int, int, int, int],
    start: float = 0.0,
    duration: float | None = None,
) -> list[str | Path]:
    x1, y1, x2, y2 = crop_bounds
    crop_width = x2 - x1
    crop_height = y2 - y1
    command: list[str | Path] = [toolchain.ffmpeg, "-hide_banner", "-loglevel", "error"]
    if start > 0.0:
        command.extend(["-ss", f"{start:.6f}"])
    command.extend(["-i", source])
    if duration is not None:
        command.extend(["-t", f"{max(0.001, duration):.6f}"])
    command.extend(
        [
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"fps={fps:.6f},crop={crop_width}:{crop_height}:{x1}:{y1}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
    )
    return command


def _even_crop_bounds(crop_bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Align rawvideo crop dimensions to the even sizes used by YUV420.

    FFmpeg's YUV420 path silently rounds an odd crop width down (for example
    345 -> 344).  The pipe reader must use the same dimensions or the final
    frame is misclassified as a truncated raw frame.
    """
    x1, y1, x2, y2 = (int(value) for value in crop_bounds)
    width = max(2, x2 - x1)
    height = max(2, y2 - y1)
    width -= width % 2
    height -= height % 2
    return x1, y1, x1 + width, y1 + height


def iter_cropped_frames(
    toolchain: Toolchain,
    source: Path,
    *,
    fps: float,
    crop_bounds: tuple[int, int, int, int],
    start: float = 0.0,
    duration: float | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield BGR frames from an FFmpeg crop without writing frame files."""
    crop_bounds = _even_crop_bounds(crop_bounds)
    x1, y1, x2, y2 = crop_bounds
    width = x2 - x1
    height = y2 - y1
    frame_size = width * height * 3
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("invalid cropped-frame decode parameters")
    command = [str(value) for value in _raw_pipe_command(toolchain, source, fps=fps, crop_bounds=crop_bounds, start=start, duration=duration)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    index = 0
    try:
        assert process.stdout is not None
        while True:
            # A pipe read is allowed to return fewer bytes than requested even
            # while FFmpeg is still writing the same rawvideo frame.  Treating
            # that short read as a partial frame caused false scan errors on
            # larger crops/resolutions.  Accumulate exactly one frame, and
            # only treat an empty first read as EOF.
            chunks: list[bytes] = []
            remaining = frame_size
            while remaining:
                chunk = process.stdout.read(remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if not payload:
                break
            if len(payload) != frame_size:
                raise RuntimeError(f"FFmpeg returned a partial raw frame ({len(payload)} of {frame_size} bytes)")
            frame = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3)).copy()
            yield start + index / fps, frame
            index += 1
    finally:
        if process.stdout is not None:
            process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.stderr is not None:
            process.stderr.close()
        return_code = process.wait()
        if return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg crop decode failed ({return_code}): {message}")


def _cache_path(cache_dir: Path, source_id: str) -> Path:
    digest = hashlib.sha1(source_id.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _threshold_fingerprint(profile: HudProfile) -> str:
    payload = {
        "normal_threshold": profile.normal_threshold,
        "headshot_threshold": profile.headshot_threshold,
        "geometry_threshold": profile.geometry_threshold,
        "panel_structure_threshold": profile.panel_structure_threshold,
        "shape_threshold": profile.shape_threshold,
        "orange_color_threshold": profile.orange_color_threshold,
        "row_y_tolerance_px": profile.row_y_tolerance_px,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def scan_source(
    record: MediaRecord,
    profile: HudProfile,
    toolchain: Toolchain,
    *,
    scan_config: V6ScanConfig | None = None,
    cache_dir: Path | None = None,
    raw_dir: Path | None = None,
    detector: SkullDetector | None = None,
    use_cache: bool = True,
    progress: Callable[[int, str], None] | None = None,
) -> V6ScanResult:
    settings = scan_config or V6ScanConfig()
    assert_source_read_only(record.file_path, raw_dir or record.file_path.parent)
    source_id = source_id_for_path(record.file_path)
    detector = detector or SkullDetector.from_profile(profile)
    template_fp = template_bank_fingerprint(profile.all_template_paths)
    expected_key = v6_cache_key(
        record.fingerprint or file_fingerprint(record.file_path),
        profile,
        template_fp,
        threshold_fingerprint=_threshold_fingerprint(profile),
        ffmpeg_version=toolchain.ffmpeg_version,
        decode_parameters={
            "scanner_version": SCANNER_VERSION,
            "truth_logic_version": TRUTH_LOGIC_VERSION,
            "coarse_fps": settings.coarse_fps,
            "dense_fps": settings.dense_fps,
            "panel_disappear_s": settings.panel_disappear_s,
            "refinement_radius_s": settings.refinement_radius_s,
        },
    )
    output_cache = _cache_path(cache_dir, source_id) if cache_dir else None
    if use_cache and output_cache:
        cached = read_cached_json(output_cache, expected_key)
        if isinstance(cached, dict):
            result = V6ScanResult.from_dict(cached)
            result.cache_hit = True
            return result

    search_bounds = _even_crop_bounds(profile.pixel_roi(record.width, record.height))
    machine = TemporalStateMachine(
        source_id,
        record.file_path,
        panel_disappear_s=settings.panel_disappear_s,
        refinement_radius_s=settings.refinement_radius_s,
    )
    coarse_states: list[SkullRowState] = []
    frames_scanned = 0
    dense_frames_scanned = 0
    errors: list[str] = []
    last_progress = 0.0

    def report(percent: int, message: str, *, force: bool = False) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if progress and (force or now - last_progress >= 0.2):
            progress(max(0, min(100, int(percent))), message)
            last_progress = now

    try:
        for timestamp, frame in iter_cropped_frames(
            toolchain,
            record.file_path,
            fps=settings.coarse_fps,
            crop_bounds=search_bounds,
        ):
            state = detector.detect_cropped(
                frame,
                timestamp,
                full_width=record.width,
                full_height=record.height,
                crop_bounds=search_bounds,
                source_id=source_id,
            )
            frames_scanned += 1
            expected_frames = max(1, int(record.duration * settings.coarse_fps))
            report(min(65, int(frames_scanned * 55 / expected_frames)), f"V6 coarse frames scanned: {frames_scanned}")
            if settings.keep_all_states or state.panel_present:
                coarse_states.append(state)
            machine.consume(state)
    except Exception as exc:
        errors.append(f"coarse_scan: {exc}")

    requests = list(machine.refinement_requests)
    for request in requests:
        try:
            states: list[SkullRowState] = []
            window_duration = max(0.001, min(record.duration, request.window_end) - max(0.0, request.window_start))
            start = max(0.0, request.window_start)
            for timestamp, frame in iter_cropped_frames(
                toolchain,
                record.file_path,
                fps=settings.dense_fps,
                crop_bounds=search_bounds,
                start=start,
                duration=window_duration,
            ):
                dense_frames_scanned += 1
                expected_dense = max(1, int(window_duration * settings.dense_fps))
                report(min(95, 65 + int(dense_frames_scanned * 30 / expected_dense)), f"V6 dense frames scanned: {dense_frames_scanned}")
                states.append(
                    detector.detect_cropped(
                        frame,
                        timestamp,
                        full_width=record.width,
                        full_height=record.height,
                        crop_bounds=search_bounds,
                        source_id=source_id,
                    )
                )
            machine.apply_refinement(request, states)
        except Exception as exc:
            errors.append(f"dense_refinement: {exc}")

    machine.finish()
    events = sorted(machine.events, key=lambda item: item.confirmation_time)
    sequences = build_sequences(events, source_id=source_id, source_path=record.file_path)
    result = V6ScanResult(
        source_id=source_id,
        source_path=record.file_path,
        width=record.width,
        height=record.height,
        duration=record.duration,
        hud_profile_id=profile.profile_id,
        status="OK" if not errors else "PARTIAL_ERROR",
        cache_key=expected_key,
        frames_scanned=frames_scanned,
        dense_frames_scanned=dense_frames_scanned,
        coarse_states=coarse_states,
        refinement_requests=[request.to_dict() for request in requests],
        events=events,
        sequences=sequences,
        errors=errors,
    )
    if output_cache:
        atomic_write_json(output_cache, {"cache_key": expected_key, "data": result.to_dict()})
    report(100, f"V6 scan complete: {len(events)} events", force=True)
    return result


def dry_run_source(record: MediaRecord, profile: HudProfile, toolchain: Toolchain, scan_config: V6ScanConfig | None = None) -> dict[str, Any]:
    settings = scan_config or V6ScanConfig()
    bounds = _even_crop_bounds(profile.pixel_roi(record.width, record.height))
    command = _raw_pipe_command(toolchain, record.file_path, fps=settings.coarse_fps, crop_bounds=bounds)
    return {
        "source_id": source_id_for_path(record.file_path),
        "source_path": str(record.file_path.resolve(strict=False)),
        "profile_id": profile.profile_id,
        "resolution": [record.width, record.height],
        "profile_resolution": [profile.width, profile.height],
        "calibrated": (record.width, record.height) == (profile.width, profile.height),
        "coarse_command": [str(value) for value in command],
        "coarse_fps": settings.coarse_fps,
        "dense_fps": settings.dense_fps,
        "scanner_version": SCANNER_VERSION,
        "uses_old_candidates": False,
        "writes_raw": False,
    }
