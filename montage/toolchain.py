"""FFmpeg discovery, runtime probing, and safe subprocess execution."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from .config import PipelineConfig


_VERSION_RE = re.compile(r"(?:ffmpeg|ffprobe) version\s+([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE)


@dataclass(frozen=True)
class Toolchain:
    ffmpeg: Path
    ffprobe: Path
    ffmpeg_version: str
    ffprobe_version: str
    nvenc_h264: bool
    nvenc_hevc: bool
    selected_reason: str
    candidates: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "ffmpeg": str(self.ffmpeg.resolve(strict=False)),
            "ffprobe": str(self.ffprobe.resolve(strict=False)),
            "ffmpeg_version": self.ffmpeg_version,
            "ffprobe_version": self.ffprobe_version,
            "nvenc_h264": self.nvenc_h264,
            "nvenc_hevc": self.nvenc_hevc,
            "selected_reason": self.selected_reason,
            "candidates": self.candidates,
        }


def _parse_version(output: str) -> str:
    match = _VERSION_RE.search(output)
    return match.group(1) if match else "0"


def _version_key(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return (0,)


def run_command(
    argv: Sequence[str | Path],
    *,
    capture_output: bool = True,
    check: bool = True,
    logger: logging.Logger | None = None,
) -> subprocess.CompletedProcess[str]:
    args = [str(value) for value in argv]
    result = subprocess.run(
        args,
        shell=False,
        check=check,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0 and logger:
        logger.error("command failed (%s): %s", result.returncode, args[0])
    return result


def find_ffmpeg_candidates(config: PipelineConfig) -> list[Path]:
    candidates: list[Path] = []
    path_result = subprocess.run(
        ["where.exe", "ffmpeg"],
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if path_result.returncode == 0:
        candidates.extend(Path(line.strip()) for line in path_result.stdout.splitlines() if line.strip())
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False)).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _sibling_ffprobe(ffmpeg: Path) -> Path:
    sibling_name = "ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe"
    sibling = ffmpeg.with_name(sibling_name)
    if sibling.exists():
        return sibling
    found = shutil.which("ffprobe")
    if found:
        return Path(found)
    return sibling


def _runtime_encode(ffmpeg: Path, encoder: str, config: PipelineConfig) -> bool:
    width = str(int(config.nvenc.get("probe_width", 1280)))
    height = str(int(config.nvenc.get("probe_height", 720)))
    duration = str(float(config.nvenc.get("probe_duration", 0.5)))
    result = run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r=30",
            "-t",
            duration,
            "-an",
            "-c:v",
            encoder,
            "-preset",
            "p4",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "null",
            "NUL",
        ],
        check=False,
    )
    return result.returncode == 0


def probe_candidate(path: Path, config: PipelineConfig, runner: Callable[..., object] | None = None) -> Toolchain:
    del runner  # The public subprocess wrapper is intentionally the single execution boundary.
    version_result = run_command([path, "-version"], check=False)
    if version_result.returncode != 0:
        raise RuntimeError(f"Unable to run FFmpeg candidate: {path}")
    version = _parse_version(version_result.stdout or "")
    encoders = run_command([path, "-hide_banner", "-encoders"], check=False)
    encoder_text = f"{encoders.stdout or ''}\n{encoders.stderr or ''}"
    listed_h264 = bool(re.search(r"\bh264_nvenc\b", encoder_text))
    listed_hevc = bool(re.search(r"\bhevc_nvenc\b", encoder_text))
    runtime_h264 = listed_h264 and _runtime_encode(path, "h264_nvenc", config)
    runtime_hevc = listed_hevc and _runtime_encode(path, "hevc_nvenc", config)
    ffprobe = _sibling_ffprobe(path)
    probe_version_result = run_command([ffprobe, "-version"], check=False)
    if probe_version_result.returncode != 0:
        raise RuntimeError(f"Unable to run sibling ffprobe: {ffprobe}")
    return Toolchain(
        ffmpeg=path,
        ffprobe=ffprobe,
        ffmpeg_version=version,
        ffprobe_version=_parse_version(probe_version_result.stdout or ""),
        nvenc_h264=runtime_h264,
        nvenc_hevc=runtime_hevc,
        selected_reason="",
        candidates=[],
    )


def discover_toolchain(config: PipelineConfig, runner: Callable[..., object] | None = None) -> Toolchain:
    del runner
    candidates = find_ffmpeg_candidates(config)
    if not candidates:
        raise FileNotFoundError("No FFmpeg executable was found with where.exe or PATH")
    inspected: list[Toolchain] = []
    errors: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            inspected.append(probe_candidate(candidate, config))
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append({"ffmpeg": str(candidate), "error": str(exc)})
    if not inspected:
        raise RuntimeError(f"No runnable FFmpeg candidate was found: {errors}")
    prefer_nvenc = bool(config.nvenc.get("prefer", True))
    selected = max(
        inspected,
        key=lambda item: (
            _version_key(item.ffmpeg_version),
            int(prefer_nvenc and item.nvenc_h264),
            int(item.nvenc_hevc),
        ),
    )
    details = [
        {
            **item.to_dict(),
            "selected": item.ffmpeg == selected.ffmpeg,
        }
        for item in inspected
    ]
    if selected.nvenc_h264:
        reason = f"selected highest-version viable FFmpeg with h264_nvenc runtime support: {selected.ffmpeg}"
    else:
        reason = f"selected highest-version runnable FFmpeg; CPU fallback required because h264_nvenc runtime is unavailable: {selected.ffmpeg}"
    return replace(selected, selected_reason=reason, candidates=details + errors)
