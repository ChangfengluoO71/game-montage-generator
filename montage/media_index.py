"""Recursive media discovery and ffprobe-backed index generation."""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from .cache import atomic_write_bytes, file_fingerprint
from .config import PipelineConfig, assert_source_read_only
from .models import MediaRecord
from .toolchain import Toolchain, run_command


def parse_fps(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            if float(denominator) != 0:
                return round(float(numerator) / float(denominator), 3)
        except ValueError:
            return 0.0
    try:
        return round(float(value), 3)
    except ValueError:
        return 0.0


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def probe_media(path: Path, toolchain: Toolchain) -> MediaRecord:
    result = run_command(
        [
            toolchain.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,bit_rate:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,channels,sample_rate:format_tags=creation_time",
            "-of",
            "json",
            path,
        ],
        check=True,
    )
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video:
        raise ValueError(f"No video stream: {path}")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    format_data = payload.get("format") or {}
    tags = format_data.get("tags") or {}
    duration = float(format_data.get("duration") or 0.0)
    fps = parse_fps(video.get("avg_frame_rate")) or parse_fps(video.get("r_frame_rate"))
    if duration <= 90.0:
        category = "short_clip"
    elif duration <= 300.0:
        category = "medium_clip"
    else:
        category = "long_clip"
    return MediaRecord(
        file_path=path.resolve(strict=True),
        file_name=path.name,
        file_size=path.stat().st_size,
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        codec=str(video.get("codec_name") or ""),
        bitrate=_optional_int(format_data.get("bit_rate")),
        audio_codec=str(audio.get("codec_name")) if audio else None,
        audio_channels=_optional_int(audio.get("channels")) if audio else None,
        audio_sample_rate=_optional_int(audio.get("sample_rate")) if audio else None,
        creation_time=str(tags.get("creation_time")) if tags.get("creation_time") else None,
        category=category,
        fingerprint=file_fingerprint(path),
    )


def _iter_media_files(config: PipelineConfig) -> Iterable[Path]:
    extensions = {extension.lower() for extension in config.video_extensions}
    for path in sorted(config.raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def _record_from_dict(data: dict[str, Any]) -> MediaRecord:
    return MediaRecord(
        file_path=Path(data["file_path"]),
        file_name=str(data["file_name"]),
        file_size=int(data["file_size"]),
        duration=float(data["duration"]),
        width=int(data["width"]),
        height=int(data["height"]),
        fps=float(data["fps"]),
        codec=str(data["codec"]),
        bitrate=int(data["bitrate"]) if data.get("bitrate") not in (None, "") else None,
        audio_codec=str(data["audio_codec"]) if data.get("audio_codec") else None,
        audio_channels=int(data["audio_channels"]) if data.get("audio_channels") not in (None, "") else None,
        audio_sample_rate=int(data["audio_sample_rate"]) if data.get("audio_sample_rate") not in (None, "") else None,
        creation_time=str(data["creation_time"]) if data.get("creation_time") else None,
        category=str(data["category"]),
        fingerprint=dict(data.get("fingerprint") or {}),
        probe_error=data.get("probe_error"),
    )


def build_media_index(
    config: PipelineConfig,
    toolchain: Toolchain,
    logger: logging.Logger,
) -> list[MediaRecord]:
    if not config.raw_dir.exists():
        raise FileNotFoundError(f"RAW directory does not exist: {config.raw_dir}")
    previous: dict[str, dict[str, Any]] = {}
    previous_path = config.analysis_dir / "media_index.json"
    if previous_path.exists():
        try:
            previous_payload = json.loads(previous_path.read_text(encoding="utf-8"))
            previous_records = previous_payload.get("records", []) if isinstance(previous_payload, dict) else []
            previous = {str(Path(item["file_path"]).resolve()): item for item in previous_records}
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            previous = {}
    records: list[MediaRecord] = []
    for path in _iter_media_files(config):
        try:
            assert_source_read_only(path, config.raw_dir)
            resolved = path.resolve(strict=True)
            fingerprint = file_fingerprint(resolved)
            cached = previous.get(str(resolved))
            if cached and cached.get("fingerprint") == fingerprint:
                records.append(_record_from_dict(cached))
            else:
                records.append(probe_media(resolved, toolchain))
        except Exception as exc:  # one corrupt recording must not abort the index
            logger.error("INDEX skipped %s: %s", path, exc)
    return records


def write_media_index(
    records: Iterable[MediaRecord],
    json_path: Path,
    csv_path: Path,
    errors: list[dict[str, object]] | None = None,
) -> None:
    materialized = list(records)
    envelope = {
        "records": [record.to_dict() for record in materialized],
        "errors": errors or [],
    }
    atomic_write_bytes(json_path, json.dumps(envelope, ensure_ascii=False, indent=2).encode("utf-8"))
    fields = [
        "file_path",
        "file_name",
        "file_size",
        "duration",
        "width",
        "height",
        "fps",
        "codec",
        "bitrate",
        "audio_codec",
        "audio_channels",
        "audio_sample_rate",
        "creation_time",
        "category",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for record in materialized:
        writer.writerow(record.to_dict())
    atomic_write_bytes(csv_path, buffer.getvalue().encode("utf-8-sig"))
