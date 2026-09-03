"""Low-cost analysis proxies for long recordings only."""

from __future__ import annotations

from pathlib import Path

from .cache import cache_key, read_cached_json, write_cached_json
from .config import PipelineConfig, assert_source_read_only
from .models import MediaRecord
from .toolchain import Toolchain, run_command


def build_proxy(record: MediaRecord, config: PipelineConfig, toolchain: Toolchain) -> Path | None:
    if record.duration <= config.long_clip_threshold:
        return None
    assert_source_read_only(record.file_path, config.raw_dir)
    width, height = config.proxy_resolution
    parameters = {
        "width": width,
        "height": height,
        "fps_mode": "passthrough",
        "ffmpeg_version": toolchain.ffmpeg_version,
        "encoder": "h264_nvenc" if toolchain.nvenc_h264 else "libx264",
    }
    key = cache_key(record.fingerprint, "proxy", parameters)
    destination = config.proxy_dir / f"{key[:20]}_{record.file_path.stem}.mp4"
    metadata_path = destination.with_suffix(".json")
    if destination.exists() and read_cached_json(metadata_path, key) is not None:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    encoder = "h264_nvenc" if toolchain.nvenc_h264 else "libx264"
    argv: list[str | Path] = [
        toolchain.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        record.file_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=fast_bilinear,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-fps_mode",
        "passthrough",
        "-c:v",
        encoder,
    ]
    if encoder == "h264_nvenc":
        argv.extend(["-preset", "p4", "-rc", "vbr", "-cq", "31", "-b:v", "0"])
    else:
        argv.extend(["-preset", "veryfast", "-crf", "30"])
    argv.extend(
        [
            "-c:a",
            "aac",
            "-ac",
            "1",
            "-ar",
            str(config.analysis_sample_rate),
            "-b:a",
            "96k",
            destination,
        ]
    )
    result = run_command(argv, check=True)
    if result.returncode != 0 or not destination.exists():
        raise RuntimeError(f"Proxy creation failed: {record.file_path}")
    write_cached_json(
        metadata_path,
        key,
        {"source": str(record.file_path), "duration": record.duration, "destination": str(destination)},
    )
    return destination
