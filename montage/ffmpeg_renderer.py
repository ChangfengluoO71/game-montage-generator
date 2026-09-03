"""Compile explicit EDL shots into bounded FFmpeg renders."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

from .audio_mix import build_audio_filter
from .cache import atomic_write_bytes
from .config import PipelineConfig, is_within
from .models import EditDecisionList, EditShot
from .toolchain import Toolchain, run_command


def _output_encoder(toolchain: Toolchain) -> tuple[str, list[str]]:
    if toolchain.nvenc_h264:
        return "h264_nvenc", ["-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
    return "libx264", ["-preset", "medium", "-crf", "18"]


def compile_shot_argv(
    shot: EditShot,
    music_file: Path,
    destination: Path,
    config: PipelineConfig,
    toolchain: Toolchain,
) -> list[str]:
    if shot.source_in < 0 or shot.source_out <= shot.source_in or shot.duration <= 0:
        raise ValueError(f"Invalid source range for {shot.source}")
    if shot.source_duration is not None and shot.source_out > shot.source_duration + 0.02:
        raise ValueError(f"Shot exceeds source duration: {shot.source}")
    if shot.music_target is not None and shot.music_target < 0:
        raise ValueError(f"Invalid music target for {shot.source}")
    if is_within(destination, config.raw_dir):
        raise ValueError(f"Render destination cannot be inside raw: {destination}")
    base_game_gain = float(config.audio_mix.get("game_gain_db", -3.0))
    base_music_gain = float(config.audio_mix.get("music_gain_db", -9.0))
    if shot.candidate_score >= 0.70:
        base_game_gain += float(config.audio_mix.get("combat_game_boost_db", 2.0))
        base_music_gain -= float(config.audio_mix.get("combat_music_duck_db", 5.0))
    graph = (
        "[0:v:0]setpts=PTS-STARTPTS,"
        f"scale=w='min(iw,{config.output_width})':h='min(ih,{config.output_height})':force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={config.output_width}:{config.output_height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v];"
        + build_audio_filter("0:a:0", "1:a:0", base_game_gain, base_music_gain)
    )
    encoder, encoder_args = _output_encoder(toolchain)
    music_start = float(shot.music_target or 0.0)
    return [
        str(toolchain.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{shot.source_in:.3f}",
        "-i",
        str(shot.source),
        "-ss",
        f"{music_start:.3f}",
        "-i",
        str(music_file),
        "-t",
        f"{shot.duration:.3f}",
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-map",
        "[aout]",
        "-fps_mode",
        "passthrough",
        "-c:v",
        encoder,
        *encoder_args,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        str(int(config.audio_mix.get("sample_rate", 48000))),
        "-b:a",
        str(config.audio_mix.get("bitrate", "320k")),
        "-shortest",
        str(destination),
    ]


def _concat_file_line(path: Path) -> str:
    escaped = str(path.resolve(strict=True)).replace("'", "'\\''")
    return f"file '{escaped}'"


def compile_concat_argv(
    concat_list: Path,
    destination: Path,
    config: PipelineConfig,
    toolchain: Toolchain,
) -> list[str]:
    encoder, encoder_args = _output_encoder(toolchain)
    if is_within(destination, config.raw_dir):
        raise ValueError(f"Concat destination cannot be inside raw: {destination}")
    return [
        str(toolchain.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-fps_mode",
        "vfr",
        "-c:v",
        encoder,
        *encoder_args,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        str(int(config.audio_mix.get("sample_rate", 48000))),
        "-b:a",
        str(config.audio_mix.get("bitrate", "320k")),
        "-movflags",
        "+faststart",
        str(destination),
    ]


def render_edit(
    edit: EditDecisionList,
    music: object,
    config: PipelineConfig,
    toolchain: Toolchain,
    output_path: Path,
) -> Path:
    del music
    if is_within(output_path, config.raw_dir):
        raise ValueError(f"Output cannot be inside raw: {output_path}")
    if not edit.shots:
        raise ValueError("Cannot render an empty edit")
    if not edit.music_source.exists():
        raise FileNotFoundError(edit.music_source)
    run_dir = config.cache_dir / "render" / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []
    for index, shot in enumerate(edit.shots):
        if not shot.source.exists():
            raise FileNotFoundError(shot.source)
        segment = run_dir / f"segment_{index:03d}.mp4"
        argv = compile_shot_argv(shot, edit.music_source, segment, config, toolchain)
        run_command(argv, check=True)
        if not segment.exists() or segment.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg did not create segment: {segment}")
        segments.append(segment)
    concat_list = run_dir / "concat.txt"
    atomic_write_bytes(concat_list, ("\n".join(_concat_file_line(segment) for segment in segments) + "\n").encode("utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}")
    run_command(compile_concat_argv(concat_list, temporary_output, config, toolchain), check=True)
    if not temporary_output.exists() or temporary_output.stat().st_size == 0:
        raise RuntimeError("FFmpeg concat did not create the final output")
    os.replace(temporary_output, output_path)
    return output_path


def _parse_fps(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe_output(path: Path, toolchain: Toolchain) -> dict[str, object]:
    result = run_command(
        [
            toolchain.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,avg_frame_rate",
            "-of",
            "json",
            path,
        ],
        check=True,
    )
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    return {
        "path": str(path.resolve(strict=False)),
        "duration": float((data.get("format") or {}).get("duration") or 0.0),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": _parse_fps(video.get("avg_frame_rate")),
        "has_video": bool(video),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
        "audio_streams": sum(stream.get("codec_type") == "audio" for stream in streams),
    }
