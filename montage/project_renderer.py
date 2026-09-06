"""Render an :class:`EditableProject` with the small, production v1 pipeline."""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import uuid
from typing import Any, Callable

from .audio_mix import build_audio_filter
from .cache import atomic_write_bytes
from .config import PipelineConfig, is_within
from .ffmpeg_renderer import _concat_file_line, _output_encoder
from .toolchain import Toolchain, run_command
from .workflow import EditableProject, TimelineClip


Progress = Callable[[int, str], None]
_EPSILON = 0.001


def _number(value: Any, name: str, *, minimum: float | None = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{name} must be a finite number{suffix}")
    return parsed


def _settings(project: EditableProject, config: PipelineConfig) -> dict[str, Any]:
    raw = project.render_settings or {}
    width = int(_number(raw.get("width", config.output_width), "render width", minimum=1))
    height = int(_number(raw.get("height", config.output_height), "render height", minimum=1))
    fps = _number(raw.get("fps", config.target_fps), "render fps", minimum=0.001)
    fade = _number(
        raw.get("fade_to_black_seconds", project.workflow.edit_rules.fade_to_black_seconds),
        "fade_to_black_seconds",
    )
    music_in = _number(raw.get("music_in", raw.get("music_start", 0.0)), "music_in")
    audio = dict(config.audio_mix)
    audio.update(project.workflow.audio_output or {})
    sample_rate = int(_number(audio.get("sample_rate", 48000), "audio sample_rate", minimum=1))
    channels = int(_number(audio.get("channels", 2), "audio channels", minimum=1))
    if channels not in (1, 2):
        raise ValueError("audio channels must be 1 or 2")
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "fade": fade,
        "music_in": music_in,
        "game_gain_db": _number(audio.get("game_gain_db", -3.0), "game_gain_db", minimum=None),
        "music_gain_db": _number(audio.get("music_gain_db", -9.0), "music_gain_db", minimum=None),
        "target_lufs": _number(audio.get("target_lufs", -14.0), "target_lufs", minimum=None),
        "true_peak_db": _number(audio.get("true_peak_db", -1.0), "true_peak_db", minimum=None),
        "sample_rate": sample_rate,
        "channels": channels,
        "bitrate": str(audio.get("bitrate", "320k")),
    }


def _ledger(project: EditableProject) -> dict[str, dict[str, Any]]:
    if not project.source_ledger:
        raise ValueError("EditableProject source_ledger is required")
    result: dict[str, dict[str, Any]] = {}
    for item in project.source_ledger:
        if not isinstance(item, dict):
            raise ValueError("source ledger entries must be objects")
        source_id = str(item.get("source_id", "")).strip()
        source_value = item.get("source_path")
        if not source_id or not source_value:
            raise ValueError("source ledger entries require source_id and source_path")
        if source_id in result:
            raise ValueError(f"duplicate source_id in source ledger: {source_id}")
        source = Path(str(source_value)).expanduser().resolve(strict=False)
        if not source.is_file():
            raise FileNotFoundError(source)
        duration = _number(item.get("duration"), f"source duration for {source_id}", minimum=0.001)
        for key in ("width", "height"):
            _number(item.get(key), f"source {source_id} {key}", minimum=1.0)
        _number(item.get("fps"), f"source {source_id} fps", minimum=0.001)
        if not isinstance(item.get("has_audio"), bool):
            raise ValueError(f"source {source_id} has_audio must be boolean")
        result[source_id] = {**item, "source": source, "duration": duration}
    return result


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _guard_generated_paths(
    config: PipelineConfig,
    output_path: Path,
    ledger: dict[str, dict[str, Any]],
    music_path: Path | None,
) -> None:
    destination_key = _path_key(output_path)
    protected_files = {_path_key(item["source"]) for item in ledger.values()}
    if music_path is not None:
        protected_files.add(_path_key(music_path))
    if destination_key in protected_files:
        raise ValueError(f"Render destination aliases a source or music input: {output_path}")

    source_roots = {item["source"].parent.resolve(strict=False) for item in ledger.values()}
    generated_dirs = (config.work_dir, config.cache_dir, config.output_dir, output_path.parent)
    for generated_dir in generated_dirs:
        if any(is_within(generated_dir, source_root) for source_root in source_roots):
            raise ValueError(f"Generated directory overlaps a source folder: {generated_dir}")


def _validate_project(project: EditableProject, config: PipelineConfig, output_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any], float]:
    project.workflow.validate()
    settings = _settings(project, config)
    destination = output_path.resolve(strict=False)
    if is_within(destination, config.raw_dir):
        raise ValueError(f"Render destination cannot be inside raw: {destination}")
    if (
        is_within(config.cache_dir, config.raw_dir)
        or is_within(config.work_dir, config.raw_dir)
        or is_within(config.output_dir, config.raw_dir)
    ):
        raise ValueError("Render work directories cannot be inside raw")
    if not project.clips:
        raise ValueError("Cannot render an empty project")
    ledger = _ledger(project)
    music_path = Path(project.music_source).expanduser().resolve(strict=False) if project.music_source else None
    if music_path is not None and not music_path.is_file():
        raise FileNotFoundError(music_path)
    _guard_generated_paths(config, destination, ledger, music_path)
    expected_timeline = 0.0
    for clip in project.clips:
        clip.validate()
        for field_name in ("source_in", "source_out", "timeline_in", "timeline_out"):
            _number(getattr(clip, field_name), f"{clip.clip_id} {field_name}")
        if clip.source_id not in ledger:
            raise ValueError(f"Unknown source_id in clip: {clip.source_id}")
        source_duration = ledger[clip.source_id]["duration"]
        source_span = clip.source_out - clip.source_in
        timeline_span = clip.timeline_out - clip.timeline_in
        if abs(source_span - timeline_span) > _EPSILON:
            raise ValueError(f"Clip duration does not preserve source duration: {clip.clip_id}")
        if clip.source_out > source_duration + 0.02:
            raise ValueError(f"Clip exceeds source duration: {clip.clip_id}")
        if abs(clip.timeline_in - expected_timeline) > _EPSILON:
            raise ValueError(f"Project timeline is not contiguous at {clip.clip_id}")
        expected_timeline = clip.timeline_out
    if expected_timeline <= 0:
        raise ValueError("Project timeline must have positive duration")
    return ledger, settings, expected_timeline


def _audio_graph(clip: TimelineClip, ledger_item: dict[str, Any], settings: dict[str, Any], music: bool) -> str:
    duration = clip.timeline_out - clip.timeline_in
    if ledger_item["has_audio"] and music:
        return build_audio_filter(
            "0:a:0",
            "1:a:0",
            settings["game_gain_db"],
            settings["music_gain_db"],
            duration=duration,
            target_lufs=settings["target_lufs"],
            true_peak_db=settings["true_peak_db"],
            sample_rate=settings["sample_rate"],
            channels=settings["channels"],
        )
    if ledger_item["has_audio"]:
        return (
            f"[0:a:0]aresample={settings['sample_rate']},volume={settings['game_gain_db']:.3f}dB,"
            f"loudnorm=I={settings['target_lufs']:.1f}:TP={settings['true_peak_db']:.1f},"
            f"apad=whole_dur={duration:.3f},atrim=duration={duration:.3f}[aout]"
        )
    if music:
        return (
            f"[1:a:0]aresample={settings['sample_rate']},volume={settings['music_gain_db']:.3f}dB,"
            f"loudnorm=I={settings['target_lufs']:.1f}:TP={settings['true_peak_db']:.1f},"
            f"apad=whole_dur={duration:.3f},atrim=duration={duration:.3f}[aout]"
        )
    layout = "mono" if settings["channels"] == 1 else "stereo"
    return f"anullsrc=r={settings['sample_rate']}:cl={layout},atrim=duration={duration:.3f}[aout]"


def _compile_segment_argv(
    clip: TimelineClip,
    ledger_item: dict[str, Any],
    settings: dict[str, Any],
    music_path: Path | None,
    destination: Path,
    toolchain: Toolchain,
) -> list[str]:
    duration = clip.timeline_out - clip.timeline_in
    encoder, encoder_args = _output_encoder(toolchain)
    video = (
        f"[0:v:0]setpts=PTS-STARTPTS,"
        f"scale=w='min(iw,{settings['width']})':h='min(ih,{settings['height']})':"
        "force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={settings['width']}:{settings['height']}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1,fps={settings['fps']:.3f}"
    )
    graph = video + "[v];" + _audio_graph(clip, ledger_item, settings, music_path is not None)
    argv: list[str | Path] = [
        toolchain.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{clip.source_in:.3f}",
        "-i",
        ledger_item["source"],
    ]
    if music_path is not None:
        argv.extend(["-stream_loop", "-1", "-ss", f"{settings['music_in'] + clip.timeline_in:.3f}", "-i", music_path])
    argv.extend([
        "-t",
        f"{duration:.3f}",
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-map",
        "[aout]",
        "-vsync",
        "cfr",
        "-r",
        f"{settings['fps']:.3f}",
        "-c:v",
        encoder,
        *encoder_args,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        str(settings["sample_rate"]),
        "-ac",
        str(settings["channels"]),
        "-b:a",
        settings["bitrate"],
        "-t",
        f"{duration:.3f}",
        destination,
    ])
    return [str(value) for value in argv]


def _compile_concat_argv(
    concat_list: Path,
    destination: Path,
    settings: dict[str, Any],
    toolchain: Toolchain,
    total_duration: float,
) -> list[str]:
    encoder, encoder_args = _output_encoder(toolchain)
    video_map = ["-map", "0:v:0"]
    if settings["fade"] > 0:
        fade = min(settings["fade"], total_duration)
        filter_args = [
            "-filter_complex",
            f"[0:v:0]fade=t=out:st={max(0.0, total_duration - fade):.3f}:d={fade:.3f}[vout]",
            "-map",
            "[vout]",
        ]
        video_map = filter_args
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
        *video_map,
        "-map",
        "0:a:0",
        "-vsync",
        "cfr",
        "-r",
        f"{settings['fps']:.3f}",
        "-c:v",
        encoder,
        *encoder_args,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        str(settings["sample_rate"]),
        "-ac",
        str(settings["channels"]),
        "-b:a",
        settings["bitrate"],
        "-movflags",
        "+faststart",
        "-t",
        f"{total_duration:.3f}",
        destination,
    ]


def _run(argv: list[str], config: PipelineConfig, stage: str) -> None:
    try:
        result = run_command(argv, check=True)
        if getattr(result, "returncode", 0) != 0:
            raise subprocess.CalledProcessError(result.returncode, argv, stderr=getattr(result, "stderr", ""))
    except (subprocess.CalledProcessError, OSError, subprocess.SubprocessError) as exc:
        stderr = getattr(exc, "stderr", "") or str(exc)
        log_dir = config.work_dir / "logs"
        if is_within(log_dir, config.raw_dir):
            raise ValueError("Renderer log directory cannot be inside raw") from exc
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"project-renderer-{uuid.uuid4().hex}.stderr.log"
        log_path.write_text(f"stage: {stage}\ncommand: {argv!r}\n{stderr}\n", encoding="utf-8")
        raise RuntimeError(f"FFmpeg {stage} failed; stderr logged to {log_path}") from exc


def render_project(
    project: EditableProject,
    config: PipelineConfig,
    toolchain: Toolchain,
    output_path: Path,
    *,
    progress: Progress | None = None,
) -> Path:
    """Compile project clips and atomically publish one AAC-track MP4."""
    output_path = Path(output_path)
    if progress:
        progress(0, "validating project")
    ledger, settings, total_duration = _validate_project(project, config, output_path)
    run_dir = config.cache_dir / "project-render" / uuid.uuid4().hex
    if is_within(run_dir, config.raw_dir):
        raise ValueError("Renderer cache directory cannot be inside raw")
    run_dir.mkdir(parents=True, exist_ok=True)
    music_path = Path(project.music_source).expanduser().resolve(strict=False) if project.music_source else None
    segments: list[Path] = []
    count = len(project.clips)
    for index, clip in enumerate(project.clips):
        segment = run_dir / f"segment_{index:03d}.mp4"
        argv = _compile_segment_argv(
            clip,
            ledger[clip.source_id],
            settings,
            music_path,
            segment,
            toolchain,
        )
        _run(argv, config, f"segment {index + 1}/{count}")
        if not segment.is_file() or segment.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg did not create segment: {segment}")
        segments.append(segment)
        if progress:
            progress(int((index + 1) * 85 / count), f"rendered segment {index + 1}/{count}")
    concat_list = run_dir / "concat.txt"
    atomic_write_bytes(concat_list, ("\n".join(_concat_file_line(path) for path in segments) + "\n").encode("utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}")
    _run(_compile_concat_argv(concat_list, temporary_output, settings, toolchain, total_duration), config, "concat")
    if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
        raise RuntimeError("FFmpeg concat did not create the final output")
    os.replace(temporary_output, output_path)
    if progress:
        progress(100, f"rendered {output_path}")
    return output_path
