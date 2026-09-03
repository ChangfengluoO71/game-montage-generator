"""Safe V2 media rendering with hard-cut video and gameplay-first audio."""

from __future__ import annotations

import os
import json
import math
import uuid
from pathlib import Path
from typing import Sequence

from .audio_mix import build_v2_audio_filter
from .config import PipelineConfig, assert_source_read_only, is_within
from .models import SourceSegment, V2EditDecisionList, V2EditShot
from .toolchain import Toolchain, run_command


def _encoder(toolchain: Toolchain, config: PipelineConfig) -> tuple[str, list[str]]:
    # nvenc_h264 is set only after the runtime probe in Toolchain discovery.
    if toolchain.nvenc_h264:
        return str(config.nvenc.get("video_encoder", "h264_nvenc")), [
            "-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", "0",
        ]
    return str(config.nvenc.get("fallback_encoder", "libx264")), ["-preset", "medium", "-crf", "18"]


def _strictly_within(path: Path, root: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    return resolved_path != resolved_root and is_within(resolved_path, resolved_root)


def _validate_destination(destination: Path, config: PipelineConfig, *, final: bool) -> None:
    resolved = destination.resolve(strict=False)
    raw = config.raw_dir.resolve(strict=False)
    baseline = config.baseline_output_path.resolve(strict=False)
    output_dir = config.output_dir.resolve(strict=False)

    if resolved == raw or is_within(resolved, raw):
        raise ValueError(f"Render destination cannot be inside raw: {destination}")
    if resolved == baseline:
        raise ValueError(f"Render destination cannot replace the immutable baseline: {destination}")
    if resolved == output_dir:
        raise ValueError(f"Render destination cannot be the output directory: {destination}")

    if _strictly_within(resolved, config.work_dir):
        return
    if final and resolved == config.v2_output_path.resolve(strict=False):
        return
    scope = "work_dir or the configured V2 output" if final else "work_dir"
    raise ValueError(f"Render destination must be below {scope}: {destination}")


def _probe_source_duration(source: Path, toolchain: Toolchain) -> float:
    result = run_command(
        [
            toolchain.ffprobe, "-hide_banner", "-loglevel", "error",
            "-show_entries", "stream=codec_type:format=duration",
            "-of", "json", str(source),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed for V2 source: {source}")
    try:
        payload = json.loads(result.stdout or "")
        streams = payload.get("streams") or []
        duration = float((payload.get("format") or {}).get("duration"))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"V2 source probe has invalid duration: {source}") from exc
    stream_types = {stream.get("codec_type") for stream in streams if isinstance(stream, dict)}
    if not {"video", "audio"} <= stream_types:
        raise ValueError(f"V2 source must contain video and audio streams: {source}")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"V2 source probe has invalid duration: {source}")
    return duration


def _parse_fps(value: object) -> float:
    if not isinstance(value, str) or not value:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _validate_rendered_v2_output(
    output: Path, edit: V2EditDecisionList, config: PipelineConfig, toolchain: Toolchain
) -> None:
    probe = run_command(
        [
            toolchain.ffprobe, "-hide_banner", "-loglevel", "error",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate:format=duration",
            "-of", "json", str(output),
        ],
        check=False,
    )
    if probe.returncode != 0:
        raise ValueError(f"ffprobe rejected V2 output: {output}")
    try:
        payload = json.loads(probe.stdout or "")
        streams = [stream for stream in payload.get("streams", []) if isinstance(stream, dict)]
        video = next(stream for stream in streams if stream.get("codec_type") == "video")
        audio = next(stream for stream in streams if stream.get("codec_type") == "audio")
        actual_duration = float((payload.get("format") or {}).get("duration"))
        width = int(video.get("width"))
        height = int(video.get("height"))
        fps = _parse_fps(video.get("avg_frame_rate"))
    except (AttributeError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
        raise ValueError(f"V2 output probe is missing required media metadata: {output}") from exc

    if not audio or width != config.output_width or height != config.output_height:
        raise ValueError(
            f"V2 output has invalid streams or geometry: {output} "
            f"(width={width}, height={height}, has_audio={bool(audio)})"
        )
    if not math.isfinite(actual_duration) or actual_duration <= 0:
        raise ValueError(f"V2 output probe has invalid duration: {output}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"V2 output probe has invalid frame cadence: {output}")
    expected_fps = float(config.target_fps)
    if expected_fps > 0 and abs(fps - expected_fps) > max(0.5, expected_fps * 0.02):
        raise ValueError(
            f"V2 output frame cadence does not match the configured cadence: "
            f"actual={fps:.3f}, expected={expected_fps:.3f}"
        )
    duration_tolerance = max(0.1, 2.0 / expected_fps) if expected_fps > 0 else 0.1
    if abs(actual_duration - edit.duration) > duration_tolerance:
        raise ValueError(
            f"V2 output duration does not match the EDL: "
            f"actual={actual_duration:.3f}, expected={edit.duration:.3f}, "
            f"tolerance={duration_tolerance:.3f}"
        )

    decode = run_command(
        [
            toolchain.ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror",
            "-i", str(output), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", os.devnull,
        ],
        check=False,
    )
    if decode.returncode != 0:
        raise ValueError(f"V2 output failed strict full decode: {output}")


def _preflight_v2_sources(edit: V2EditDecisionList, config: PipelineConfig, toolchain: Toolchain) -> None:
    requested_out: dict[Path, float] = {}
    for shot in edit.shots:
        requested_out[shot.source.resolve(strict=False)] = max(
            requested_out.get(shot.source.resolve(strict=False), 0.0), shot.source_out
        )
        for segment in shot.source_segments:
            key = segment.source.resolve(strict=False)
            requested_out[key] = max(requested_out.get(key, 0.0), segment.source_out)

    for source, source_out in requested_out.items():
        assert_source_read_only(source, config.raw_dir)
        actual_duration = _probe_source_duration(source, toolchain)
        if source_out > actual_duration + 0.02:
            raise ValueError(
                f"V2 source range exceeds probed duration for {source}: "
                f"source_out={source_out:.3f}, duration={actual_duration:.3f}"
            )


def compile_v2_segment_argv(
    segment: SourceSegment,
    config: PipelineConfig,
    toolchain: Toolchain,
    destination: Path,
) -> list[str]:
    if segment.source_in < 0 or segment.source_out <= segment.source_in or segment.duration <= 0:
        raise ValueError("invalid V2 source segment range")
    if abs(segment.duration - (segment.source_out - segment.source_in)) > 0.001:
        raise ValueError("V2 source segment duration does not match its range")
    _validate_destination(destination, config, final=False)
    encoder, encoder_args = _encoder(toolchain, config)
    graph = (
        "[0:v:0]setpts=PTS-STARTPTS,"
        f"scale=w='min(iw,{config.output_width})':h='min(ih,{config.output_height})':"
        "force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={config.output_width}:{config.output_height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v]"
    )
    return [
        str(toolchain.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{segment.source_in:.3f}", "-t", f"{segment.duration:.3f}",
        "-i", str(segment.source), "-filter_complex", graph,
        "-map", "[v]", "-map", "0:a:0?", "-fps_mode", "passthrough",
        "-c:v", encoder, *encoder_args, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-b:a", str(config.audio_mix.get("bitrate", "320k")),
        "-t", f"{segment.duration:.3f}", str(destination),
    ]


def choose_audio_overlap(previous: V2EditShot, current: V2EditShot, config: PipelineConfig) -> tuple[int, int, str]:
    """Return one conservative audio-only overlap mode; never overlap impact cuts."""
    if previous.impact_cut or current.impact_cut:
        return 0, 0, "direct"
    low, high = config.audio_overlap_ms
    low = max(100, int(low))
    high = min(250, int(high), int(config.impact_tail_max_ms))
    if high < low or current.transition_compatibility_score < 0.80:
        return 0, 0, "direct"
    requested = current.audio_j_cut_ms or previous.audio_l_cut_ms or low
    overlap = min(high, max(low, int(requested)))
    # Prefer a J-cut when no explicit Task 6 L-cut metadata was retained.
    if current.audio_j_cut_ms or not previous.audio_l_cut_ms:
        return overlap, 0, "j_cut"
    return 0, overlap, "l_cut"


def _audio_chain(segment_count: int, edit: V2EditDecisionList, config: PipelineConfig) -> str:
    if segment_count == 1:
        game_label = "0:a:0"
        prefix = ""
    else:
        prefix_parts: list[str] = []
        current_label = "0:a:0"
        for index in range(1, segment_count):
            mode = "direct"
            if segment_count == len(edit.shots):
                _, _, mode = choose_audio_overlap(edit.shots[index - 1], edit.shots[index], config)
            next_label = f"{index}:a:0"
            output_label = f"game_{index}"
            if mode in {"j_cut", "l_cut"}:
                previous, current = edit.shots[index - 1], edit.shots[index]
                j_cut, l_cut, _ = choose_audio_overlap(previous, current, config)
                overlap = max(j_cut, l_cut) / 1000.0
                prefix_parts.append(f"[{current_label}][{next_label}]acrossfade=d={overlap:.3f}:c1=tri:c2=tri[{output_label}]")
            else:
                prefix_parts.append(f"[{current_label}][{next_label}]concat=n=2:v=0:a=1[{output_label}]")
            current_label = output_label
        game_label = current_label
        prefix = ";".join(prefix_parts) + ";"
    music_index = segment_count
    mix = build_v2_audio_filter(
        game_label, f"{music_index}:a:0", 0.0, float(config.audio_mix.get("music_gain_db", -9.0)),
        duration=edit.duration,
        attack_ms=int(config.audio_mix.get("attack_ms", 50)),
        release_ms=int(config.audio_mix.get("release_ms", 500)),
        target_lufs=float(config.audio_mix.get("target_lufs", -14.0)),
        true_peak_db=float(config.audio_mix.get("true_peak_db", -1.0)),
    )
    return prefix + mix


def compile_v2_final_argv(
    segment_paths: Sequence[Path],
    edit: V2EditDecisionList,
    config: PipelineConfig,
    toolchain: Toolchain,
    destination: Path,
) -> list[str]:
    if not segment_paths:
        raise ValueError("V2 render requires at least one segment")
    if len(segment_paths) != len(edit.shots) and len(edit.shots) > 1:
        # Flattened condensed source segments are valid; overlap metadata is then conservatively skipped.
        pass
    _validate_destination(destination, config, final=True)
    count = len(segment_paths)
    video_inputs = "".join(f"[{index}:v:0]" for index in range(count))
    graph = f"{video_inputs}concat=n={count}:v=1:a=0[v];"
    graph += _audio_chain(count, edit, config)
    encoder, encoder_args = _encoder(toolchain, config)
    argv = [str(toolchain.ffmpeg), "-hide_banner", "-loglevel", "error", "-y"]
    for path in segment_paths:
        argv += ["-i", str(path)]
    argv += ["-ss", f"{edit.music_in:.3f}", "-t", f"{edit.duration:.3f}", "-i", str(edit.music_source)]
    argv += [
        "-filter_complex", graph, "-map", "[v]", "-map", "[aout]",
        "-fps_mode", "passthrough", "-c:v", encoder, *encoder_args,
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000",
        "-b:a", str(config.audio_mix.get("bitrate", "320k")), "-t", f"{edit.duration:.3f}",
        "-movflags", "+faststart", str(destination),
    ]
    return argv


def render_v2_edit(
    edit: V2EditDecisionList,
    config: PipelineConfig,
    toolchain: Toolchain,
    output_path: Path,
) -> Path:
    from .beam_timeline import validate_v2_edit

    _validate_destination(output_path, config, final=True)
    expected = config.v2_output_path.resolve(strict=False)
    if output_path.resolve(strict=False) != expected:
        raise ValueError("V2 renderer may replace only the configured V2 output")
    validate_v2_edit(edit, config)
    if not edit.music_source.is_file():
        raise FileNotFoundError(edit.music_source)
    _preflight_v2_sources(edit, config, toolchain)
    run_dir = config.render_v2_dir / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)
    segment_paths: list[Path] = []
    try:
        for shot_index, shot in enumerate(edit.shots):
            for part_index, segment in enumerate(shot.source_segments):
                assert_source_read_only(segment.source, config.raw_dir)
                destination = run_dir / f"segment_{shot_index:03d}_{part_index:02d}.mp4"
                run_command(compile_v2_segment_argv(segment, config, toolchain, destination), check=True)
                if not destination.is_file() or destination.stat().st_size == 0:
                    raise RuntimeError(f"FFmpeg did not create V2 segment: {destination}")
                segment_paths.append(destination)
        temporary = run_dir / "v2-output.mp4"
        run_command(compile_v2_final_argv(segment_paths, edit, config, toolchain, temporary), check=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not create a non-empty V2 output")
        _validate_rendered_v2_output(temporary, edit, config, toolchain)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_path)
        return output_path
    finally:
        # The cache is intentionally retained for diagnostics; only the output is replaced.
        pass
