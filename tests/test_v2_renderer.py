from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from montage.audio_mix import build_v2_audio_filter
from montage.models import PayoffEvent, SourceSegment, V2EditDecisionList, V2EditShot
from montage.v2_renderer import (
    choose_audio_overlap,
    compile_v2_final_argv,
    compile_v2_segment_argv,
    render_v2_edit,
)


def _shot(source: Path, index: int, *, impact=False, compatibility=0.9) -> V2EditShot:
    start = float(index * 5)
    segment = SourceSegment(source, start, start + 5.0, 5.0)
    anchor = PayoffEvent(f"e{index}", "kill", start + 4.0, 0.9, 0.9, 0.8, {"impact": 0.8} if impact else {})
    return V2EditShot(
        source=source, source_in=start, source_out=start + 5.0, duration=5.0,
        candidate_score=0.8, duplicate_group=f"g{index}", timeline_in=start,
        timeline_out=start + 5.0, transition="hard_cut", music_target=19.0 + start,
        music_event_type="beat", sync_offset=0.0, rationale="test", source_duration=100.0,
        source_segments=(segment,), parent_candidate_id=f"c{index}", variant_id=f"v{index}",
        primary_anchor=anchor, context_integrity_score=1.0,
        transition_compatibility_score=compatibility, impact_cut=impact,
    )


def _edit(source: Path, music: Path) -> V2EditDecisionList:
    shots = tuple(_shot(source, index) for index in range(9))
    return V2EditDecisionList(
        kind="preview_v2", music_source=music, baseline_music_in=19.0,
        baseline_music_out=74.252, music_in=19.0, music_out=74.252,
        duration=45.0, music_reason="baseline", shots=shots,
    )


def test_segment_argv_fits_without_upscale_and_keeps_source_cadence(fake_toolchain, base_config, tmp_path):
    source = tmp_path / "clip.mp4"
    destination = base_config.work_dir / "segment.mp4"
    argv = compile_v2_segment_argv(SourceSegment(source, 2.0, 7.0, 5.0), base_config, fake_toolchain, destination)

    graph = argv[argv.index("-filter_complex") + 1]
    assert "min(iw,1920)" in graph and "min(ih,1200)" in graph
    assert "scale=w=1920" not in graph
    assert "-fps_mode" in argv and argv[argv.index("-fps_mode") + 1] == "passthrough"
    assert "-r" not in argv
    assert "-map" in argv and "0:a:0?" in argv


def test_final_argv_has_hard_cut_video_and_single_music_input(fake_toolchain, base_config, tmp_path):
    source = tmp_path / "clip.mp4"
    music = tmp_path / "music.flac"
    segment_paths = [tmp_path / f"segment-{i}.mp4" for i in range(2)]
    edit = replace(_edit(source, music), shots=(_shot(source, 0), _shot(source, 1)))
    argv = compile_v2_final_argv(segment_paths, edit, base_config, fake_toolchain, base_config.work_dir / "out.mp4")
    graph = argv[argv.index("-filter_complex") + 1]

    assert graph.count("concat=n=2") == 1
    assert "xfade=" not in graph and "glitch" not in graph and "rgb" not in graph
    assert "sidechaincompress" in graph and "acrossfade" in graph
    assert argv.count(str(music)) == 1
    assert "-ar" in argv and argv[argv.index("-ar") + 1] == "48000"


def test_audio_overlap_is_conservative_and_impact_edges_are_direct(base_config, tmp_path):
    source = tmp_path / "clip.mp4"
    previous = _shot(source, 0)
    current = _shot(source, 1, impact=True)
    assert choose_audio_overlap(previous, current, base_config) == (0, 0, "direct")

    current = _shot(source, 1, compatibility=0.95)
    j_cut, l_cut, mode = choose_audio_overlap(previous, current, base_config)
    assert mode in {"direct", "j_cut", "l_cut"}
    assert (j_cut == 0) ^ (l_cut == 0) or (j_cut, l_cut) == (0, 0)
    if j_cut or l_cut:
        assert 100 <= max(j_cut, l_cut) <= 250
        assert max(j_cut, l_cut) <= base_config.impact_tail_max_ms


def test_audio_filter_has_smooth_ducking_and_never_mutes_gameplay():
    graph = build_v2_audio_filter("game", "music", 0.0, -9.0, duration=45.0)
    assert "sidechaincompress" in graph
    assert "attack=50" in graph and "release=500" in graph
    assert "amix=inputs=2" in graph
    assert "volume=0dB" not in graph


def test_renderer_rejects_raw_destination_before_running_ffmpeg(fake_toolchain, base_config, tmp_path):
    source = tmp_path / "clip.mp4"
    music = tmp_path / "music.flac"
    edit = _edit(source, music)
    raw_destination = base_config.raw_dir / "preview_60s_v2.mp4"
    with pytest.raises(ValueError, match="raw"):
        render_v2_edit(edit, base_config, fake_toolchain, raw_destination)


@pytest.mark.parametrize("destination", [
    "baseline",
    "output_dir",
    "external",
])
def test_segment_argv_rejects_destinations_outside_work_policy(fake_toolchain, base_config, tmp_path, destination):
    source = tmp_path / "clip.mp4"
    destinations = {
        "baseline": base_config.baseline_output_path,
        "output_dir": base_config.output_dir,
        "external": tmp_path / "outside" / "segment.mp4",
    }
    with pytest.raises(ValueError):
        compile_v2_segment_argv(
            SourceSegment(source, 0.0, 1.0, 1.0), base_config, fake_toolchain, destinations[destination]
        )


@pytest.mark.parametrize("destination", [
    "baseline",
    "output_dir",
    "external",
])
def test_final_argv_rejects_destinations_except_work_intermediate(fake_toolchain, base_config, tmp_path, destination):
    source = tmp_path / "clip.mp4"
    music = tmp_path / "music.flac"
    edit = _edit(source, music)
    segment = base_config.work_dir / "segment.mp4"
    destinations = {
        "baseline": base_config.baseline_output_path,
        "output_dir": base_config.output_dir,
        "external": tmp_path / "outside" / "final.mp4",
    }
    with pytest.raises(ValueError):
        compile_v2_final_argv([segment], edit, base_config, fake_toolchain, destinations[destination])


def test_renderer_preflights_all_sources_with_selected_ffprobe_before_rendering(
    fake_toolchain, base_config, tmp_path, monkeypatch
):
    test_config = replace(
        base_config,
        raw_dir=tmp_path / "raw",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
    )
    source = test_config.raw_dir / "task8-preflight-a.mp4"
    second_source = test_config.raw_dir / "task8-preflight-b.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"fixture")
    second_source.write_bytes(b"fixture")
    music = tmp_path / "music.flac"
    music.write_bytes(b"fixture")
    edit = _edit(source, music)
    shots = list(edit.shots)
    shots[-1] = _shot(second_source, 8)
    edit = replace(edit, shots=tuple(shots))
    commands = []

    def fake_run(argv, **kwargs):
        commands.append([str(value) for value in argv])
        if str(argv[0]) == str(fake_toolchain.ffprobe):
            if str(source) in [str(value) for value in argv]:
                stdout = '{"streams":[{"codec_type":"video"},{"codec_type":"audio"}],"format":{"duration":"100.0"}}'
            else:
                stdout = '{"streams":[{"codec_type":"video"}],"format":{"duration":"100.0"}}'
            return CompletedProcess(argv, 0, stdout=stdout, stderr="")
        raise AssertionError("render command must not run after failed preflight")

    monkeypatch.setattr("montage.v2_renderer.run_command", fake_run)
    with pytest.raises(ValueError, match="video and audio"):
        render_v2_edit(edit, test_config, fake_toolchain, test_config.v2_output_path)
    assert len(commands) == 2
    assert commands[0][0] == str(fake_toolchain.ffprobe)
    assert commands[1][0] == str(fake_toolchain.ffprobe)


def test_cpu_toolchain_uses_safe_h264_fallback(fake_toolchain, base_config, tmp_path):
    cpu = replace(fake_toolchain, nvenc_h264=False)
    argv = compile_v2_segment_argv(
        SourceSegment(tmp_path / "clip.mp4", 0.0, 1.0, 1.0), base_config, cpu, base_config.work_dir / "out.mp4"
    )
    assert argv[argv.index("-c:v") + 1] == "libx264"
