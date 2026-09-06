from pathlib import Path
import subprocess
from dataclasses import replace

import pytest

from montage.project_renderer import render_project
from montage.workflow import EditableProject, MontageWorkflow, TimelineClip


def _project(tmp_path: Path, *, music: Path | None = None, has_audio: bool = True) -> EditableProject:
    source = tmp_path / "raw" / "game [01].mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    workflow = MontageWorkflow(
        "game",
        "Game",
        audio_output={
            "game_gain_db": -4.0,
            "music_gain_db": -18.0,
            "target_lufs": -16.0,
            "true_peak_db": -2.0,
            "sample_rate": 44100,
            "channels": 1,
        },
    )
    return EditableProject(
        "project",
        workflow,
        [
            TimelineClip("c1", "s1", 1.0, 3.0, 0.0, 2.0),
            TimelineClip("c2", "s1", 5.0, 7.0, 2.0, 4.0),
        ],
        [
            {
                "source_id": "s1",
                "source_path": str(source),
                "duration": 10.0,
                "width": 1920,
                "height": 1200,
                "fps": 60.0,
                "has_audio": has_audio,
            }
        ],
        str(music) if music else None,
        {"width": 1280, "height": 720, "fps": 30.0, "fade_to_black_seconds": 10.0},
    )


def test_render_project_compiles_settings_and_timeline_music(tmp_path, base_config, fake_toolchain, monkeypatch):
    music = tmp_path / "music.mp4"
    music.write_bytes(b"music")
    project = _project(tmp_path, music=music)
    commands: list[list[str]] = []

    def fake_run(argv, **kwargs):
        commands.append([str(value) for value in argv])
        output = Path(str(argv[-1]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"encoded")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("montage.project_renderer.run_command", fake_run)
    output = render_project(project, base_config, fake_toolchain, tmp_path / "output" / "result.mp4")

    assert output.exists()
    segment = commands[0]
    assert "-stream_loop" in segment and "-ss" in segment
    graph = segment[segment.index("-filter_complex") + 1]
    assert "-4.000dB" in graph and "-18.000dB" in graph
    assert "scale=w='min(iw,1280)':h='min(ih,720)'" in graph
    assert "fps=30.000" in graph
    assert "asplit=2" in graph
    assert "-ar" in segment and segment[segment.index("-ar") + 1] == "44100"
    assert "-ac" in segment and segment[segment.index("-ac") + 1] == "1"
    ss_positions = [index for index, value in enumerate(commands[0]) if value == "-ss"]
    assert [commands[0][index + 1] for index in ss_positions] == ["1.000", "0.000"]
    assert "fade=t=out" in commands[2][commands[2].index("-filter_complex") + 1]
    assert "st=0.000:d=4.000" in commands[2][commands[2].index("-filter_complex") + 1]
    assert commands[2][-1].endswith(".tmp.mp4")


def test_render_project_supports_no_music_and_missing_game_audio(tmp_path, base_config, fake_toolchain, monkeypatch):
    project = _project(tmp_path, has_audio=False)
    commands: list[list[str]] = []

    def fake_run(argv, **kwargs):
        commands.append([str(value) for value in argv])
        output = Path(str(argv[-1]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"encoded")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("montage.project_renderer.run_command", fake_run)
    render_project(project, base_config, fake_toolchain, tmp_path / "output" / "silent.mp4")

    segment = commands[0]
    assert "-stream_loop" not in segment
    graph = segment[segment.index("-filter_complex") + 1]
    assert "anullsrc" in graph
    assert "amix" not in graph


def test_render_project_rejects_invalid_ranges_and_raw_output_before_ffmpeg(tmp_path, base_config, fake_toolchain, monkeypatch):
    project = _project(tmp_path)
    project.clips[1].timeline_in = 2.1
    project.clips[1].timeline_out = 4.1
    monkeypatch.setattr("montage.project_renderer.run_command", lambda *args, **kwargs: pytest.fail("ffmpeg called"))
    with pytest.raises(ValueError, match="contiguous"):
        render_project(project, base_config, fake_toolchain, tmp_path / "output" / "result.mp4")

    project = _project(tmp_path)
    with pytest.raises(ValueError, match="raw"):
        render_project(project, base_config, fake_toolchain, tmp_path / "raw" / "result.mp4")


def test_render_project_records_ffmpeg_stderr_and_keeps_existing_output(tmp_path, base_config, fake_toolchain, monkeypatch):
    project = _project(tmp_path)
    output = tmp_path / "output" / "result.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"old")

    def fail_run(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr="decoder exploded")

    monkeypatch.setattr("montage.project_renderer.run_command", fail_run)
    with pytest.raises(RuntimeError, match="FFmpeg"):
        render_project(project, base_config, fake_toolchain, output)
    assert output.read_bytes() == b"old"
    logs = list((base_config.work_dir / "logs").glob("project-renderer-*.stderr.log"))
    assert logs and "decoder exploded" in logs[-1].read_text(encoding="utf-8")


def test_render_project_rejects_output_aliasing_selected_source(tmp_path, base_config, fake_toolchain, monkeypatch):
    music = tmp_path / "music.mp4"
    music.write_bytes(b"music")
    project = _project(tmp_path, music=music)
    source = Path(project.source_ledger[0]["source_path"]).resolve()
    config = replace(base_config, raw_dir=tmp_path / "different-raw")
    monkeypatch.setattr("montage.project_renderer.run_command", lambda *args, **kwargs: pytest.fail("ffmpeg called"))

    for alias in (source, music):
        with pytest.raises(ValueError, match="source or music"):
            render_project(project, config, fake_toolchain, alias)

    assert source.read_bytes() == b"source"
    assert music.read_bytes() == b"music"
