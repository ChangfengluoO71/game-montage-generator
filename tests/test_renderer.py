from pathlib import Path

import pytest

from montage.audio_mix import build_audio_filter
from montage.ffmpeg_renderer import compile_concat_argv, compile_shot_argv
from montage.models import EditShot


def edit_shot(
    source=Path(r"D:\91\集锦\raw\战斗 [A].mp4"),
    source_in=2.0,
    source_out=7.0,
    source_duration=None,
):
    return EditShot(
        source=source,
        source_in=source_in,
        source_out=source_out,
        duration=source_out - source_in,
        candidate_score=0.8,
        duplicate_group="dg-001",
        timeline_in=0.0,
        timeline_out=source_out - source_in,
        transition="hard_cut",
        music_target=70.0,
        music_event_type="strong_beat",
        sync_offset=0.041,
        rationale="test shot",
        source_duration=source_duration,
    )


def test_compile_argv_keeps_unicode_source_and_never_uses_shell(fake_toolchain, base_config):
    shot = edit_shot()

    argv = compile_shot_argv(shot, Path(r"C:\音乐\決意.flac"), Path("segment.mp4"), base_config, fake_toolchain)

    assert argv[0].endswith("ffmpeg.exe")
    assert str(shot.source) in argv
    assert "-filter_complex" in argv
    assert "shell=True" not in argv


def test_audio_filter_ducks_music_but_keeps_game_audio():
    graph = build_audio_filter("game", "music", 2.0, -6.0)

    assert "sidechaincompress" in graph
    assert "amix" in graph
    assert "game" in graph and "music" in graph


def test_renderer_rejects_shot_outside_source_duration(fake_toolchain, base_config):
    with pytest.raises(ValueError):
        compile_shot_argv(
            edit_shot(source_in=10.0, source_out=12.0, source_duration=11.0),
            base_config.music_file,
            Path("x.mp4"),
            base_config,
            fake_toolchain,
        )


def test_concat_reencodes_to_avoid_non_monotonic_segment_timestamps(fake_toolchain, base_config):
    argv = compile_concat_argv(Path("concat.txt"), Path("preview.mp4"), base_config, fake_toolchain)

    video_codec_index = argv.index("-c:v")
    assert argv[video_codec_index + 1] != "copy"
    assert "vfr" in argv
