from dataclasses import replace
from pathlib import Path

import pytest

from montage.config import load_config
from montage.models import MediaRecord
from montage.toolchain import Toolchain


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def base_config(tmp_path):
    config = load_config(PROJECT_ROOT / "config.yaml")
    return replace(
        config,
        raw_dir=tmp_path / "raw",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
    )


@pytest.fixture
def fake_toolchain():
    return Toolchain(
        ffmpeg=Path("ffmpeg.exe"),
        ffprobe=Path("ffprobe.exe"),
        ffmpeg_version="8.0",
        ffprobe_version="8.0",
        nvenc_h264=True,
        nvenc_hevc=True,
        selected_reason="test",
        candidates=[],
    )


def make_record(path: Path, duration: float) -> MediaRecord:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"fixture")
    return MediaRecord(
        file_path=path,
        file_name=path.name,
        file_size=path.stat().st_size,
        duration=duration,
        width=1920,
        height=1200,
        fps=60.0,
        codec="h264",
        bitrate=1000,
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=48000,
        creation_time=None,
        category="short_clip" if duration <= 90 else "long_clip",
        fingerprint={"absolute_path": str(path), "size": path.stat().st_size, "mtime": path.stat().st_mtime},
    )


@pytest.fixture
def short_record(tmp_path):
    return make_record(tmp_path / "raw" / "片段 [15].mp4", 15.0)


@pytest.fixture
def long_record(tmp_path):
    return make_record(tmp_path / "raw" / "long.mp4", 301.0)
