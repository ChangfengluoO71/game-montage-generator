import logging
from dataclasses import replace
from pathlib import Path

from montage.config import load_config
from montage.media_index import build_media_index, write_media_index
from montage.models import MediaRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def config_for(tmp_path):
    base = load_config(PROJECT_ROOT / "config.yaml")
    return replace(
        base,
        raw_dir=tmp_path / "raw",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        music_file=tmp_path / "music.flac",
    )


def media_record(path: Path, duration: float) -> MediaRecord:
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


def test_media_index_classifies_short_and_long(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    config.raw_dir.mkdir()
    short = config.raw_dir / "短 [15].mp4"
    short.write_bytes(b"x")
    long = config.raw_dir / "long.mp4"
    long.write_bytes(b"y")

    monkeypatch.setattr(
        "montage.media_index.probe_media",
        lambda path, _: media_record(path, 15 if path == short else 301),
    )

    records = build_media_index(config, object(), logging.getLogger("test"))

    assert {record.category for record in records} == {"short_clip", "long_clip"}


def test_media_index_csv_has_requested_columns(tmp_path):
    path = tmp_path / "a.mp4"
    path.write_bytes(b"x")
    record = media_record(path, 20)

    write_media_index([record], tmp_path / "index.json", tmp_path / "index.csv")
    csv_text = (tmp_path / "index.csv").read_text(encoding="utf-8")

    assert "file_path" in csv_text
    assert "source_start" not in csv_text


def test_media_index_reuses_unchanged_probe(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    config.raw_dir.mkdir()
    source = config.raw_dir / "unchanged.mp4"
    source.write_bytes(b"source")
    calls = []

    def fake_probe(path, _):
        calls.append(path)
        return media_record(path, 20)

    monkeypatch.setattr("montage.media_index.probe_media", fake_probe)
    first = build_media_index(config, object(), logging.getLogger("test"))
    write_media_index(first, config.analysis_dir / "media_index.json", config.analysis_dir / "media_index.csv")
    second = build_media_index(config, object(), logging.getLogger("test"))

    assert len(calls) == 1
    assert second[0].file_path == source.resolve()
