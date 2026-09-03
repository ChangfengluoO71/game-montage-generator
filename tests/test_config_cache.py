import json
from pathlib import Path

import pytest

from montage.cache import atomic_write_json, cache_key
from montage.config import assert_source_read_only, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_config_resolves_audited_unicode_paths():
    config = load_config(PROJECT_ROOT / "config.yaml")

    assert config.raw_dir == Path(r"D:\91\集锦\raw")
    assert config.music_file.name == "01. 決意の唄.flac"
    assert config.output_width == 1920
    assert config.output_height == 1200


def test_cache_key_changes_when_mtime_changes():
    first = cache_key(
        {"absolute_path": "C:/x.mp4", "size": 10, "mtime": 1.0},
        "probe",
        {},
    )
    second = cache_key(
        {"absolute_path": "C:/x.mp4", "size": 10, "mtime": 2.0},
        "probe",
        {},
    )

    assert first != second


def test_raw_path_is_allowed_only_as_a_source(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "片段 [01].mp4"
    source.write_bytes(b"source")

    assert_source_read_only(source, raw)
    with pytest.raises(ValueError):
        assert_source_read_only(raw / "nested" / "generated.mp4", raw)


def test_atomic_json_write_replaces_complete_file(tmp_path):
    target = tmp_path / "analysis.json"

    atomic_write_json(target, {"ok": True})

    assert json.loads(target.read_text(encoding="utf-8"))["ok"] is True
    assert not list(tmp_path.glob("*.tmp"))
