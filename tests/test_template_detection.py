from pathlib import Path
from types import SimpleNamespace

import numpy as np

from montage.template_detection import TemplateEvent, scan_template_source


def _rule(tmp_path: Path):
    template = tmp_path / "marker.png"
    template.write_bytes(b"fixture")
    detector = SimpleNamespace(
        templates=[str(template)],
        positive_samples=[],
        roi={"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
        thresholds={"template": 0.8},
    )
    return SimpleNamespace(detector=detector, id="kill", label="kill")


def test_template_source_scales_decode_and_reuses_cache(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    record = SimpleNamespace(file_path=source, width=1920, height=1200, fps=60.0, duration=10.0)
    rule = _rule(tmp_path)
    calls: list[tuple[int, int, float]] = []
    expected = TemplateEvent(1.25, 0.91, "source", "kill", "kill", {"detector": "template_match"})

    def fake_frames(toolchain, path, *, width, height, fps, scale_width=None, scale_height=None):
        calls.append((scale_width or width, scale_height or height, fps))
        yield 0.0, np.zeros((scale_height or height, scale_width or width, 3), dtype=np.uint8)

    monkeypatch.setattr("montage.template_detection.iter_full_frames", fake_frames)
    monkeypatch.setattr(
        "montage.template_detection.scan_template_frames",
        lambda frames, *args, **kwargs: (list(frames), [expected])[1],
    )
    cache_dir = tmp_path / "cache"

    first = scan_template_source(record, rule, object(), cache_dir=cache_dir, fps_cap=12.0, decode_height=720)
    assert first == [expected]
    assert calls == [(1152, 720, 12.0)]

    monkeypatch.setattr(
        "montage.template_detection.iter_full_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    second = scan_template_source(record, rule, object(), cache_dir=cache_dir, fps_cap=12.0, decode_height=720)
    assert second == [expected]
