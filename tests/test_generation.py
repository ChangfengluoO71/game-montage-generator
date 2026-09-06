from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from montage.generation import (
    GenerationEvent,
    GenerationFailure,
    GenerationRequest,
    run_generation,
)
from montage.workflow import DetectorConfig, MontageWorkflow, WorkflowRule


def _workflow(detector_type: str = "template_match") -> MontageWorkflow:
    detector = DetectorConfig(
        detector_type=detector_type,
        event_label="击杀",
        roi={"coordinate_space": "normalized", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        templates=[],
        positive_samples=[],
        negative_samples=[],
        thresholds={"template": 0.7},
    )
    return MontageWorkflow(
        "test-game",
        "Test Game",
        detector,
        rules=[WorkflowRule("kill", "击杀", detector, {"source": "test"})],
    )


def test_generation_request_snapshots_workflow_and_uses_explicit_sources(tmp_path: Path) -> None:
    source_dir = tmp_path / "素材"
    source_dir.mkdir()
    source = source_dir / "片段.mp4"
    source.write_bytes(b"fixture")
    workflow = _workflow()
    request = GenerationRequest(workflow, source_dir, tmp_path / "work", tmp_path / "output", source_paths=(source,))

    assert request.source_paths == (source,)
    snapshot = request.workflow_snapshot()
    workflow.rules[0].metadata["changed"] = True
    assert snapshot.rules[0].metadata == {"source": "test"}


def test_generation_rejects_write_destination_inside_selected_sources(tmp_path: Path) -> None:
    source_dir = tmp_path / "素材"
    source_dir.mkdir()
    request = GenerationRequest(_workflow(), source_dir, source_dir / "work", tmp_path / "output")

    with pytest.raises(ValueError, match="source"):
        run_generation(request)


def test_generation_persists_source_ledger_and_event_provenance(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "素材"
    source_dir.mkdir()
    source = source_dir / "clip.mp4"
    source.write_bytes(b"fixture")
    request = GenerationRequest(_workflow(), source_dir, tmp_path / "work", tmp_path / "output", source_paths=(source,))

    class Record:
        file_path = source.resolve()
        file_name = source.name
        file_size = source.stat().st_size
        duration = 4.0
        width = 1280
        height = 720
        fps = 30.0
        codec = "h264"
        bitrate = None
        audio_codec = None
        audio_channels = None
        audio_sample_rate = None
        creation_time = None
        category = "short_clip"
        fingerprint = {"size": file_size}
        probe_error = None

    monkeypatch.setattr("montage.generation.probe_media", lambda path, toolchain: Record())
    monkeypatch.setattr("montage.generation.discover_toolchain", lambda config: object())
    from montage.kill_truth.scanner import source_id_for_path

    monkeypatch.setattr(
        "montage.generation.scan_template_source",
        lambda record, rule, toolchain, progress=None, **kwargs: [
            GenerationEvent(source_id_for_path(record.file_path), "evt-1", 1.25, rule.id, rule.label, 0.91, {"positive_samples": []})
        ],
    )

    result = run_generation(request)

    assert result.events[0].rule_id == "kill"
    assert result.events[0].label == "击杀"
    assert result.events[0].confidence == 0.91
    assert result.events_path.is_file()
    assert result.source_ledger_path.is_file()
    assert json.loads(result.events_path.read_text(encoding="utf-8"))["events"][0]["event_id"] == "evt-1"
    ledger = json.loads(result.source_ledger_path.read_text(encoding="utf-8"))
    assert ledger["sources"][0]["source_path"] == str(source.resolve())
    assert ledger["sources"][0]["width"] == 1280


def test_generation_rejects_music_without_audio_before_scanning(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "素材"
    source_dir.mkdir()
    source = source_dir / "clip.mp4"
    source.write_bytes(b"fixture")
    music = tmp_path / "video-only.mp4"
    music.write_bytes(b"fixture")
    request = GenerationRequest(
        _workflow(),
        source_dir,
        tmp_path / "work",
        tmp_path / "output",
        music,
        source_paths=(source,),
    )

    class Toolchain:
        ffprobe = "ffprobe.exe"

    monkeypatch.setattr("montage.generation.discover_toolchain", lambda config: Toolchain())
    monkeypatch.setattr(
        "montage.generation.run_command",
        lambda *args, **kwargs: __import__("subprocess").CompletedProcess(args[0], 0, '{"streams":[]}', ""),
    )
    monkeypatch.setattr("montage.generation.probe_media", lambda *args, **kwargs: pytest.fail("video scan started"))

    with pytest.raises(GenerationFailure, match="no audio stream"):
        run_generation(request)


def test_template_match_uses_unicode_path_and_rising_edge(tmp_path: Path) -> None:
    from montage.template_detection import scan_template_frames

    template = np.zeros((10, 12), dtype=np.uint8)
    template[2:8, 3:9] = 255
    template_path = tmp_path / "标志模板.png"
    ok, encoded = cv2.imencode(".png", template)
    assert ok
    encoded.tofile(str(template_path))
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    frame[15:25, 20:32] = template[:, :, None]
    frames = [(0.0, frame), (0.1, frame), (0.2, np.zeros_like(frame)), (0.3, frame)]

    events = scan_template_frames(frames, (0.0, 0.0, 1.0, 1.0), [template_path], threshold=0.8, source_id="src", rule_id="rule", label="击杀")

    assert [event.timestamp for event in events] == [0.0, 0.3]


def test_template_match_does_not_rearm_on_one_noisy_frame(tmp_path: Path) -> None:
    from montage.template_detection import scan_template_frames

    template = np.zeros((8, 8), dtype=np.uint8)
    template[2:6, 2:6] = 255
    template_path = tmp_path / "marker.png"
    ok, encoded = cv2.imencode(".png", template)
    assert ok
    encoded.tofile(str(template_path))
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[12:20, 12:20] = template[:, :, None]
    frames = [(0.000, frame), (0.033, np.zeros_like(frame)), (0.066, frame)]

    events = scan_template_frames(
        frames,
        (0.0, 0.0, 1.0, 1.0),
        [template_path],
        threshold=0.8,
        source_id="src",
        rule_id="rule",
        label="击杀",
    )

    assert [event.timestamp for event in events] == [0.0]


def test_partial_error_is_a_generation_failure(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "素材"
    source_dir.mkdir()
    source = source_dir / "clip.mp4"
    source.write_bytes(b"fixture")
    request = GenerationRequest(_workflow(), source_dir, tmp_path / "work", tmp_path / "output", source_paths=(source,))

    class Record:
        file_path = source.resolve()
        file_name = source.name
        file_size = 1
        duration = 1.0
        width = 1280
        height = 720
        fps = 30.0
        codec = "h264"
        bitrate = audio_codec = audio_channels = audio_sample_rate = creation_time = None
        category = "short_clip"
        fingerprint = {}
        probe_error = None

    monkeypatch.setattr("montage.generation.probe_media", lambda path, toolchain: Record())
    monkeypatch.setattr("montage.generation.discover_toolchain", lambda config: object())
    monkeypatch.setattr("montage.generation.scan_template_source", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("PARTIAL_ERROR: decode")))

    with pytest.raises(GenerationFailure, match="decode"):
        run_generation(request)


def test_generation_fails_when_every_implicit_folder_source_is_undecodable(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "素材"
    source_dir.mkdir()
    (source_dir / "broken.mp4").write_bytes(b"not-video")
    request = GenerationRequest(_workflow(), source_dir, tmp_path / "work", tmp_path / "output")
    monkeypatch.setattr("montage.generation.discover_toolchain", lambda config: object())
    monkeypatch.setattr("montage.generation.probe_media", lambda path, toolchain: (_ for _ in ()).throw(RuntimeError("bad container")))

    with pytest.raises(GenerationFailure, match="No selected video"):
        run_generation(request)


def test_generation_exports_project_and_renders_when_requested(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "素材"
    source_dir.mkdir()
    source = source_dir / "clip.mp4"
    source.write_bytes(b"fixture")
    request = replace(
        GenerationRequest(_workflow(), source_dir, tmp_path / "work", tmp_path / "output", source_paths=(source,)),
        render=True,
    )

    from montage.kill_truth.scanner import source_id_for_path

    class Record:
        file_path = source.resolve()
        file_name = source.name
        file_size = source.stat().st_size
        duration = 4.0
        width = 1280
        height = 720
        fps = 30.0
        codec = "h264"
        bitrate = None
        audio_codec = None
        audio_channels = None
        audio_sample_rate = None
        creation_time = None
        category = "short_clip"
        fingerprint = {"size": file_size}
        probe_error = None

    monkeypatch.setattr("montage.generation.probe_media", lambda path, toolchain: Record())
    monkeypatch.setattr("montage.generation.discover_toolchain", lambda config: object())
    monkeypatch.setattr(
        "montage.generation.scan_template_source",
        lambda record, rule, toolchain, progress=None, **kwargs: [
            GenerationEvent(source_id_for_path(record.file_path), "evt-1", 1.25, rule.id, rule.label, 0.91, {})
        ],
    )

    def fake_render(project, config, toolchain, output_path, *, progress=None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        if progress:
            progress(100, "rendered")
        return output_path

    monkeypatch.setattr("montage.generation.render_project", fake_render)
    result = run_generation(request)

    assert result.project_path is not None and result.project_path.is_file()
    assert result.output_path is not None and result.output_path.is_file()
    assert result.status == "OK"
    assert json.loads((result.run_dir / "result.json").read_text(encoding="utf-8"))["output_path"] == str(result.output_path.resolve())
