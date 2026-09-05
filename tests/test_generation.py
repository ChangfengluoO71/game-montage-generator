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
    monkeypatch.setattr(
        "montage.generation.scan_template_source",
        lambda record, rule, toolchain, progress=None: [
            GenerationEvent(record.file_path.stem, "evt-1", 1.25, rule.id, rule.label, 0.91, {"positive_samples": []})
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
