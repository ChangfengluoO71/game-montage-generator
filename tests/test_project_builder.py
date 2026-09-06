from __future__ import annotations

from pathlib import Path

import pytest

from montage.project_builder import build_project
from montage.workflow import DetectorConfig, EditRules, MontageWorkflow, WorkflowRule


def _workflow(*, pre: float = 2.0, post: float = 2.0, merge: float = 2.0, bridge: float = 1.0, fade: float = 4.0) -> MontageWorkflow:
    detector = DetectorConfig(event_label="HIGHLIGHT")
    return MontageWorkflow("apex", "Apex", detector, EditRules(pre, post, merge, bridge, fade))


def _ledger(*durations: tuple[str, float]) -> list[dict]:
    return [{"source_id": source_id, "source_path": f"D:/raw/{source_id}.mp4", "duration": duration, "width": 1920, "height": 1080, "fps": 60.0, "has_audio": True} for source_id, duration in durations]


def _event(event_id: str, source_id: str, timestamp: float, rule_id: str = "kill", label: str = "击杀", confidence: float = 0.9) -> dict:
    return {"event_id": event_id, "source_id": source_id, "timestamp": timestamp, "rule_id": rule_id, "label": label, "confidence": confidence}


def test_build_project_bounds_windows_and_assigns_contiguous_timeline() -> None:
    project = build_project(
        _workflow(pre=3.0, post=4.0, bridge=0.0),
        [_event("e1", "a", 1.0), _event("e2", "a", 28.0)],
        _ledger(("a", 30.0)),
        project_id="p1",
        music_source="D:/music.mp3",
        render_settings={"encoder": "libx264"},
    )

    assert [(clip.source_in, clip.source_out) for clip in project.clips] == [(0.0, 5.0), (25.0, 30.0)]
    assert [(clip.timeline_in, clip.timeline_out) for clip in project.clips] == [(0.0, 5.0), (5.0, 10.0)]
    assert project.clips[0].event_ids == ["e1"]
    assert project.clips[0].review_status == "pending"
    assert "击杀" in project.clips[0].ai_reason
    assert project.music_source == "D:/music.mp3"
    assert project.render_settings == {"encoder": "libx264", "fade_to_black_seconds": 4.0}


def test_build_project_merges_overlapping_and_nearby_same_source_events() -> None:
    project = build_project(
        _workflow(pre=1.0, post=1.0, merge=2.0),
        [_event("e2", "a", 5.0), _event("e1", "a", 1.0), _event("e3", "a", 7.0)],
        _ledger(("a", 20.0)),
        project_id="p1",
    )

    assert len(project.clips) == 1
    assert (project.clips[0].source_in, project.clips[0].source_out) == (0.0, 8.0)
    assert project.clips[0].event_ids == ["e1", "e2", "e3"]


def test_build_project_bridges_large_gap_without_retaining_middle() -> None:
    project = build_project(
        _workflow(pre=1.0, post=1.0, merge=2.0, bridge=2.0),
        [_event("e1", "a", 5.0), _event("e2", "a", 15.0)],
        _ledger(("a", 30.0)),
        project_id="p1",
    )

    assert [(clip.source_in, clip.source_out) for clip in project.clips] == [(4.0, 7.0), (13.0, 16.0)]
    assert project.clips[0].event_ids == ["e1"]
    assert project.clips[1].event_ids == ["e2"]


def test_build_project_never_merges_across_sources_and_uses_ledger_order() -> None:
    project = build_project(
        _workflow(pre=1.0, post=1.0),
        [_event("b-event", "b", 4.0), _event("a-event", "a", 4.0)],
        _ledger(("a", 10.0), ("b", 10.0)),
        project_id="p1",
    )

    assert [clip.source_id for clip in project.clips] == ["a", "b"]
    assert project.clips[0].timeline_out == project.clips[1].timeline_in


def test_build_project_deep_copies_workflow_and_ledger() -> None:
    workflow = _workflow()
    ledger = _ledger(("a", 10.0))
    project = build_project(workflow, [_event("e1", "a", 4.0)], ledger, project_id="p1")

    workflow.metadata["changed"] = True
    ledger[0]["duration"] = 99.0
    assert "changed" not in project.workflow.metadata
    assert project.source_ledger[0]["duration"] == 10.0


def test_build_project_rejects_empty_and_malformed_events() -> None:
    with pytest.raises(ValueError, match="at least one event"):
        build_project(_workflow(), [], _ledger(("a", 10.0)), project_id="p1")
    with pytest.raises(ValueError, match="unknown source"):
        build_project(_workflow(), [_event("e1", "missing", 1.0)], _ledger(("a", 10.0)), project_id="p1")
    with pytest.raises(ValueError, match="duplicate event"):
        build_project(_workflow(), [_event("e1", "a", 1.0), _event("e1", "a", 2.0)], _ledger(("a", 10.0)), project_id="p1")
    with pytest.raises(ValueError, match="finite"):
        build_project(_workflow(), [_event("e1", "a", float("nan"))], _ledger(("a", 10.0)), project_id="p1")


def test_build_project_rejects_unknown_rule_for_multi_rule_workflow() -> None:
    detector = DetectorConfig(event_label="Kill")
    workflow = MontageWorkflow("apex", "Apex", detector, rules=[WorkflowRule("kill", "Kill", detector)])

    with pytest.raises(ValueError, match="unknown rule"):
        build_project(workflow, [_event("e1", "a", 4.0, rule_id="knockdown")], _ledger(("a", 10.0)), project_id="p1")


def test_build_project_rejects_degenerate_source_ranges() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        build_project(_workflow(pre=0.0, post=0.0), [_event("e1", "a", 0.0)], _ledger(("a", 10.0)), project_id="p1")


def test_build_project_empty_music_and_render_settings_are_preserved() -> None:
    project = build_project(_workflow(), [_event("e1", "a", 4.0)], _ledger(("a", 10.0)), project_id="p1")

    assert project.music_source is None
    assert project.render_settings["fade_to_black_seconds"] == 4.0


def test_build_project_project_json_round_trip(tmp_path: Path) -> None:
    project = build_project(_workflow(), [_event("e1", "a", 4.0)], _ledger(("a", 10.0)), project_id="p1")
    path = tmp_path / "project.json"
    project.export_json(path)

    from montage.workflow import EditableProject

    assert EditableProject.import_json(path).to_dict() == project.to_dict()
