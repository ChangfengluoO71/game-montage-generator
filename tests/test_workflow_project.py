import json
from pathlib import Path

import pytest

from montage.workflow import EditableProject, MontageWorkflow, TimelineClip


def test_editable_project_round_trip(tmp_path: Path) -> None:
    workflow = MontageWorkflow.import_json(Path("examples/battlefield6_workflow.json"))
    project = EditableProject(
        "demo-project",
        workflow,
        [TimelineClip("clip-1", "source-1", 1.5, 4.0, 0.0, 2.5, ["event-1"], "rapid skull refresh", "pending")],
        [{"source_id": "source-1", "status": "partially_used"}],
        "music.mp4",
        {"width": 1920, "height": 1200},
    )
    path = tmp_path / "project.json"
    project.export_json(path)
    restored = EditableProject.import_json(path)
    assert restored.to_dict() == project.to_dict()


def _project(clips=None, ledger=None, project_id="p1") -> EditableProject:
    workflow = MontageWorkflow.import_json(Path("examples/battlefield6_workflow.json"))
    return EditableProject(
        project_id,
        workflow,
        clips if clips is not None else [TimelineClip("c1", "source-1", 1.0, 3.0, 0.0, 2.0, ["e1"])],
        ledger if ledger is not None else [{"source_id": "source-1", "duration": 10.0}],
    )


def test_timeline_clip_rejects_non_finite_values_and_bad_event_ids() -> None:
    for clip in (
        TimelineClip("c1", "s1", float("nan"), 2.0, 0.0, 2.0),
        TimelineClip("c1", "s1", 0.0, float("inf"), 0.0, 2.0),
        TimelineClip("c1", "s1", 0.0, 2.0, 0.0, 2.0, "e1"),
        TimelineClip("c1", "s1", 0.0, 2.0, 0.0, 2.0, ["e1", 2]),
    ):
        with pytest.raises(ValueError):
            clip.validate()


def test_editable_project_validates_containers_ids_contiguous_timelines_and_bounds() -> None:
    with pytest.raises(ValueError, match="project_id"):
        _project(project_id=None).to_dict()
    with pytest.raises(ValueError, match="source_ledger"):
        _project(ledger="invalid").to_dict()
    with pytest.raises(ValueError, match="duplicate clip"):
        _project(clips=[TimelineClip("same", "s1", 0.0, 1.0, 0.0, 1.0), TimelineClip("same", "s1", 1.0, 2.0, 1.0, 2.0)], ledger=[{"source_id": "s1", "duration": 3.0}]).to_dict()
    with pytest.raises(ValueError, match="contiguous"):
        _project(clips=[TimelineClip("c1", "s1", 0.0, 1.0, 0.0, 1.0), TimelineClip("c2", "s1", 1.0, 2.0, 1.1, 2.1)], ledger=[{"source_id": "s1", "duration": 3.0}]).to_dict()
    with pytest.raises(ValueError, match="unknown source"):
        _project(clips=[TimelineClip("c1", "missing", 0.0, 1.0, 0.0, 1.0)]).to_dict()
    with pytest.raises(ValueError, match="source bounds"):
        _project(clips=[TimelineClip("c1", "source-1", 0.0, 11.0, 0.0, 11.0)]).to_dict()


def test_editable_project_import_rejects_null_project_id_and_string_ledger(tmp_path: Path) -> None:
    project = _project()
    payload = project.to_dict()
    payload["project_id"] = None
    null_path = tmp_path / "null.json"
    null_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="project_id"):
        EditableProject.import_json(null_path)

    payload = project.to_dict()
    payload["source_ledger"] = "invalid"
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source_ledger"):
        EditableProject.import_json(ledger_path)
