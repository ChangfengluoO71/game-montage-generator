from pathlib import Path

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
