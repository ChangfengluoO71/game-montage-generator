from pathlib import Path

import pytest

from montage.workflow import DetectorConfig, EditRules, MontageWorkflow, resolve_roi


def test_battlefield_example_imports() -> None:
    workflow = MontageWorkflow.import_json(Path("examples/battlefield6_workflow.json"))
    assert workflow.game_id == "battlefield_6"
    assert workflow.edit_rules.event_pre_seconds == 1.5
    assert workflow.detector.event_label == "OWN_KILL"


def test_normalized_roi_maps_across_resolution() -> None:
    roi = {"coordinate_space": "normalized", "x1": 0.3, "y1": 0.55, "x2": 0.48, "y2": 0.74}
    assert resolve_roi(roi, 1920, 1200) == (576, 660, 922, 888)
    assert resolve_roi(roi, 2560, 1600) == (768, 880, 1229, 1184)


def test_invalid_edit_rule_rejected() -> None:
    with pytest.raises(ValueError, match="bridge"):
        EditRules(merge_gap_seconds=1.0, long_gap_bridge_seconds=2.0).validate()


def test_invalid_detector_roi_rejected() -> None:
    with pytest.raises(ValueError, match="ROI"):
        DetectorConfig(roi={"coordinate_space": "normalized", "x1": 0.7, "y1": 0.2, "x2": 0.2, "y2": 0.4}).validate()


def test_workflow_round_trip(tmp_path: Path) -> None:
    source = MontageWorkflow.import_json(Path("examples/battlefield6_workflow.json"))
    exported = tmp_path / "shared-workflow.json"
    source.export_json(exported)
    restored = MontageWorkflow.import_json(exported)
    assert restored.to_dict() == source.to_dict()
