from pathlib import Path

import pytest

from montage.workflow import DetectorConfig, EditRules, MontageWorkflow, WorkflowRule, resolve_roi


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


def test_legacy_engine_example_round_trips_through_desktop_adapter() -> None:
    source = MontageWorkflow.import_json(Path("examples/battlefield6_workflow.json"))
    desktop = source.to_desktop_dict()
    restored = MontageWorkflow.from_dict(desktop)

    assert restored.game_id == source.game_id
    assert restored.display_name == source.display_name
    assert restored.detector.to_dict() == source.detector.to_dict()
    assert restored.edit_rules.to_dict() == source.edit_rules.to_dict()
    assert restored.audio_output == source.audio_output
    assert restored.metadata == source.metadata
    assert restored.profiles == source.profiles


def test_rule_engine_label_and_threshold_metadata_survive_desktop_round_trip() -> None:
    detector = DetectorConfig(event_label="ENGINE_EVENT", thresholds={"template": 0.65, "geometry": 0.42})
    source = MontageWorkflow("game", "Game", detector, rules=[WorkflowRule("rule-1", "User label", detector, {"tag": "kept"})])

    restored = MontageWorkflow.from_dict(source.to_desktop_dict())

    assert restored.rules[0].label == "User label"
    assert restored.detector.event_label == "ENGINE_EVENT"
    assert restored.detector.thresholds == {"template": 0.65, "geometry": 0.42}
    assert restored.rules[0].metadata == {"tag": "kept"}


def test_desktop_duplicate_rule_ids_are_normalized() -> None:
    data = {
        "schema": "game-montage-workflow-v1",
        "game": {"id": "apex", "display_name": "Apex"},
        "detectors": {"rules": [
            {"id": "rule", "label": "击杀", "search_roi": [0, 0, 1, 1], "threshold": 0.5, "templates": ["a"], "positive_samples": ["b"]},
            {"id": "rule", "label": "击倒", "search_roi": [0, 0, 1, 1], "threshold": 0.5, "templates": ["c"], "positive_samples": ["d"]},
        ]},
    }

    workflow = MontageWorkflow.from_dict(data)

    assert [rule.id for rule in workflow.rules] == ["rule", "rule-2"]


@pytest.mark.parametrize("thresholds", [{}, {"geometry": 0.4}])
def test_desktop_adapter_does_not_invent_template_threshold(thresholds) -> None:
    detector = DetectorConfig(thresholds=thresholds)
    source = MontageWorkflow("game", "Game", detector)

    restored = MontageWorkflow.from_dict(source.to_desktop_dict())

    assert restored.detector.thresholds == thresholds


@pytest.mark.parametrize("field", ["templates", "positive_samples", "negative_samples"])
def test_detector_sample_fields_must_be_arrays(field: str) -> None:
    detector = DetectorConfig(**{field: "invalid"})

    with pytest.raises(ValueError, match="array"):
        detector.validate()


def test_desktop_workflow_imports_as_multi_rule_engine_contract() -> None:
    desktop = {
        "schema": "game-montage-workflow-v1",
        "game": {"id": "apex-legends", "display_name": "Apex Legends"},
        "detectors": {
            "rules": [
                {
                    "id": "kill",
                    "label": "击杀",
                    "type": "template_match",
                    "search_roi": [0.3, 0.55, 0.48, 0.74],
                    "threshold": 0.65,
                    "templates": ["marker.png"],
                    "positive_samples": ["positive.png"],
                    "negative_samples": ["negative.png"],
                    "metadata": {"source": "apex", "confidence": "high"},
                },
                {
                    "id": "knockdown",
                    "label": "击倒",
                    "type": "template_match",
                    "search_roi": [0.2, 0.4, 0.6, 0.8],
                    "threshold": 0.72,
                    "templates": ["knock-marker.png"],
                    "positive_samples": ["knock-positive.png"],
                    "negative_samples": [],
                    "metadata": {"priority": 2},
                },
            ]
        },
        "edit_rules": {"event_pre_seconds": 2.0, "merge_gap_seconds": 3.0, "long_gap_bridge_seconds": 1.0},
        "audio_output": {"music_gain_db": -18.0, "sample_rate": 44100, "channels": 2},
        "metadata": {"created_by": "test", "editable": True},
    }

    workflow = MontageWorkflow.from_dict(desktop)

    assert [rule.id for rule in workflow.rules] == ["kill", "knockdown"]
    assert workflow.detector.event_label == "击杀"
    assert workflow.rules[0].metadata == {"source": "apex", "confidence": "high"}
    assert workflow.rules[1].detector.thresholds == {"template": 0.72}
    assert workflow.edit_rules.event_pre_seconds == 2.0
    assert workflow.edit_rules.event_post_seconds == 0.5
    assert workflow.audio_output["music_gain_db"] == -18.0

    restored = MontageWorkflow.from_dict(workflow.to_dict())
    assert restored.to_dict() == workflow.to_dict()
    assert restored.to_desktop_dict() == desktop | {
        "edit_rules": {
            "event_pre_seconds": 2.0,
            "event_post_seconds": 0.5,
            "merge_gap_seconds": 3.0,
            "long_gap_bridge_seconds": 1.0,
            "fade_to_black_seconds": 5.0,
        },
        "audio_output": {
            "game_gain_db": -6.0,
            "music_gain_db": -18.0,
            "target_lufs": -16.0,
            "true_peak_db": -2.0,
            "sample_rate": 44100,
            "channels": 2,
        },
    }


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EditRules(event_pre_seconds=float("nan")),
        lambda: EditRules(event_post_seconds=float("inf")),
        lambda: DetectorConfig(roi={"coordinate_space": "normalized", "x1": 0.0, "y1": 0.0, "x2": float("inf"), "y2": 1.0}),
        lambda: DetectorConfig(thresholds={"template": float("nan")}),
    ],
)
def test_non_finite_workflow_values_rejected(factory) -> None:
    with pytest.raises(ValueError, match="finite"):
        factory().validate()
