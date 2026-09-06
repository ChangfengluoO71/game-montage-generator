from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from montage.desktop_app import CustomProfile, DetectionRule, MontageLab, ProfileWizard
from montage.workflow import DetectorConfig, EditRules, MontageWorkflow, WorkflowRule


def test_custom_profile_serializes_multiple_rules(tmp_path: Path) -> None:
    def rule(name: str, marker: str, positive: str) -> DetectionRule:
        return DetectionRule(name, tmp_path / marker, tmp_path / positive, None, (0.3, 0.55, 0.48, 0.74), 0.65)

    profile = CustomProfile("Apex Legends", [rule("击杀", "kill.png", "kill-full.png"), rule("击倒", "knock.png", "knock-full.png")])
    workflow = profile.workflow()
    assert profile.game_id == "apex-legends"
    assert workflow["schema"] == "game-montage-workflow-v1"
    assert [item["label"] for item in workflow["detectors"]["rules"]] == ["击杀", "击倒"]


def test_detection_rule_keeps_normalized_roi(tmp_path: Path) -> None:
    item = DetectionRule("击杀", tmp_path / "m.png", tmp_path / "p.png", None, (0.3, 0.55, 0.48, 0.74), 0.65).to_dict()
    assert item["type"] == "template_match"
    assert item["search_roi"] == [0.3, 0.55, 0.48, 0.74]


def test_profile_wizard_saves_two_same_game_rules_that_engine_can_import(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)

    def save(rule_name: str, marker: str, positive: str) -> None:
        wizard = ProfileWizard(tmp_path)
        wizard.name.setText("Apex Legends")
        wizard.event_label.setText(rule_name)
        wizard.marker_path.setText(str(tmp_path / marker))
        wizard.positive_path.setText(str(tmp_path / positive))
        wizard.save_profile()
        wizard.close()

    save("击杀", "kill.png", "kill-full.png")
    save("击倒", "knock.png", "knock-full.png")

    saved = tmp_path / "apex-legends-workflow.json"
    workflow = MontageWorkflow.import_json(saved)
    assert workflow.game_id == "apex-legends"
    assert [rule.label for rule in workflow.rules] == ["击杀", "击倒"]
    assert len({rule.id for rule in workflow.rules}) == 2
    assert workflow.rules[1].detector.templates == [str(tmp_path / "knock.png")]
    assert workflow.metadata["editable"] is True
    del app


def test_profile_wizard_append_preserves_existing_engine_rule_and_workflow_fields(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    detector = DetectorConfig(
        detector_type="skull_row",
        event_label="ENGINE_KILL",
        templates=[{"kind": "normal", "file": "old-normal.png"}, {"kind": "headshot", "file": "old-headshot.png"}],
        positive_samples=["old-positive.png"],
        negative_samples=["old-negative.png"],
        thresholds={"template": 0.65, "geometry": 0.91, "shape": 0.37, "color": 0.14},
    )
    existing = MontageWorkflow(
        "apex_legacy",
        "Apex Legends",
        detector,
        EditRules(event_pre_seconds=2.25, merge_gap_seconds=3.5, long_gap_bridge_seconds=1.25),
        {"game_gain_db": -4.0, "music_gain_db": -19.0, "sample_rate": 44100, "channels": 2},
        profiles=[{"profile_id": "apex-1080p", "width": 1920, "height": 1080}],
        metadata={"owner": "qa", "revision": 7},
        rules=[WorkflowRule("engine-kill", "Engine kill", detector, {"detector_note": "retain"})],
    )
    saved = tmp_path / "apex-legends-workflow.json"
    existing.export_desktop_json(saved)

    wizard = ProfileWizard(tmp_path)
    wizard.name.setText("Apex Legends")
    wizard.event_label.setText("击倒")
    wizard.marker_path.setText(str(tmp_path / "new-marker.png"))
    wizard.positive_path.setText(str(tmp_path / "new-positive.png"))
    wizard.save_profile()
    wizard.close()

    restored = MontageWorkflow.import_json(saved)
    assert restored.game_id == "apex_legacy"
    assert [rule.id for rule in restored.rules] == ["engine-kill", "rule"]
    assert restored.rules[0].detector.templates == detector.templates
    assert restored.rules[0].detector.thresholds == detector.thresholds
    assert restored.rules[0].metadata == {"detector_note": "retain"}
    assert restored.profiles == existing.profiles
    assert restored.metadata == existing.metadata
    assert restored.edit_rules.to_dict() == existing.edit_rules.to_dict()
    assert restored.audio_output == existing.audio_output | {"target_lufs": -16.0, "true_peak_db": -2.0}
    del app


def test_select_music_rejects_file_without_audio_before_persisting(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    window = MontageLab()
    window.settings.clear()
    selected = tmp_path / "video-only.mp4"
    selected.write_bytes(b"fixture")
    warnings: list[str] = []

    monkeypatch.setattr(
        "montage.desktop_app.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(selected), ""),
    )
    monkeypatch.setattr(window, "_validate_music_file", lambda path: (False, "no audio stream"))
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(str(args[2] if len(args) > 2 else "")),
    )

    window.select_music()

    assert window.music_file.text() == ""
    assert window.settings.value("music_file", "") == ""
    assert warnings and "no audio stream" in warnings[0]
    window.close()
    del app
