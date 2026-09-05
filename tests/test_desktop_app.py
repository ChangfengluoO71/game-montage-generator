from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from montage.desktop_app import CustomProfile, DetectionRule, ProfileWizard
from montage.workflow import MontageWorkflow


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
