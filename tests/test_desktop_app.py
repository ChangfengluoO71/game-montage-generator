from pathlib import Path

from montage.desktop_app import CustomProfile, DetectionRule


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
