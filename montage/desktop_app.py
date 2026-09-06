"""Montage Lab local desktop application.

Run with ``python -m montage.desktop_app``.  This native Qt application is
local-first: it stores preferences and custom detector workflows below the
user's application-data directory and never writes into selected RAW folders.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QTextEdit,
    QWidget,
)

from .workflow import DEFAULT_AUDIO_OUTPUT, MontageWorkflow, WorkflowRule
from .generation import GenerationRequest, GenerationResult
from .generation_worker import GenerationWorker

APP_ORGANIZATION = "MontageLab"
APP_NAME = "GameMontageGenerator"
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm"}
DEFAULT_EDIT_RULES = {
    "event_pre_seconds": 1.5,
    "event_post_seconds": 0.5,
    "merge_gap_seconds": 2.0,
    "long_gap_bridge_seconds": 2.0,
    "fade_to_black_seconds": 5.0,
    "allow_early_end": True,
}


@dataclass
class DetectionRule:
    name: str
    marker_sample: Path
    positive_sample: Path
    negative_sample: Path | None
    roi: tuple[float, float, float, float]
    threshold: float
    rule_id: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str | None:
        return self.rule_id

    def to_dict(self, *, rule_id: str | None = None) -> dict:
        if not all(math.isfinite(float(value)) for value in self.roi + (self.threshold,)):
            raise ValueError("rule ROI and threshold must be finite")
        if not 0 <= self.threshold <= 1:
            raise ValueError("rule threshold must be between 0 and 1")
        return {"id": rule_id or self.rule_id or "rule", "label": self.name, "type": "template_match", "search_roi": list(self.roi), "threshold": self.threshold, "templates": [str(self.marker_sample)], "positive_samples": [str(self.positive_sample)], "negative_samples": [str(self.negative_sample)] if self.negative_sample else [], "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: dict) -> "DetectionRule":
        roi = data.get("search_roi")
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            raise ValueError("desktop rule search_roi must contain four values")
        templates = data.get("templates", [])
        if not isinstance(templates, list) or not templates:
            raise ValueError("desktop rule requires a template sample")
        positive = data.get("positive_samples", [])
        if not isinstance(positive, list) or not positive:
            raise ValueError("desktop rule requires a positive sample")
        negative = data.get("negative_samples", [])
        marker = templates[0].get("file", "") if isinstance(templates[0], dict) else templates[0]
        if not marker:
            raise ValueError("desktop rule template sample requires a file")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("desktop rule metadata must be an object")
        return cls(str(data.get("label", "高光事件")), Path(marker), Path(positive[0]), Path(negative[0]) if negative else None, tuple(float(value) for value in roi), float(data.get("threshold", 0.65)), str(data.get("id", "rule")), dict(metadata))

@dataclass
class CustomProfile:
    name: str
    rules: list[DetectionRule]
    edit_rules: dict = field(default_factory=dict)
    audio_output: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=lambda: {"created_by": "Montage Lab desktop application", "editable": True})

    @property
    def game_id(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-") or "custom-game"

    def workflow(self) -> dict:
        seen: set[str] = set()
        serialized_rules = []
        for rule in self.rules:
            base_id = rule.rule_id or re.sub(r"[^a-z0-9]+", "-", rule.name.lower()).strip("-") or "rule"
            rule_id = base_id
            suffix = 2
            while rule_id in seen:
                rule_id = f"{base_id}-{suffix}"
                suffix += 1
            seen.add(rule_id)
            serialized_rules.append(rule.to_dict(rule_id=rule_id))
        edit_rules = dict(DEFAULT_EDIT_RULES); edit_rules.update(self.edit_rules)
        audio_output = dict(DEFAULT_AUDIO_OUTPUT); audio_output.update(self.audio_output)
        result = {"schema": "game-montage-workflow-v1", "game": {"id": self.game_id, "display_name": self.name}, "detectors": {"rules": serialized_rules}, "edit_rules": edit_rules, "audio_output": audio_output, "metadata": dict(self.metadata)}
        return result

    @classmethod
    def from_workflow_dict(cls, data: dict) -> "CustomProfile":
        workflow = MontageWorkflow.from_dict(data)
        desktop = workflow.to_desktop_dict()
        rules = [DetectionRule.from_dict(item) for item in desktop["detectors"]["rules"]]
        return cls(workflow.display_name, rules, desktop.get("edit_rules", {}), desktop.get("audio_output", {}), dict(desktop.get("metadata", {})))


class ProfileWizard(QDialog):
    """A real modal, four-step custom profile wizard."""

    def __init__(self, profiles_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.profiles_dir = profiles_dir
        self.setWindowTitle("创建自定义游戏配置")
        self.setMinimumSize(680, 480)
        self.stack = QStackedWidget()
        self.status = QLabel("步骤 1 / 4：游戏信息")
        self.status.setObjectName("subtitle")
        self.next_button = QPushButton("下一步")
        self.back_button = QPushButton("返回")
        self.back_button.clicked.connect(self.go_back)
        self.next_button.clicked.connect(self.go_next)
        self._build_pages()
        buttons = QHBoxLayout()
        buttons.addWidget(self.back_button)
        buttons.addStretch()
        buttons.addWidget(self.next_button)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("自创游戏配置", objectName="title"))
        layout.addWidget(QLabel("创建的配置会保存到本机，并可导出为 JSON 分享。"))
        layout.addWidget(self.status)
        layout.addWidget(self.stack, 1)
        layout.addLayout(buttons)
        self._refresh_buttons()

    def _build_pages(self) -> None:
        self.name = QLineEdit()
        self.name.setPlaceholderText("例如：The Finals")
        self.event_label = QLineEdit("击杀")
        info = QWidget(); form = QFormLayout(info)
        form.addRow("游戏名称 *", self.name)
        form.addRow("规则名称 *", self.event_label)
        form.addRow(QLabel("同一个游戏可添加多条规则，例如 Apex 的“击杀”和“击倒”。检测实现由应用自动选择，无需用户判断。"))
        self.stack.addWidget(info)

        samples = QWidget(); sample_layout = QFormLayout(samples)
        self.marker_path = QLineEdit(); self.marker_path.setReadOnly(True)
        self.positive_path = QLineEdit(); self.positive_path.setReadOnly(True)
        self.negative_path = QLineEdit(); self.negative_path.setReadOnly(True)
        sample_layout.addRow("击杀标志区域图 *", self._picker(self.marker_path, "选择标志图片", "Images (*.png *.jpg *.jpeg *.webp)"))
        sample_layout.addRow("完整分辨率正例 *", self._picker(self.positive_path, "选择正例画面", "Media (*.png *.jpg *.jpeg *.webp *.mp4 *.mkv *.mov)"))
        sample_layout.addRow("负样本（可选）", self._picker(self.negative_path, "选择负样本", "Media (*.png *.jpg *.jpeg *.webp *.mp4 *.mkv *.mov)"))
        hint = QLabel("标志图是命中时 HUD 的裁剪图；正例是包含该标志的完整画面。RAW 文件不会被改写。")
        hint.setWordWrap(True); sample_layout.addRow(hint)
        self.stack.addWidget(samples)

        roi = QWidget(); roi_layout = QVBoxLayout(roi)
        roi_layout.addWidget(QLabel("检测区域（normalized ROI）", objectName="heading"))
        roi_layout.addWidget(QLabel("值范围为 0 到 1；同一个配置可映射到不同分辨率。"))
        roi_form = QFormLayout(); self.roi_boxes = []
        for label, value in (("X1", 0.30), ("Y1", 0.55), ("X2", 0.48), ("Y2", 0.74)):
            box = QLineEdit(str(value)); self.roi_boxes.append(box); roi_form.addRow(label, box)
        self.threshold = QSlider(Qt.Orientation.Horizontal); self.threshold.setRange(10, 99); self.threshold.setValue(65)
        self.threshold_label = QLabel("0.65")
        self.threshold.valueChanged.connect(lambda value: self.threshold_label.setText(f"{value / 100:.2f}"))
        threshold_layout = QHBoxLayout(); threshold_layout.addWidget(self.threshold); threshold_layout.addWidget(self.threshold_label)
        roi_form.addRow("匹配阈值", threshold_layout)
        roi_layout.addLayout(roi_form); roi_layout.addStretch()
        self.stack.addWidget(roi)

        self.review = QLabel(); self.review.setWordWrap(True)
        review_page = QWidget(); review_layout = QVBoxLayout(review_page); review_layout.addWidget(QLabel("确认保存", objectName="heading")); review_layout.addWidget(self.review); review_layout.addStretch()
        self.stack.addWidget(review_page)

    def _picker(self, target: QLineEdit, caption: str, filter_text: str) -> QWidget:
        holder = QWidget(); layout = QHBoxLayout(holder); layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("选择文件")
        button.clicked.connect(lambda: self._pick_file(target, caption, filter_text))
        layout.addWidget(target, 1); layout.addWidget(button)
        return holder

    def _pick_file(self, target: QLineEdit, caption: str, filter_text: str) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, caption, "", filter_text)
        if selected:
            target.setText(selected)

    def _profile(self) -> CustomProfile:
        values = tuple(float(box.text()) for box in self.roi_boxes)
        rule = DetectionRule(
            name=self.event_label.text().strip() or "高光事件",
            marker_sample=Path(self.marker_path.text()),
            positive_sample=Path(self.positive_path.text()),
            negative_sample=Path(self.negative_path.text()) if self.negative_path.text() else None,
            roi=values,
            threshold=self.threshold.value() / 100,
        )
        return CustomProfile(name=self.name.text().strip(), rules=[rule])

    def _validate(self, page: int) -> bool:
        if page == 0 and not self.name.text().strip():
            QMessageBox.warning(self, "缺少信息", "请填写游戏名称。"); return False
        if page == 1 and (not self.marker_path.text() or not self.positive_path.text()):
            QMessageBox.warning(self, "缺少样本", "请至少选择击杀标志区域图和完整分辨率正例。"); return False
        if page == 2:
            try: x1, y1, x2, y2 = self._profile().rules[0].roi
            except ValueError: QMessageBox.warning(self, "ROI 无效", "ROI 必须是数字。"); return False
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                QMessageBox.warning(self, "ROI 无效", "ROI 必须在 0~1，且左上角坐标小于右下角。"); return False
        return True

    def go_next(self) -> None:
        page = self.stack.currentIndex()
        if page < 3 and not self._validate(page): return
        if page == 2:
            profile = self._profile()
            rule = profile.rules[0]
            self.review.setText(
                f"Game: {profile.name}\nRule: {rule.name}\nROI: {rule.roi}\n"
                f"Threshold: {rule.threshold:.2f}\nMarker: {rule.marker_sample.name}\n"
                f"Positive: {rule.positive_sample.name}"
            )
        if page == 3:
            self.save_profile(); return
        self.stack.setCurrentIndex(page + 1); self._refresh_buttons()

    def go_back(self) -> None:
        self.stack.setCurrentIndex(max(0, self.stack.currentIndex() - 1)); self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        page = self.stack.currentIndex(); self.back_button.setEnabled(page > 0)
        self.next_button.setText("保存配置" if page == 3 else "下一步")
        self.status.setText(f"步骤 {page + 1} / 4：{['游戏信息', '上传样本', '检测区域', '确认保存'][page]}")

    def save_profile(self) -> None:
        profile = self._profile(); self.profiles_dir.mkdir(parents=True, exist_ok=True)
        output = self.profiles_dir / f"{profile.game_id}-workflow.json"
        if output.exists():
            existing = MontageWorkflow.import_json(output)
            incoming = MontageWorkflow.from_dict(profile.workflow())
            existing_rules = list(existing.rules)
            if not existing_rules:
                existing_rules = [WorkflowRule("legacy-detector", existing.detector.event_label, existing.detector)]
            used_ids = {rule.id for rule in existing_rules}
            new_rule = incoming.rules[0]
            base_id = new_rule.id
            suffix = 2
            while new_rule.id in used_ids:
                new_rule.id = f"{base_id}-{suffix}"
                suffix += 1
            merged = MontageWorkflow(
                existing.game_id,
                existing.display_name,
                existing_rules[0].detector,
                existing.edit_rules,
                existing.audio_output,
                existing.profiles,
                existing.metadata,
                existing_rules + [new_rule],
            )
            merged.export_desktop_json(output)
        else:
            output.write_text(json.dumps(profile.workflow(), ensure_ascii=False, indent=2), encoding="utf-8")
        QMessageBox.information(self, "已保存", f"配置已保存到：\n{output}")
        self.accept()


class MontageLab(QMainWindow):
    def __init__(self) -> None:
        super().__init__(); self.setWindowTitle("Montage Lab — Game Montage Generator"); self.resize(980, 660)
        self.settings = QSettings(APP_ORGANIZATION, APP_NAME)
        self.data_dir = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir = self.data_dir / "profiles"
        self._migrate_legacy_profiles()
        self._generation_thread: QThread | None = None
        self._generation_worker: GenerationWorker | None = None
        self._music_validation_toolchain = None
        self.last_generation_result: GenerationResult | None = None
        self._deferred_close = False
        self._build_ui(); self._restore_preferences(); self.refresh_profiles()

    def _build_ui(self) -> None:
        root = QWidget(); self.setCentralWidget(root); layout = QVBoxLayout(root)
        header = QHBoxLayout(); header.addWidget(QLabel("Montage Lab", objectName="title")); header.addStretch(); self.settings_button = QPushButton("全局设置")
        self.settings_button.clicked.connect(self.open_settings); header.addWidget(self.settings_button); layout.addLayout(header)
        self.steps = QStackedWidget(); layout.addWidget(self.steps, 1)
        self._build_game_page(); self._build_media_page(); self._build_rules_page(); self._build_ready_page()
        footer = QHBoxLayout(); self.back = QPushButton("← 返回"); self.next = QPushButton("下一步 →"); self.back.clicked.connect(self.previous); self.next.clicked.connect(self.advance); footer.addWidget(self.back); footer.addStretch(); footer.addWidget(self.next); layout.addLayout(footer); self._refresh_navigation()

    def _build_game_page(self) -> None:
        page = QWidget(); layout = QVBoxLayout(page); layout.addWidget(QLabel("步骤 1：选择游戏配置", objectName="heading")); layout.addWidget(QLabel("选择已有配置，或为一个游戏创建多条高光规则。"))
        self.game_buttons = QButtonGroup(self); self.game_buttons.setExclusive(True)
        self.profile_list = QVBoxLayout(); layout.addLayout(self.profile_list)
        custom = QPushButton("＋ 添加自定义游戏 / 新规则"); custom.clicked.connect(self.open_custom_profile); layout.addWidget(custom)
        layout.addWidget(QLabel("同一个游戏可添加多条规则，例如 Apex Legends / 击杀、击倒。"))
        imported = QPushButton("↥ 导入外部配置 JSON"); imported.clicked.connect(self.import_profile); layout.addWidget(imported); layout.addStretch(); self.steps.addWidget(page)

    def _build_media_page(self) -> None:
        page = QWidget(); form = QFormLayout(page); form.addRow(QLabel("步骤 2：素材与音乐", objectName="heading"))
        self.video_folder = QLineEdit(); self.video_folder.setReadOnly(True); self.video_count = QLabel("未选择素材")
        choose_video = QPushButton("选择扫描文件夹"); choose_video.clicked.connect(self.select_video_folder)
        choose_file = QPushButton("选择单个视频试产"); choose_file.clicked.connect(self.select_video_file)
        row = QHBoxLayout(); row.addWidget(self.video_folder); row.addWidget(choose_video); row.addWidget(choose_file); form.addRow("视频素材文件夹", row); form.addRow("扫描计数", self.video_count)
        self.music_file = QLineEdit(); self.music_file.setReadOnly(True); choose_music = QPushButton("选择背景音乐"); choose_music.clicked.connect(self.select_music); row2 = QHBoxLayout(); row2.addWidget(self.music_file); row2.addWidget(choose_music); form.addRow("背景音乐（独立）", row2); form.addRow(QLabel("视频文件夹只会扫描，不会改写 RAW；MP4 音乐会在后续 worker 中提取第一条音轨。")); self.steps.addWidget(page)

    def _build_rules_page(self) -> None:
        page = QWidget()
        form = QFormLayout(page)
        form.addRow(QLabel("步骤 3：剪辑规则", objectName="heading"))
        self.rule_boxes = {}
        for key, label, value in (("pre", "事件前导（秒）", 1.5), ("post", "事件后保留（秒）", .5), ("merge", "合并阈值（秒）", 2.0), ("bridge", "桥接长度（秒）", 2.0), ("fade", "提前结束淡出（秒）", 5.0)):
            box = QLineEdit(str(value))
            self.rule_boxes[key] = box
            form.addRow(label, box)
        self.allow_early_end = QCheckBox("素材不足时允许提前结束（画面淡黑、音频淡出）")
        self.allow_early_end.setChecked(True)
        form.addRow("素材不足策略", self.allow_early_end)
        self.steps.addWidget(page)

    def _build_ready_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("步骤 4：准备生成", objectName="heading"))
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.generate_button = QPushButton("开始生成 AI 初剪")
        self.generate_button.clicked.connect(self.start_generation)
        layout.addWidget(self.generate_button)
        self.generation_progress = QProgressBar()
        self.generation_progress.setRange(0, 100)
        layout.addWidget(self.generation_progress)
        self.generation_status = QLabel()
        self.generation_status.setWordWrap(True)
        layout.addWidget(self.generation_status)
        self.generation_log = QTextEdit()
        self.generation_log.setReadOnly(True)
        self.generation_log.setMaximumHeight(150)
        layout.addWidget(self.generation_log)
        self.generation_result = QLabel("尚未生成结果")
        self.generation_result.setWordWrap(True)
        layout.addWidget(self.generation_result)
        self.open_output_button = QPushButton("打开输出目录")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self.open_output_directory)
        layout.addWidget(self.open_output_button)
        layout.addStretch()
        self.steps.addWidget(page)

    def refresh_profiles(self) -> None:
        for button in self.game_buttons.buttons():
            self.game_buttons.removeButton(button)
        while self.profile_list.count():
            item = self.profile_list.takeAt(0); widget = item.widget()
            if widget: widget.deleteLater()
        builtin = QPushButton("Battlefield 6 · Skull Row（内置）"); builtin.setCheckable(True); self.game_buttons.addButton(builtin); self.profile_list.addWidget(builtin)
        builtin.setProperty("workflow_path", str(Path(__file__).resolve().parents[1] / "examples" / "battlefield6_workflow.json"))
        builtin.clicked.connect(lambda checked=False, button=builtin: self._select_profile(button))
        for path in sorted(self.profiles_dir.glob("*-workflow.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8")); game = data.get("game", {}).get("display_name", path.stem); rules = data.get("detectors", {}).get("rules", [])
                labels = ", ".join(r.get("label", "Unnamed") for r in rules)
                button = QPushButton(f"{game} | {len(rules)} rules | {labels}")
                button.setProperty("workflow_path", str(path))
                button.setCheckable(True); self.game_buttons.addButton(button); self.profile_list.addWidget(button)
                button.clicked.connect(lambda checked=False, selected=button: self._select_profile(selected))
            except (OSError, json.JSONDecodeError):
                continue

        last_path = str(self.settings.value("last_workflow_path", ""))
        buttons = self.game_buttons.buttons()
        selected = next((button for button in buttons if str(button.property("workflow_path")) == last_path), buttons[0] if buttons else None)
        if selected is not None:
            selected.setChecked(True)

    def _select_profile(self, button: QPushButton) -> None:
        path = button.property("workflow_path")
        if path:
            self.settings.setValue("last_workflow_path", str(path))

    def _selected_workflow(self) -> MontageWorkflow:
        button = self.game_buttons.checkedButton()
        path_value = button.property("workflow_path") if button else None
        if path_value:
            return MontageWorkflow.import_json(Path(str(path_value)))
        return MontageWorkflow.import_json(Path(__file__).resolve().parents[1] / "examples" / "battlefield6_workflow.json")

    def start_generation(self) -> None:
        if self._generation_thread is not None and self._generation_thread.isRunning():
            return
        source = Path(self.video_folder.text().strip())
        if not source.is_dir():
            self.generation_status.setText("Select an existing source directory before starting.")
            return
        music_text = self.music_file.text().strip()
        music_source = Path(music_text).resolve(strict=False) if music_text else None
        if music_source is not None:
            valid, message = self._validate_music_file(music_source)
            if not valid:
                self.generation_status.setText(f"Music selection is invalid: {message}")
                return
        try:
            workflow = self._selected_workflow()
            from dataclasses import replace
            from .workflow import EditRules
            values = {"event_pre_seconds": float(self.rule_boxes["pre"].text()), "event_post_seconds": float(self.rule_boxes["post"].text()), "merge_gap_seconds": float(self.rule_boxes["merge"].text()), "long_gap_bridge_seconds": float(self.rule_boxes["bridge"].text()), "fade_to_black_seconds": float(self.rule_boxes["fade"].text()), "allow_early_end": self.allow_early_end.isChecked()}
            workflow = replace(workflow, edit_rules=EditRules(**values))
            from .config import load_config
            pipeline = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
            selected_button = self.game_buttons.checkedButton()
            workflow_path = Path(str(selected_button.property("workflow_path"))) if selected_button and selected_button.property("workflow_path") else None
            request = GenerationRequest(
                workflow,
                source,
                pipeline.work_dir,
                pipeline.output_dir,
                music_source,
                source_paths=self._selected_source_paths,
                workflow_path=workflow_path,
                render=True,
            )
        except Exception as exc:
            self.generation_status.setText(f"Generation setup failed: {exc}")
            return
        self.generation_log.clear()
        self.last_generation_result = None
        self.generation_progress.setValue(0)
        self.generation_status.setText("Scanning in background…")
        self.generation_result.setText("正在构建项目和渲染 MP4…")
        self.open_output_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.back.setEnabled(False)
        self.next.setEnabled(False)
        self.settings_button.setEnabled(False)
        self._generation_thread = QThread(self)
        self._generation_worker = GenerationWorker(request)
        self._generation_worker.moveToThread(self._generation_thread)
        self._generation_thread.started.connect(self._generation_worker.run)
        self._generation_worker.progress.connect(self._on_generation_progress)
        self._generation_worker.log.connect(self._on_generation_log)
        self._generation_worker.succeeded.connect(self._on_generation_success)
        self._generation_worker.failed.connect(self._on_generation_failure)
        self._generation_worker.finished.connect(self._generation_thread.quit)
        self._generation_worker.finished.connect(self._generation_worker.deleteLater)
        self._generation_thread.finished.connect(self._generation_thread.deleteLater)
        self._generation_thread.finished.connect(self._generation_finished)
        self._generation_thread.start()

    def _on_generation_progress(self, percent: int, message: str) -> None:
        self.generation_progress.setValue(percent)
        self.generation_status.setText(message)

    def _on_generation_log(self, message: str) -> None:
        self.generation_log.append(message)

    def _on_generation_success(self, result: GenerationResult) -> None:
        self.last_generation_result = result
        self.generation_progress.setValue(100)
        self.generation_status.setText(f"{result.status}: {len(result.events)} events · {result.run_dir}")
        self.generation_result.setText(
            f"项目：{result.project_path or '未生成'}\n输出：{result.output_path or '未渲染'}"
        )
        self.open_output_button.setEnabled(bool(result.output_path))
        if result.output_path:
            self.settings.setValue("last_output_path", str(result.output_path))
        for diagnostic in result.diagnostics:
            self.generation_log.append(diagnostic)

    def _on_generation_failure(self, message: str) -> None:
        self.generation_status.setText(f"Generation failed: {message}")
        self.generation_result.setText("生成失败；请查看下方日志和 diagnostics.json。")
        self.generation_log.append(message)

    def _generation_finished(self) -> None:
        self.generate_button.setEnabled(True)
        self.settings_button.setEnabled(True)
        self._refresh_navigation()
        self._generation_worker = None
        self._generation_thread = None
        if self._deferred_close:
            self._deferred_close = False
            self.close()

    def closeEvent(self, event) -> None:
        if self._generation_thread is not None and self._generation_thread.isRunning():
            self._deferred_close = True
            self.generation_status.setText("Finishing current scan before closing…")
            event.ignore()
            return
        event.accept()

    def open_custom_profile(self) -> None:
        wizard = ProfileWizard(self.profiles_dir, self)
        if wizard.exec():
            profile_path = self.profiles_dir / f"{wizard._profile().game_id}-workflow.json"
            self.settings.setValue("last_custom_profile", wizard.name.text().strip())
            self.settings.setValue("last_workflow_path", str(profile_path))
            self.refresh_profiles()

    def import_profile(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "导入工作流配置", "", "Workflow JSON (*.json)")
        if not selected:
            return
        try:
            workflow = MontageWorkflow.import_json(Path(selected))
            target = self.profiles_dir / f"{workflow.game_id}-workflow.json"
            suffix = 2
            while target.exists():
                target = self.profiles_dir / f"{workflow.game_id}-{suffix}-workflow.json"
                suffix += 1
            workflow.export_desktop_json(target)
            self.settings.setValue("last_workflow_path", str(target))
            self.refresh_profiles()
            QMessageBox.information(self, "已导入", f"配置已保存到：\n{target}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "导入失败", str(exc))

    def select_video_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select source folder")
        if selected:
            self._selected_source_paths = None
            self.video_folder.setText(str(Path(selected).resolve(strict=False)))
            self.settings.setValue("video_folder", selected)
            count = sum(1 for path in Path(selected).rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
            self.video_count.setText(f"{count} 个视频")

    def select_video_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "选择单个视频试产", "", "Video (*.mp4 *.mkv *.mov *.webm)")
        if selected:
            path = Path(selected).resolve(strict=False)
            self._selected_source_paths = (path,)
            self.video_folder.setText(str(path.parent))
            self.video_count.setText(f"单个试产：{path.name}")
            self.settings.setValue("video_folder", str(path.parent))

    def select_music(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select background music",
            "",
            "Audio/Video (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.mp4 *.mkv *.mov *.webm)",
        )
        if not selected:
            return
        path = Path(selected).resolve(strict=False)
        valid, message = self._validate_music_file(path)
        if not valid:
            QMessageBox.warning(
                self,
                "\u97f3\u4e50\u6587\u4ef6\u65e0\u97f3\u8f68",
                message or "The selected file has no audio stream. Choose an audio-bearing file.",
            )
            return
        self.music_file.setText(str(path))
        self.settings.setValue("music_file", str(path))

    def _validate_music_file(self, path: Path) -> tuple[bool, str]:
        """Probe a user-selected music container before it enters the workflow."""
        if not path.is_file():
            return False, f"Music file does not exist: {path}"
        try:
            from .config import load_config
            from .generation import _music_has_audio
            from .toolchain import discover_toolchain

            if self._music_validation_toolchain is None:
                config_path = Path(__file__).resolve().parents[1] / "config.yaml"
                self._music_validation_toolchain = discover_toolchain(load_config(config_path))
            if not _music_has_audio(path, self._music_validation_toolchain):
                return False, f"Selected file has no audio stream: {path}"
        except Exception as exc:
            return False, f"Unable to inspect selected music: {exc}"
        return True, ""

    def open_settings(self) -> None:
        dialog = QDialog(self); dialog.setWindowTitle("全局设置"); form = QFormLayout(dialog)
        language = QComboBox(); language.addItems(["简体中文", "English"]); language.setCurrentText(self.settings.value("language", "简体中文"))
        theme = QComboBox(); theme.addItems(["Dark Tactical", "Light", "High Contrast"]); theme.setCurrentText(self.settings.value("theme", "Dark Tactical"))
        bound = QLineEdit(self.settings.value("video_folder", "")); bound.setReadOnly(True); bind_button = QPushButton("绑定已有游戏的素材文件夹")
        bind_button.clicked.connect(lambda: self._bind_global_folder(bound)); form.addRow("语言", language); form.addRow("主题", theme); form.addRow("默认素材文件夹", bound); form.addRow(bind_button)
        save = QPushButton("保存设置"); save.clicked.connect(lambda: (self.settings.setValue("language", language.currentText()), self.settings.setValue("theme", theme.currentText()), dialog.accept())); form.addRow(save); dialog.exec()

    def _bind_global_folder(self, target: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(self, "绑定 Battlefield 6 素材目录")
        if selected: target.setText(selected); self.settings.setValue("video_folder", selected)

    def _restore_preferences(self) -> None:
        folder = self.settings.value("video_folder", "")
        self._selected_source_paths: tuple[Path, ...] | None = None
        if folder:
            self.video_folder.setText(str(folder))
            path = Path(str(folder))
            if path.is_dir():
                count = sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES)
                self.video_count.setText(f"{count} 个视频")
        music = self.settings.value("music_file", "")
        if music:
            path = Path(str(music)).resolve(strict=False)
            valid, _ = self._validate_music_file(path)
            if valid:
                self.music_file.setText(str(path))
            else:
                self.settings.remove("music_file")

    def _migrate_legacy_profiles(self) -> None:
        legacy = Path("D:/HKEY_CURRENT_USER/Software/MontageLab/profiles")
        if not legacy.is_dir():
            return
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        for source in legacy.glob("*-workflow.json"):
            target = self.profiles_dir / source.name
            if not target.exists():
                try:
                    shutil.copy2(source, target)
                except OSError:
                    continue
            if str(self.settings.value("last_workflow_path", "")) == str(source):
                self.settings.setValue("last_workflow_path", str(target))

    def open_output_directory(self) -> None:
        output = Path(self.settings.value("last_output_path", ""))
        if not output:
            return
        directory = output if output.is_dir() else output.parent
        if directory.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def advance(self) -> None:
        current = self.steps.currentIndex()
        if current == 1 and not self.video_folder.text(): QMessageBox.warning(self, "缺少素材", "请选择视频素材文件夹。"); return
        if current == 2:
            try: bridge = float(self.rule_boxes["bridge"].text()); merge = float(self.rule_boxes["merge"].text()); fade = float(self.rule_boxes["fade"].text())
            except ValueError: QMessageBox.warning(self, "规则无效", "剪辑规则必须是数字。"); return
            if bridge > merge: QMessageBox.warning(self, "规则无效", "桥接长度不能超过合并阈值。"); return
            if fade < 0: QMessageBox.warning(self, "规则无效", "提前结束淡出时长不能为负数。"); return
        if current == 2: self.summary.setText(f"视频素材：{self.video_folder.text() or '未选择'}\n背景音乐：{self.music_file.text() or '未选择'}\n事件窗口：前 {self.rule_boxes['pre'].text()} 秒，后 {self.rule_boxes['post'].text()} 秒\n素材不足时提前结束：{'允许' if self.allow_early_end.isChecked() else '不允许'}，淡出 {self.rule_boxes['fade'].text()} 秒")
        self.steps.setCurrentIndex(min(3, current + 1)); self._refresh_navigation()

    def previous(self) -> None:
        self.steps.setCurrentIndex(max(0, self.steps.currentIndex() - 1)); self._refresh_navigation()

    def _refresh_navigation(self) -> None:
        current = self.steps.currentIndex(); self.back.setEnabled(current > 0); self.next.setText("完成" if current == 3 else "下一步 →")


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName(APP_ORGANIZATION); app.setApplicationName(APP_NAME)
    app.setStyleSheet("QWidget { font-family: Segoe UI; font-size: 13px; } QLabel#title { font-size: 26px; font-weight: 700; } QLabel#heading { font-size: 20px; font-weight: 600; } QLabel#subtitle { color: #65717a; } QPushButton { padding: 8px 13px; } QLineEdit, QComboBox { padding: 7px; min-width: 280px; }")
    window = MontageLab(); window.show(); return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
