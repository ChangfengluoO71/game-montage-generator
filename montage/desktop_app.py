"""Montage Lab local desktop application.

Run with ``python -m montage.desktop_app``.  This native Qt application is
local-first: it stores preferences and custom detector workflows below the
user's application-data directory and never writes into selected RAW folders.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
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
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

APP_ORGANIZATION = "MontageLab"
APP_NAME = "GameMontageGenerator"
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm"}


@dataclass
class DetectionRule:
    name: str
    marker_sample: Path
    positive_sample: Path
    negative_sample: Path | None
    roi: tuple[float, float, float, float]
    threshold: float

    def to_dict(self) -> dict:
        return {"id": re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-") or "rule", "label": self.name, "type": "template_match", "search_roi": list(self.roi), "threshold": self.threshold, "templates": [str(self.marker_sample)], "positive_samples": [str(self.positive_sample)], "negative_samples": [str(self.negative_sample)] if self.negative_sample else []}

@dataclass
class CustomProfile:
    name: str
    rules: list[DetectionRule]

    @property
    def game_id(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-") or "custom-game"

    def workflow(self) -> dict:
        return {"schema": "game-montage-workflow-v1", "game": {"id": self.game_id, "display_name": self.name}, "detectors": {"rules": [rule.to_dict() for rule in self.rules]}, "metadata": {"created_by": "Montage Lab desktop application", "editable": True}}


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
            try: x1, y1, x2, y2 = self._profile().roi
            except ValueError: QMessageBox.warning(self, "ROI 无效", "ROI 必须是数字。"); return False
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                QMessageBox.warning(self, "ROI 无效", "ROI 必须在 0~1，且左上角坐标小于右下角。"); return False
        return True

    def go_next(self) -> None:
        page = self.stack.currentIndex()
        if page < 3 and not self._validate(page): return
        if page == 2:
            profile = self._profile()
            self.review.setText(f"游戏：{profile.name}\n事件：{profile.event_label}\n检测方式：{self.detector.currentText()}\nROI：{profile.roi}\n阈值：{profile.threshold:.2f}\n标志样本：{profile.marker_sample.name}\n完整正例：{profile.positive_sample.name}")
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
        output.write_text(json.dumps(profile.workflow(), ensure_ascii=False, indent=2), encoding="utf-8")
        QMessageBox.information(self, "已保存", f"配置已保存到：\n{output}")
        self.accept()


class MontageLab(QMainWindow):
    def __init__(self) -> None:
        super().__init__(); self.setWindowTitle("Montage Lab — Game Montage Generator"); self.resize(980, 660)
        self.settings = QSettings(APP_ORGANIZATION, APP_NAME)
        self.data_dir = Path(self.settings.fileName()).parent
        self.profiles_dir = self.data_dir / "profiles"
        self._build_ui(); self._restore_preferences()

    def _build_ui(self) -> None:
        root = QWidget(); self.setCentralWidget(root); layout = QVBoxLayout(root)
        header = QHBoxLayout(); header.addWidget(QLabel("Montage Lab", objectName="title")); header.addStretch(); settings = QPushButton("全局设置")
        settings.clicked.connect(self.open_settings); header.addWidget(settings); layout.addLayout(header)
        self.steps = QStackedWidget(); layout.addWidget(self.steps, 1)
        self._build_game_page(); self._build_media_page(); self._build_rules_page(); self._build_ready_page()
        footer = QHBoxLayout(); self.back = QPushButton("← 返回"); self.next = QPushButton("下一步 →"); self.back.clicked.connect(self.previous); self.next.clicked.connect(self.advance); footer.addWidget(self.back); footer.addStretch(); footer.addWidget(self.next); layout.addLayout(footer); self._refresh_navigation()

    def _build_game_page(self) -> None:
        page = QWidget(); layout = QVBoxLayout(page); layout.addWidget(QLabel("步骤 1：选择游戏配置", objectName="heading")); layout.addWidget(QLabel("选择已有配置，或创建一个可复用的新游戏配置。"))
        self.game_buttons = QButtonGroup(self); self.game_buttons.setExclusive(True)
        built_in = QPushButton("Battlefield 6 · Skull Row（内置）"); built_in.setCheckable(True); built_in.setChecked(True); self.game_buttons.addButton(built_in); layout.addWidget(built_in)
        custom = QPushButton("＋ 自创游戏配置"); custom.clicked.connect(self.open_custom_profile); layout.addWidget(custom)
        imported = QPushButton("↥ 导入外部配置 JSON"); imported.clicked.connect(self.import_profile); layout.addWidget(imported); layout.addStretch(); self.steps.addWidget(page)

    def _build_media_page(self) -> None:
        page = QWidget(); form = QFormLayout(page); form.addRow(QLabel("步骤 2：素材与音乐", objectName="heading"))
        self.video_folder = QLineEdit(); self.video_folder.setReadOnly(True); choose_video = QPushButton("选择扫描文件夹"); choose_video.clicked.connect(self.select_video_folder); row = QHBoxLayout(); row.addWidget(self.video_folder); row.addWidget(choose_video); form.addRow("视频素材文件夹", row)
        self.music_file = QLineEdit(); self.music_file.setReadOnly(True); choose_music = QPushButton("选择背景音乐"); choose_music.clicked.connect(self.select_music); row2 = QHBoxLayout(); row2.addWidget(self.music_file); row2.addWidget(choose_music); form.addRow("背景音乐（独立）", row2); form.addRow(QLabel("视频文件夹只会扫描，不会改写 RAW；MP4 音乐会在后续 worker 中提取第一条音轨。")); self.steps.addWidget(page)

    def _build_rules_page(self) -> None:
        page = QWidget(); form = QFormLayout(page); form.addRow(QLabel("步骤 3：剪辑规则", objectName="heading")); self.rule_boxes = {}
        for key, label, value in (("pre", "事件前导（秒）", 1.5), ("post", "事件后保留（秒）", .5), ("merge", "合并阈值（秒）", 2.0), ("bridge", "桥接长度（秒）", 2.0), ("fade", "结尾淡黑（秒）", 5.0)):
            box = QLineEdit(str(value)); self.rule_boxes[key] = box; form.addRow(label, box)
        self.steps.addWidget(page)

    def _build_ready_page(self) -> None:
        page = QWidget(); layout = QVBoxLayout(page); layout.addWidget(QLabel("步骤 4：准备生成", objectName="heading")); self.summary = QLabel(); self.summary.setWordWrap(True); layout.addWidget(self.summary); generate = QPushButton("开始生成 AI 初剪"); generate.clicked.connect(lambda: QMessageBox.information(self, "本地 Worker", "Worker 接口将在下一实现切片接入；当前项目配置已完成。")); layout.addWidget(generate); layout.addStretch(); self.steps.addWidget(page)

    def open_custom_profile(self) -> None:
        wizard = ProfileWizard(self.profiles_dir, self)
        if wizard.exec():
            self.settings.setValue("last_custom_profile", wizard.name.text().strip())

    def import_profile(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "导入工作流配置", "", "Workflow JSON (*.json)")
        if selected: QMessageBox.information(self, "已导入", f"已选择配置：\n{selected}")

    def select_video_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择视频素材文件夹")
        if selected:
            count = sum(1 for path in Path(selected).rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)
            self.video_folder.setText(f"{selected}  ({count} 个视频)"); self.settings.setValue("video_folder", selected)

    def select_music(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "选择背景音乐", "", "Audio/Video (*.mp3 *.wav *.m4a *.aac *.mp4 *.mkv)")
        if selected: self.music_file.setText(selected)

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
        if folder: self.video_folder.setText(str(folder))

    def advance(self) -> None:
        current = self.steps.currentIndex()
        if current == 1 and not self.video_folder.text(): QMessageBox.warning(self, "缺少素材", "请选择视频素材文件夹。"); return
        if current == 2:
            try: bridge = float(self.rule_boxes["bridge"].text()); merge = float(self.rule_boxes["merge"].text())
            except ValueError: QMessageBox.warning(self, "规则无效", "剪辑规则必须是数字。"); return
            if bridge > merge: QMessageBox.warning(self, "规则无效", "桥接长度不能超过合并阈值。"); return
        if current == 2: self.summary.setText(f"视频素材：{self.video_folder.text() or '未选择'}\n背景音乐：{self.music_file.text() or '未选择'}\n事件窗口：前 {self.rule_boxes['pre'].text()} 秒，后 {self.rule_boxes['post'].text()} 秒")
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
