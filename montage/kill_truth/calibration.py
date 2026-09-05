"""Calibration-set preparation and honest V6 detector reporting."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from ..cache import atomic_write_json
from ..config import PipelineConfig
from ..models import MediaRecord
from ..toolchain import Toolchain
from .models import SkullRowState
from .profile import HudProfile
from .skull_detector import SkullDetector


@dataclass(frozen=True)
class CalibrationSample:
    source_glob: str
    timestamp: float
    expected_panel: bool
    expected_count: int = 0
    expected_headshot_count: int = 0
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_glob": self.source_glob,
            "timestamp": self.timestamp,
            "expected_panel": self.expected_panel,
            "expected_count": self.expected_count,
            "expected_headshot_count": self.expected_headshot_count,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationSample":
        return cls(
            source_glob=str(data["source_glob"]),
            timestamp=float(data["timestamp"]),
            expected_panel=bool(data.get("expected_panel", False)),
            expected_count=int(data.get("expected_count", 0)),
            expected_headshot_count=int(data.get("expected_headshot_count", 0)),
            label=str(data.get("label", "")),
        )


def default_calibration_samples() -> list[CalibrationSample]:
    """Return manually inspected direct-RAW calibration labels.

    These labels are frame-level calibration anchors, not V5-derived events.
    They intentionally include panel-free objective/reload frames and both
    normal and orange skull examples.
    """
    return [
        CalibrationSample("*2026-09-01 21-14-19.mp4", 2.5, False, label="reward text without skull row"),
        CalibrationSample("*2026-09-01 21-14-19.mp4", 3.5, True, 2, 0, "double-kill skull row"),
        CalibrationSample("*2026-09-01 21-14-19.mp4", 4.0, True, 2, 0, "double-kill persistent state"),
        CalibrationSample("*2026-09-01 21-14-19.mp4", 29.0, True, 1, 0, "single personal kill card"),
        CalibrationSample("*2026-09-01 21-14-19.mp4", 50.0, True, 1, 0, "single personal kill card"),
        CalibrationSample("*2026-09-01 21-14-19.mp4", 73.0, True, 1, 1, "orange headshot skull"),
        CalibrationSample("*2026-08-30 11-59-30.mp4", 5.0, False, label="kill-feed only / panel not yet visible"),
        CalibrationSample("*2026-08-30 11-59-30.mp4", 5.5, False, label="assist/reward X glyph without skull"),
        CalibrationSample("*2026-07-05 22-53-34.mp4", 2.0, False, label="red X reward glyph without skull"),
        CalibrationSample("*2026-07-05 22-53-34.mp4", 12.0, True, 1, 0, "single skull plus assist glyph"),
    ]


def calibration_manifest_path(config: PipelineConfig) -> Path:
    return config.v6_kill_truth_dir / "calibration" / "calibration_manifest.json"


def ensure_calibration_manifest(config: PipelineConfig) -> list[CalibrationSample]:
    path = calibration_manifest_path(config)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [CalibrationSample.from_dict(item) for item in payload.get("samples", [])]
    samples = default_calibration_samples()
    atomic_write_json(
        path,
        {
            "schema": "v6-calibration-manifest-v1",
            "source_of_labels": "direct_raw_manual_inspection",
            "old_v5_events_used_as_truth": False,
            "samples": [sample.to_dict() for sample in samples],
        },
    )
    return samples


def _find_source(raw_dir: Path, source_glob: str) -> Path | None:
    matches = sorted(raw_dir.glob(source_glob))
    return next((item for item in matches if item.is_file()), None)


def _extract_frame(source: Path, timestamp: float, toolchain: Toolchain) -> np.ndarray:
    command = [
        str(toolchain.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timestamp):.6f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    result = subprocess.run(command, shell=False, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"calibration frame decode failed: {detail}")
    frame = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"calibration frame is not a valid image: {source} @ {timestamp}")
    return frame


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"unable to encode calibration image: {path}")
    encoded.tofile(str(path))


def _extract_template(source: Path, timestamp: float, crop: tuple[int, int, int, int], output: Path, toolchain: Toolchain) -> None:
    x, y, width, height = crop
    command = [
        str(toolchain.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timestamp):.6f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        f"crop={width}:{height}:{x}:{y}",
        "-y",
        str(output),
    ]
    result = subprocess.run(command, shell=False, capture_output=True, check=False)
    if result.returncode != 0 or not output.exists():
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"template extraction failed: {detail}")


def default_profile(config: PipelineConfig, toolchain: Toolchain) -> HudProfile:
    """Create/update the calibrated 1920x1200 profile from direct RAW frames."""
    profile_root = config.v6_profiles_dir / config.v6_profile_id
    normal_dir = profile_root / "templates" / "normal_skull"
    headshot_dir = profile_root / "templates" / "headshot_skull"
    normal_dir.mkdir(parents=True, exist_ok=True)
    headshot_dir.mkdir(parents=True, exist_ok=True)
    source_0901 = _find_source(config.raw_dir, "*2026-09-01 21-14-19.mp4")
    source_0705 = _find_source(config.raw_dir, "*2026-07-05 22-53-34.mp4")
    normal_jobs = [
        (source_0901, 3.5, (678, 746, 34, 38), normal_dir / "normal_01.png"),
        (source_0901, 4.0, (678, 746, 34, 38), normal_dir / "normal_02.png"),
        (source_0901, 29.0, (710, 746, 34, 38), normal_dir / "normal_03.png"),
        (source_0901, 50.0, (710, 746, 34, 38), normal_dir / "normal_04.png"),
    ]
    for source, timestamp, crop, output in normal_jobs:
        if source and not output.exists():
            _extract_template(source, timestamp, crop, output, toolchain)
    # The 07-05 frame contains a red headshot/reward glyph variation rather
    # than the orange skull shape.  It is retained as a hard visual variant
    # in calibration samples, but must not contaminate the skull template
    # bank.  Only genuine skull-shaped headshot examples enter this bank.
    headshot_jobs = [
        (source_0901, 73.0, (714, 746, 34, 38), headshot_dir / "headshot_01.png"),
    ]
    for source, timestamp, crop, output in headshot_jobs:
        if source and not output.exists():
            _extract_template(source, timestamp, crop, output, toolchain)
    normal_paths = tuple(sorted(normal_dir.glob("*.png")))
    headshot_paths = tuple(
        sorted(path for path in headshot_dir.glob("*.png") if path.name != "headshot_02.png")
    )
    profile = HudProfile(
        profile_id=config.v6_profile_id,
        profile_version="v1",
        detector_version=config.v6_detector_version,
        width=1920,
        height=1200,
        search_roi=(0.30, 0.55, 0.48, 0.74),
        # Use the lower personal-card skull row as the truth-bearing row.
        # The upper ``双杀`` banner also contains decorative skulls, but it is
        # not the stable per-player feedback row and created avoidable false
        # count transitions on other resolutions.
        # Direct-RAW calibration places this row at roughly y=0.62--0.65
        # for the 1920x1200 HUD. Keep the vertical gate tight: the training
        # range contains skull-like light shapes below the actual card, and
        # allowing the whole rough search ROI to become a row turns those
        # shapes into false panel states.
        row_rois=((0.30, 0.60, 0.40, 0.66),),
        normal_threshold=0.65,
        headshot_threshold=0.64,
        geometry_threshold=0.65,
        panel_structure_threshold=0.45,
        shape_threshold=0.36,
        orange_color_threshold=0.12,
        row_y_tolerance_px=12,
        normal_template_paths=normal_paths,
        headshot_template_paths=headshot_paths,
    )
    profile.save(profile_root / "profile.json")
    return profile


def _sample_result(sample: CalibrationSample, state: SkullRowState, source: Path) -> dict[str, Any]:
    panel_ok = state.panel_present == sample.expected_panel
    count_ok = not sample.expected_panel or state.skull_count == sample.expected_count
    headshot_ok = not sample.expected_panel or state.headshot_count == sample.expected_headshot_count
    return {
        **sample.to_dict(),
        "source_path": str(source.resolve(strict=False)),
        "detected": state.to_dict(),
        "panel_ok": panel_ok,
        "count_ok": count_ok,
        "headshot_ok": headshot_ok,
        "sample_ok": panel_ok and count_ok and headshot_ok,
    }


def _frame_calibration_metrics(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute frame-level panel metrics with correct negative semantics.

    ``panel_ok`` is a classification result: it is true when the detector's
    panel-presence decision matches the manual label.  For a negative sample,
    ``panel_ok=True`` is therefore a correct negative, not a false positive.
    Keep these metrics explicitly separate from semantic OwnKillEvent metrics.
    """
    items = list(results)
    positive = [item for item in items if item.get("expected_panel")]
    negative = [item for item in items if not item.get("expected_panel")]
    positive_panel_hits = sum(bool(item.get("panel_ok")) for item in positive)
    positive_full_hits = sum(bool(item.get("panel_ok") and item.get("count_ok") and item.get("headshot_ok")) for item in positive)
    negative_correct = sum(bool(item.get("panel_ok")) for item in negative)
    negative_false_positives = sum(not bool(item.get("panel_ok")) for item in negative)
    return {
        "positive_panel_samples": len(positive),
        "positive_panel_hits": positive_panel_hits,
        "positive_full_hits": positive_full_hits,
        "negative_panel_samples": len(negative),
        "negative_panel_correct": negative_correct,
        "negative_panel_false_positives": negative_false_positives,
        "frame_panel_precision": (
            positive_panel_hits / (positive_panel_hits + negative_false_positives)
            if positive_panel_hits + negative_false_positives
            else 0.0
        ),
        "frame_panel_recall": positive_panel_hits / len(positive) if positive else 0.0,
    }


def run_calibration(
    config: PipelineConfig,
    toolchain: Toolchain,
    records: Iterable[MediaRecord],
    profile: HudProfile,
) -> dict[str, Any]:
    samples = ensure_calibration_manifest(config)
    detector = SkullDetector.from_profile(profile)
    frame_dir = config.v6_kill_truth_dir / "calibration" / "frames"
    results: list[dict[str, Any]] = []
    source_by_path = {record.file_path.resolve(strict=False): record for record in records}
    del source_by_path
    for index, sample in enumerate(samples, start=1):
        source = _find_source(config.raw_dir, sample.source_glob)
        if source is None:
            results.append({**sample.to_dict(), "sample_ok": False, "error": "source_not_found"})
            continue
        try:
            frame = _extract_frame(source, sample.timestamp, toolchain)
            output_frame = frame_dir / f"sample_{index:02d}_{source.stem}_{sample.timestamp:.3f}.jpg"
            _write_image(output_frame, frame)
            state = detector.detect(frame, sample.timestamp, source_id=source.stem)
            results.append({**_sample_result(sample, state, source), "frame_path": str(output_frame.resolve(strict=False))})
        except Exception as exc:
            results.append({**sample.to_dict(), "source_path": str(source), "sample_ok": False, "error": str(exc)})
    frame_metrics = _frame_calibration_metrics(results)
    failures = [item for item in results if not item.get("sample_ok", False)]
    report = {
        "schema": "v6-calibration-result-v1",
        "profile_id": profile.profile_id,
        "resolution": [profile.width, profile.height],
        "search_roi": list(profile.search_roi),
        "row_rois": [list(item) for item in profile.row_rois],
        "templates": {
            "normal_count": len(profile.normal_template_paths),
            "headshot_count": len(profile.headshot_template_paths),
            "headshot_bank_complete": len(profile.headshot_template_paths) >= 3,
        },
        "thresholds": {
            "normal": profile.normal_threshold,
            "headshot": profile.headshot_threshold,
            "geometry": profile.geometry_threshold,
            "shape": profile.shape_threshold,
            "orange_color": profile.orange_color_threshold,
        },
        "coarse_scan_fps": config.v6_coarse_fps,
        "dense_scan_fps": config.v6_dense_fps,
        "sample_count": len(results),
        **frame_metrics,
        # Retain the old key as a compatibility alias, but give it the
        # corrected false-positive meaning rather than the old inverted one.
        "negative_panel_hits": frame_metrics["negative_panel_false_positives"],
        "failures": failures,
        "results": results,
        "semantic_event_precision": None,
        "semantic_event_recall": None,
        "semantic_event_metrics_status": "NOT_ESTABLISHED_CALIBRATION_FRAME_LABELS_ONLY",
        "calibration_gate": "PASS" if not failures else "FAIL",
        "full_scan_authorized": not failures,
    }
    atomic_write_json(config.v6_kill_truth_dir / "calibration" / "calibration_results.json", report)
    return report


def render_calibration_report(config: PipelineConfig, profile: HudProfile, result: Mapping[str, Any], toolchain: Toolchain) -> Path:
    path = config.v6_calibration_report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Battlefield V6 Calibration Report",
        "",
        f"- Calibration gate: **{result.get('calibration_gate')}**",
        f"- Full scan authorized by detector calibration: **{result.get('full_scan_authorized')}**",
        f"- Resolution: `{profile.width}x{profile.height}`",
        f"- HUD profile: `{profile.profile_id}`",
        f"- Search ROI (normalized): `{list(profile.search_roi)}`",
        f"- Row ROIs (normalized): `{[list(item) for item in profile.row_rois]}`",
        f"- Normal template count: `{len(profile.normal_template_paths)}`",
        f"- Headshot template count: `{len(profile.headshot_template_paths)}`; complete bank: `{result['templates']['headshot_bank_complete']}`",
        f"- Thresholds: normal `{profile.normal_threshold}`, headshot `{profile.headshot_threshold}`, geometry `{profile.geometry_threshold}`, shape `{profile.shape_threshold}`, orange `{profile.orange_color_threshold}`",
        f"- Coarse/dense FPS: `{config.v6_coarse_fps}` / `{config.v6_dense_fps}`",
        f"- Samples: `{result.get('sample_count')}`; positive panel hits `{result.get('positive_panel_hits')}/{result.get('positive_panel_samples')}`; negative panel false positives `{result.get('negative_panel_false_positives')}`; correct negatives `{result.get('negative_panel_correct')}/{result.get('negative_panel_samples')}`",
        f"- Frame-panel precision/recall: `{result.get('frame_panel_precision'):.3f}` / `{result.get('frame_panel_recall'):.3f}`",
        "",
        "## Toolchain",
        "",
        f"- FFmpeg: `{toolchain.ffmpeg.resolve(strict=False)}` ({toolchain.ffmpeg_version})",
        f"- ffprobe: `{toolchain.ffprobe.resolve(strict=False)}` ({toolchain.ffprobe_version})",
        f"- NVENC h264 runtime: `{toolchain.nvenc_h264}`",
        "",
        "## 语义质量边界",
        "",
        "本报告的样本标签是人工检查的 UI frame labels，只验证 panel/skull detector 的校准，不等同于 OwnKillEvent 的 precision/recall。完整语义指标必须在 10–15 分钟 gold set 上按 ±500ms 匹配后计算；当前没有把 frame 命中数字伪装成 semantic PASS。",
        "",
        "## Known limitations",
        "",
        "- headshot template bank 少于 3 个实例时标记为 incomplete；headshot 类型仍可为 UNKNOWN，不会强行分类。",
        "- 当前 profile 只校准了 1920x1200；其他分辨率需要独立 profile 校准后才能作为全量 truth 输入。",
        "- detection frame、confirmation time 和 impact time 是三个不同概念；本阶段尚未进行 impact time 估计或音乐层处理。",
        "",
        "## STOP",
        "",
        "V6 在 OwnKillEvent/rapid multi-kill 事实经过 review 和 gold-set 验证前，不进入 Montage、Music、Beat Sync 或 Render。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
