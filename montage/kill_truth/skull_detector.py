"""OpenCV skull detection restricted to the calibrated personal-kill ROI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .models import DetectedSkull, SkullRowState
from .profile import HudProfile, pixel_rect
from .skull_row import build_skull_row


@dataclass(frozen=True)
class DetectorConfig:
    scale_factors: tuple[float, ...] = (0.95, 1.0, 1.05)
    max_hits_per_template: int = 32
    merge_distance_factor: float = 0.70


@dataclass(frozen=True)
class _Hit:
    detection: DetectedSkull


def _load_images(paths: Iterable[Path]) -> tuple[np.ndarray, ...]:
    images: list[np.ndarray] = []
    for path in paths:
        # ``cv2.imread`` on Windows can still route Unicode paths through an
        # ANSI code page.  ``fromfile`` + ``imdecode`` keeps profile assets
        # usable when the worktree or source names contain Chinese text.
        try:
            encoded = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        except OSError:
            image = None
        if image is not None and image.size:
            images.append(image)
    return tuple(images)


def _orange_ratio(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (((hsv[:, :, 0] <= 18) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 80))
    return float(np.count_nonzero(mask)) / float(mask.size)


def _edge_shape_score(image: np.ndarray, template: np.ndarray) -> float:
    """Measure skull-silhouette agreement independently of brightness.

    The feedback panel also contains square/X-shaped assist and reward
    glyphs.  Their light/dark mass can correlate with a skull at a low
    grayscale template threshold, so correlation is not sufficient evidence.
    Edge overlap preserves the skull silhouette and eye/jaw structure while
    tolerating compression and moderate brightness variation.
    """
    if image.size == 0 or template.size == 0:
        return 0.0
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
    image_gray = cv2.resize(image_gray, (template_gray.shape[1], template_gray.shape[0]), interpolation=cv2.INTER_AREA)
    image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX)
    template_gray = cv2.normalize(template_gray, None, 0, 255, cv2.NORM_MINMAX)
    image_edges = cv2.Canny(image_gray, 50, 120) > 0
    template_edges = cv2.Canny(template_gray, 50, 120) > 0
    image_count = int(np.count_nonzero(image_edges))
    template_count = int(np.count_nonzero(template_edges))
    if not image_count or not template_count:
        return 0.0
    overlap = int(np.count_nonzero(image_edges & template_edges))
    precision = overlap / image_count
    recall = overlap / template_count
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1 = max(lx, rx)
    y1 = max(ly, ry)
    x2 = min(lx + lw, rx + rw)
    y2 = min(ly + lh, ry + rh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


class SkullDetector:
    """Detect structural skull icons, never the whole dynamic panel."""

    def __init__(
        self,
        profile: HudProfile,
        *,
        normal_templates: Sequence[np.ndarray] | None = None,
        headshot_templates: Sequence[np.ndarray] | None = None,
        detector_config: DetectorConfig | None = None,
    ) -> None:
        self.profile = profile
        self.config = detector_config or DetectorConfig()
        if normal_templates is None:
            normal_templates = _load_images(profile.normal_template_paths)
        if headshot_templates is None:
            headshot_templates = _load_images(profile.headshot_template_paths)
        if not normal_templates and profile.template_paths:
            normal_paths = [path for path in profile.template_paths if "headshot" not in path.parent.name.lower()]
            headshot_paths = [path for path in profile.template_paths if "headshot" in path.parent.name.lower()]
            normal_templates = _load_images(normal_paths)
            headshot_templates = _load_images(headshot_paths)
        self.normal_templates = tuple(image for image in normal_templates if image is not None and image.size)
        self.headshot_templates = tuple(image for image in headshot_templates if image is not None and image.size)

    @classmethod
    def from_profile(cls, profile: HudProfile, *, detector_config: DetectorConfig | None = None) -> "SkullDetector":
        return cls(profile, detector_config=detector_config)

    @staticmethod
    def _deduplicate_hits(hits: list[_Hit], merge_distance_factor: float) -> list[DetectedSkull]:
        selected: list[DetectedSkull] = []
        for hit in sorted(hits, key=lambda item: item.detection.template_score, reverse=True):
            item = hit.detection
            duplicate_index: int | None = None
            for index, existing in enumerate(selected):
                distance = float(np.linalg.norm(np.asarray(item.center) - np.asarray(existing.center)))
                threshold = max(item.bbox[2], item.bbox[3], existing.bbox[2], existing.bbox[3]) * merge_distance_factor
                if _iou(item.bbox, existing.bbox) >= 0.25 or distance <= threshold:
                    duplicate_index = index
                    break
            if duplicate_index is None:
                selected.append(item)
            else:
                existing = selected[duplicate_index]
                # A grayscale normal-template match can tie an orange
                # headshot-template match.  Preserve the color-confirmed
                # candidate when its structural score is still comparable.
                if (
                    item.kind.upper() == "HEADSHOT"
                    and item.color_score >= 0.20
                    and item.template_score >= existing.template_score - 0.10
                ):
                    selected[duplicate_index] = item
        return selected

    def _template_hits(
        self,
        frame: np.ndarray,
        bounds: tuple[int, int, int, int],
        template: np.ndarray,
        *,
        kind: str,
        threshold: float,
    ) -> list[_Hit]:
        x1, y1, x2, y2 = bounds
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return []
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
        hits: list[_Hit] = []
        for scale in self.config.scale_factors:
            width = max(4, int(round(template_gray.shape[1] * scale)))
            height = max(4, int(round(template_gray.shape[0] * scale)))
            if width >= region_gray.shape[1] or height >= region_gray.shape[0]:
                continue
            scaled = cv2.resize(template_gray, (width, height), interpolation=cv2.INTER_LINEAR)
            response = cv2.matchTemplate(region_gray, scaled, cv2.TM_CCOEFF_NORMED)
            response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
            local_max = cv2.dilate(response, np.ones((3, 3), dtype=np.uint8))
            ys, xs = np.where((response >= threshold) & (response >= local_max - 1e-6))
            ranked = sorted(zip(xs.tolist(), ys.tolist()), key=lambda item: float(response[item[1], item[0]]), reverse=True)
            for px, py in ranked[: self.config.max_hits_per_template]:
                bx = x1 + int(px)
                by = y1 + int(py)
                crop = frame[by : by + height, bx : bx + width]
                color_score = _orange_ratio(crop)
                shape_score = _edge_shape_score(crop, scaled)
                hits.append(
                    _Hit(
                        DetectedSkull(
                            bbox=(bx, by, width, height),
                            center=(bx + width / 2.0, by + height / 2.0),
                            kind=kind,
                            template_score=float(response[py, px]),
                            color_score=color_score,
                            scale=scale,
                            shape_score=shape_score,
                        )
                    )
                )
        return hits

    def _detect_region(self, frame: np.ndarray, bounds: tuple[int, int, int, int]) -> list[DetectedSkull]:
        hits: list[_Hit] = []
        for template in self.normal_templates:
            hits.extend(self._template_hits(frame, bounds, template, kind="NORMAL", threshold=self.profile.normal_threshold))
        for template in self.headshot_templates:
            hits.extend(self._template_hits(frame, bounds, template, kind="HEADSHOT", threshold=self.profile.headshot_threshold))
        # Filter each raw hit before de-duplication.  A low-quality assist
        # glyph must not suppress a valid skull match at the same location.
        hits = [hit for hit in hits if hit.detection.shape_score >= self.profile.shape_threshold]
        merged = self._deduplicate_hits(hits, self.config.merge_distance_factor)
        classified: list[DetectedSkull] = []
        for item in merged:
            if item.kind.upper() == "HEADSHOT" and item.color_score >= self.profile.orange_color_threshold:
                classified.append(item)
            else:
                classified.append(
                    DetectedSkull(
                        bbox=item.bbox,
                        center=item.center,
                        kind="NORMAL",
                        template_score=item.template_score,
                        color_score=item.color_score,
                        scale=item.scale,
                        shape_score=item.shape_score,
                    )
                )
        return classified

    def _panel_structure_score(
        self,
        frame: np.ndarray,
        row_bounds: tuple[int, int, int, int],
    ) -> float:
        """Score the supporting personal-card neighborhood below the row.

        This score is auxiliary structure only.  It is intentionally based on
        the local card neighborhood and never on generic motion, audio, or
        unrelated HUD changes.
        """
        x1, y1, x2, y2 = row_bounds
        row_width = max(1, x2 - x1)
        row_height = max(1, y2 - y1)
        pad_x = max(45, int(row_width * 2.5))
        context_x1 = max(0, x1 - pad_x)
        context_x2 = min(frame.shape[1], x2 + pad_x)
        context_y1 = max(0, y1 - max(2, int(row_height * 0.25)))
        context_y2 = min(frame.shape[0], y2 + max(72, int(row_height * 2.2)))
        context = frame[context_y1:context_y2, context_x1:context_x2]
        if context.size == 0:
            return 0.0
        hsv = cv2.cvtColor(context, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
        # Victim names in this HUD are saturated red/orange.  Keep the hue
        # band narrow enough to exclude the yellow/brown training-range
        # surfaces that otherwise made a non-panel scene look like a card.
        name_mask = (
            (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168))
             & (hsv[:, :, 1] >= 100)
             & (hsv[:, :, 2] >= 80))
        )
        name_density = float(np.count_nonzero(name_mask)) / float(name_mask.size)
        name_score = min(1.0, name_density / 0.045)
        white_mask = ((gray >= 135) & (hsv[:, :, 1] < 115)).astype(np.uint8) * 255
        horizontal = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(row_width * 0.16)), 1)),
        )
        vertical = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(row_height * 0.10)))),
        )
        outline_density = float(np.count_nonzero(horizontal | vertical)) / float(white_mask.size)
        outline_score = min(1.0, outline_density / 0.018)
        return float(min(1.0, 0.72 * name_score + 0.28 * outline_score))

    def detect(self, frame: np.ndarray, timestamp: float, *, source_id: str = "") -> SkullRowState:
        if frame is None or frame.size == 0:
            return build_skull_row(timestamp, (), source_id=source_id)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        height, width = frame.shape[:2]
        return self.detect_cropped(
            frame,
            timestamp,
            full_width=width,
            full_height=height,
            crop_bounds=(0, 0, width, height),
            source_id=source_id,
        )

    def detect_cropped(
        self,
        frame: np.ndarray,
        timestamp: float,
        *,
        full_width: int,
        full_height: int,
        crop_bounds: tuple[int, int, int, int],
        source_id: str = "",
    ) -> SkullRowState:
        """Detect on an FFmpeg-cropped ROI while retaining full-frame coordinates."""
        if frame is None or frame.size == 0:
            return build_skull_row(timestamp, (), source_id=source_id)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
        row_rois = self.profile.pixel_row_rois(full_width, full_height) or (
            self.profile.pixel_roi(full_width, full_height),
        )
        states: list[SkullRowState] = []
        for row_bounds, normalized in zip(row_rois, self.profile.row_rois or (self.profile.search_roi,)):
            row_x1, row_y1, row_x2, row_y2 = row_bounds
            local_bounds = (
                max(0, row_x1 - crop_x1),
                max(0, row_y1 - crop_y1),
                min(frame.shape[1], row_x2 - crop_x1),
                min(frame.shape[0], row_y2 - crop_y1),
            )
            if local_bounds[2] <= local_bounds[0] or local_bounds[3] <= local_bounds[1]:
                continue
            local_detections = self._detect_region(frame, local_bounds)
            if local_detections:
                detected_x1 = min(item.bbox[0] for item in local_detections)
                detected_y1 = min(item.bbox[1] for item in local_detections)
                detected_x2 = max(item.bbox[0] + item.bbox[2] for item in local_detections)
                detected_y2 = max(item.bbox[1] + item.bbox[3] for item in local_detections)
                detected_row_bounds = (detected_x1, detected_y1, detected_x2, detected_y2)
            else:
                detected_row_bounds = local_bounds
            detections = [
                DetectedSkull(
                    bbox=(item.bbox[0] + crop_x1, item.bbox[1] + crop_y1, item.bbox[2], item.bbox[3]),
                    center=(item.center[0] + crop_x1, item.center[1] + crop_y1),
                    kind=item.kind,
                    template_score=item.template_score,
                    color_score=item.color_score,
                    scale=item.scale,
                    shape_score=item.shape_score,
                )
                for item in local_detections
            ]
            row_y = ((normalized[1] + normalized[3]) / 2.0) * full_height
            structure_score = self._panel_structure_score(
                frame,
                detected_row_bounds,
            )
            state = build_skull_row(
                timestamp,
                detections,
                panel_structure_score=structure_score,
                panel_structure_threshold=self.profile.panel_structure_threshold,
                geometry_threshold=self.profile.geometry_threshold,
                y_tolerance=float(self.profile.row_y_tolerance_px),
                preferred_y=row_y,
                source_id=source_id,
            )
            if state.panel_present:
                states.append(state)
        if not states:
            return build_skull_row(timestamp, (), source_id=source_id)
        return max(states, key=lambda item: (item.confidence, item.skull_count, item.geometry_score))
