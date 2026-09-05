"""Versioned normalized HUD profiles and V6 cache identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..cache import atomic_write_json

NormalizedRect = tuple[float, float, float, float]


def _rect(value: Iterable[float]) -> NormalizedRect:
    values = tuple(float(item) for item in value)
    if len(values) != 4:
        raise ValueError("normalized ROI must contain x1, y1, x2, y2")
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"invalid normalized ROI: {values}")
    return values


def pixel_rect(rect: NormalizedRect, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _rect(rect)
    return (
        max(0, min(width - 1, int(width * x1))),
        max(0, min(height - 1, int(height * y1))),
        max(1, min(width, int(width * x2))),
        max(1, min(height, int(height * y2))),
    )


@dataclass(frozen=True)
class HudProfile:
    profile_id: str
    width: int
    height: int
    search_roi: NormalizedRect
    row_rois: tuple[NormalizedRect, ...]
    normal_threshold: float
    headshot_threshold: float
    geometry_threshold: float
    panel_structure_threshold: float = 0.45
    shape_threshold: float = 0.36
    orange_color_threshold: float = 0.22
    row_y_tolerance_px: int = 12
    template_paths: tuple[Path, ...] = ()
    normal_template_paths: tuple[Path, ...] = ()
    headshot_template_paths: tuple[Path, ...] = ()
    profile_version: str = "v1"
    detector_version: str = "v6-skull-row-5-calibrated-y-gate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "search_roi", _rect(self.search_roi))
        object.__setattr__(self, "row_rois", tuple(_rect(item) for item in self.row_rois))
        object.__setattr__(self, "template_paths", tuple(Path(item) for item in self.template_paths))
        object.__setattr__(self, "normal_template_paths", tuple(Path(item) for item in self.normal_template_paths))
        object.__setattr__(self, "headshot_template_paths", tuple(Path(item) for item in self.headshot_template_paths))
        if self.width < 1 or self.height < 1:
            raise ValueError("profile resolution must be positive")
        for name in (
            "normal_threshold",
            "headshot_threshold",
            "geometry_threshold",
            "panel_structure_threshold",
            "shape_threshold",
            "orange_color_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")

    def pixel_roi(self, width: int | None = None, height: int | None = None) -> tuple[int, int, int, int]:
        return pixel_rect(self.search_roi, width or self.width, height or self.height)

    def pixel_row_rois(self, width: int | None = None, height: int | None = None) -> tuple[tuple[int, int, int, int], ...]:
        actual_width = width or self.width
        actual_height = height or self.height
        return tuple(pixel_rect(item, actual_width, actual_height) for item in self.row_rois)

    def with_version(self, version: str) -> "HudProfile":
        return replace(self, profile_version=str(version))

    @property
    def all_template_paths(self) -> tuple[Path, ...]:
        paths = self.template_paths + self.normal_template_paths + self.headshot_template_paths
        return tuple(dict.fromkeys(paths))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "detector_version": self.detector_version,
            "width": self.width,
            "height": self.height,
            "search_roi": list(self.search_roi),
            "row_rois": [list(item) for item in self.row_rois],
            "normal_threshold": self.normal_threshold,
            "headshot_threshold": self.headshot_threshold,
            "geometry_threshold": self.geometry_threshold,
            "panel_structure_threshold": self.panel_structure_threshold,
            "shape_threshold": self.shape_threshold,
            "orange_color_threshold": self.orange_color_threshold,
            "row_y_tolerance_px": self.row_y_tolerance_px,
            "template_paths": [str(Path(item).resolve(strict=False)) for item in self.template_paths],
            "normal_template_paths": [str(Path(item).resolve(strict=False)) for item in self.normal_template_paths],
            "headshot_template_paths": [str(Path(item).resolve(strict=False)) for item in self.headshot_template_paths],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HudProfile":
        return cls(
            profile_id=str(data["profile_id"]),
            profile_version=str(data.get("profile_version", "v1")),
            detector_version=str(data.get("detector_version", "v6-skull-row-5-calibrated-y-gate")),
            width=int(data["width"]),
            height=int(data["height"]),
            search_roi=tuple(data["search_roi"]),
            row_rois=tuple(tuple(item) for item in data.get("row_rois", [])),
            normal_threshold=float(data.get("normal_threshold", 0.82)),
            headshot_threshold=float(data.get("headshot_threshold", 0.82)),
            geometry_threshold=float(data.get("geometry_threshold", 0.62)),
            panel_structure_threshold=float(data.get("panel_structure_threshold", 0.45)),
            shape_threshold=float(data.get("shape_threshold", 0.36)),
            orange_color_threshold=float(data.get("orange_color_threshold", 0.22)),
            row_y_tolerance_px=int(data.get("row_y_tolerance_px", 12)),
            template_paths=tuple(Path(item) for item in data.get("template_paths", [])),
            normal_template_paths=tuple(Path(item) for item in data.get("normal_template_paths", [])),
            headshot_template_paths=tuple(Path(item) for item in data.get("headshot_template_paths", [])),
        )

    def save(self, path: Path) -> None:
        atomic_write_json(path, self.to_dict())


def load_profile(path: Path) -> HudProfile:
    return HudProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))


def template_bank_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item).casefold()):
        resolved = path.resolve(strict=False)
        digest.update(str(resolved).encode("utf-8"))
        if resolved.exists() and resolved.is_file():
            stat = resolved.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
            digest.update(resolved.read_bytes())
    return digest.hexdigest()


def v6_cache_key(
    source_fingerprint: Mapping[str, Any],
    profile: HudProfile,
    template_fingerprint: str,
    *,
    threshold_fingerprint: str = "",
    ffmpeg_version: str = "",
    decode_parameters: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "source": dict(source_fingerprint),
        "profile": profile.to_dict(),
        "template_fingerprint": template_fingerprint,
        "threshold_fingerprint": threshold_fingerprint,
        "ffmpeg_version": ffmpeg_version,
        "decode_parameters": dict(decode_parameters or {}),
        "stage": "v6-kill-truth",
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
