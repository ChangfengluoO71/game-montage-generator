"""Local HTML review artifacts for V6 truth, not montage review."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np

from ..cache import atomic_write_json
from ..models import MediaRecord
from ..toolchain import Toolchain
from .calibration import _extract_frame, _write_image
from .models import OwnKillEvent
from .profile import HudProfile
from .scanner import V6ScanResult


def _review_event_dir(root: Path, event: OwnKillEvent) -> Path:
    safe_source = "".join(character if character.isalnum() or character in "-_" else "_" for character in event.source_id)
    return root / safe_source / event.event_id.replace("/", "_")


def _crop_panel(frame: np.ndarray, profile: HudProfile) -> np.ndarray:
    x1, y1, x2, y2 = profile.pixel_roi(frame.shape[1], frame.shape[0])
    return frame[y1:y2, x1:x2]


def render_kill_review(
    results: Iterable[V6ScanResult],
    profile_by_id: Mapping[str, HudProfile],
    output_dir: Path,
    toolchain: Toolchain,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for result in results:
        profile = profile_by_id.get(result.hud_profile_id)
        if profile is None:
            continue
        for event in result.events:
            event_dir = _review_event_dir(output_dir, event)
            event_dir.mkdir(parents=True, exist_ok=True)
            images: dict[str, str] = {}
            for label, offset in (("before", -0.15), ("transition", 0.0), ("after", 0.15)):
                timestamp = max(0.0, event.confirmation_time + offset)
                try:
                    frame = _extract_frame(event.source_path, timestamp, toolchain)
                    full_path = event_dir / f"{label}_full.jpg"
                    panel_path = event_dir / f"{label}_panel.jpg"
                    _write_image(full_path, frame)
                    _write_image(panel_path, _crop_panel(frame, profile))
                    images[f"{label}_full"] = full_path.relative_to(output_dir).as_posix()
                    images[f"{label}_panel"] = panel_path.relative_to(output_dir).as_posix()
                except Exception as exc:
                    images[f"{label}_error"] = str(exc)
            entries.append(
                {
                    "event": event.to_dict(),
                    "sequence_id": event.sequence_id,
                    "images": images,
                    "source_name": event.source_path.name,
                }
            )
    atomic_write_json(output_dir / "review_manifest.json", {"schema": "v6-kill-review-v1", "events": entries})
    html_path = output_dir / "kill_review.html"
    sections: list[str] = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<title>Battlefield V6 Kill Truth Review</title>",
        "<style>body{font-family:Segoe UI,Arial;background:#111;color:#eee}section{border:1px solid #444;margin:16px;padding:12px} .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px} img{max-width:100%;background:#222}button{margin:4px;padding:6px 10px} .meta{font-family:Consolas,monospace;white-space:pre-wrap}</style>",
        "</head><body><h1>V6 OwnKillEvent Review</h1>",
        "<p>该页面只审阅 Personal Kill Panel / Skull Row truth；CONFIRM、FALSE_POSITIVE、UNKNOWN 不会修改 RAW。</p>",
    ]
    for index, entry in enumerate(entries, start=1):
        event = entry["event"]
        assert isinstance(event, dict)
        images = entry["images"]
        assert isinstance(images, dict)
        sections.append("<section>")
        sections.append(f"<h2>{index}. {html.escape(str(entry['source_name']))} — {html.escape(str(event.get('event_id')))}</h2>")
        sections.append("<div class='meta'>" + html.escape(json_like(event)) + "</div>")
        sections.append("<div class='grid'>")
        for label in ("before", "transition", "after"):
            full = images.get(f"{label}_full")
            panel = images.get(f"{label}_panel")
            if full:
                sections.append(f"<figure><figcaption>{label} full frame</figcaption><img src='{html.escape(str(full))}'></figure>")
            if panel:
                sections.append(f"<figure><figcaption>{label} panel ROI</figcaption><img src='{html.escape(str(panel))}'></figure>")
        sections.append("</div>")
        sections.append(
            f"<button onclick=\"mark(this,'CONFIRM')\">CONFIRM</button><button onclick=\"mark(this,'FALSE_POSITIVE')\">FALSE_POSITIVE</button><button onclick=\"mark(this,'UNKNOWN')\">UNKNOWN</button><input class='notes' placeholder='notes'><span class='decision'></span>"
        )
        sections.append("</section>")
    sections.append("<button id='export' onclick='exportAnnotations()'>DOWNLOAD annotations.json</button><script>const annotations={};function mark(button,value){const section=button.parentElement;const eventId=section.querySelector('.meta').textContent.match(/\\\"event_id\\\": \\\"([^\\\"]+)/)?.[1]||'';const item={event_id:eventId,review_status:value,reviewer:'cfl',notes:section.querySelector('.notes').value,reviewed_at:new Date().toISOString()};annotations[eventId]=item;section.querySelector('.decision').textContent='  '+value;section.querySelector('.decision').dataset.value=value;}function exportAnnotations(){const blob=new Blob([JSON.stringify({schema:'v6-review-annotations-v1',annotations:Object.values(annotations)},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='annotations.json';a.click();URL.revokeObjectURL(a.href);}</script></body></html>")
    html_path.write_text("\n".join(sections), encoding="utf-8")
    return html_path


def json_like(value: Mapping[str, object]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
