"""Static local review assets for candidate inspection."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Sequence

from .cache import atomic_write_bytes
from .config import PipelineConfig, assert_source_read_only
from .models import Candidate
from .toolchain import Toolchain, run_command


def render_review_assets(candidates: Sequence[Candidate], config: PipelineConfig, toolchain: Toolchain) -> Path:
    review_dir = config.review_dir
    review_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for candidate in candidates:
        item_dir = review_dir / candidate.candidate_id
        item_dir.mkdir(parents=True, exist_ok=True)
        thumbnail = item_dir / "thumbnail.jpg"
        preview = item_dir / "preview.mp4"
        item: dict[str, object] = {
            "candidate_id": candidate.candidate_id,
            "source_file": str(candidate.source_file),
            "source_start": candidate.source_start,
            "source_end": candidate.source_end,
            "duration": candidate.duration,
            "score": candidate.final_score,
            "thumbnail": str(thumbnail.relative_to(review_dir)),
            "preview": str(preview.relative_to(review_dir)),
        }
        try:
            assert_source_read_only(candidate.source_file, config.raw_dir)
            midpoint = candidate.source_start + min(candidate.duration / 2.0, 4.0)
            run_command(
                [
                    toolchain.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{midpoint:.3f}",
                    "-i",
                    candidate.source_file,
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=480:-2:flags=fast_bilinear",
                    "-q:v",
                    "3",
                    thumbnail,
                ],
                check=True,
            )
            run_command(
                [
                    toolchain.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{candidate.source_start:.3f}",
                    "-i",
                    candidate.source_file,
                    "-t",
                    f"{min(candidate.duration, 8.0):.3f}",
                    "-vf",
                    "scale=640:-2:flags=fast_bilinear",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "30",
                    preview,
                ],
                check=True,
            )
            item["status"] = "ok"
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)
        manifest.append(item)
    manifest_path = review_dir / "review_manifest.json"
    atomic_write_bytes(manifest_path, json.dumps({"candidates": manifest}, ensure_ascii=False, indent=2).encode("utf-8"))
    cards: list[str] = []
    for item in manifest:
        cards.append(
            "<article class='card'>"
            f"<img src='{html.escape(str(item['thumbnail']))}' alt='thumbnail'>"
            f"<h2>{html.escape(str(item['candidate_id']))}</h2>"
            f"<p>{html.escape(Path(str(item['source_file'])).name)}<br>"
            f"{float(item['source_start']):.3f}s - {float(item['source_end']):.3f}s | "
            f"{float(item['duration']):.2f}s | score {float(item['score']):.3f}</p>"
            f"<video controls preload='metadata' src='{html.escape(str(item['preview']))}'></video>"
            f"<button data-id='{html.escape(str(item['candidate_id']))}' data-value='KEEP'>KEEP</button>"
            f"<button data-id='{html.escape(str(item['candidate_id']))}' data-value='DROP'>DROP</button>"
            f"<button data-id='{html.escape(str(item['candidate_id']))}' data-value='STAR'>STAR</button>"
            "</article>"
        )
    document = """<!doctype html><html><head><meta charset='utf-8'><title>Battlefield Review</title><style>body{font-family:system-ui;background:#111;color:#eee;margin:2rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}.card{background:#222;padding:1rem;border-radius:8px}.card img,.card video{max-width:100%;display:block;margin:.5rem 0}button{margin:.2rem;padding:.4rem .7rem}</style></head><body><h1>Battlefield candidate review</h1><div class='grid'>""" + "".join(cards) + """</div><script>for(const button of document.querySelectorAll('button')){button.onclick=()=>{const key='battlefield-review';const state=JSON.parse(localStorage.getItem(key)||'{}');state[button.dataset.id]=button.dataset.value;localStorage.setItem(key,JSON.stringify(state));button.parentElement.dataset.state=button.dataset.value;}};</script></body></html>"""
    html_path = review_dir / "review.html"
    atomic_write_bytes(html_path, document.encode("utf-8"))
    return html_path
