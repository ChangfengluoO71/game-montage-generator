"""Cache fingerprints and atomic generated-artifact writers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "absolute_path": str(resolved),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def cache_key(
    fingerprint: dict[str, Any],
    stage: str,
    parameters: dict[str, Any],
) -> str:
    payload = {"fingerprint": fingerprint, "stage": stage, "parameters": parameters}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def v2_cache_key(
    source_fingerprint: dict[str, object],
    stage: str,
    parameters: dict[str, object],
) -> str:
    ffmpeg_version = parameters.get("ffmpeg_version")
    if not isinstance(ffmpeg_version, str) or not ffmpeg_version.strip() or ffmpeg_version.strip().lower() == "unknown":
        raise ValueError("recorded runtime-tested FFmpeg version is required for V2 cache keys")
    payload = {
        "source_fingerprint": source_fingerprint,
        "stage": stage,
        "stage_version": parameters.get("stage_version", "v2"),
        "ffmpeg_version": ffmpeg_version,
        "parameters": parameters,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def v2_variant_fingerprint_cache_key(
    source_fingerprint: dict[str, object],
    *,
    variant_id: str,
    parent_candidate_id: str,
    ranges: list[tuple[float, float]],
    interval: float,
    detector_version: str,
    ffmpeg_version: str,
    ffmpeg_path: str,
) -> str:
    """Return the stable identity for one V2 variant fingerprint calculation."""
    return v2_cache_key(
        source_fingerprint,
        "variant_fingerprint_v2",
        {
            "stage_version": "v2-dedupe-fingerprint-1",
            "ffmpeg_version": ffmpeg_version,
            "ffmpeg_path": ffmpeg_path,
            "variant_id": variant_id,
            "parent_candidate_id": parent_candidate_id,
            "ranges": [[float(start), float(end)] for start, end in ranges],
            "interval": float(interval),
            "detector_version": detector_version,
        },
    )


def baseline_manifest(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(resolved), "size": stat.st_size, "mtime": stat.st_mtime, "sha256": digest.hexdigest()}


def assert_baseline_unchanged(before: dict[str, object], path: Path) -> None:
    try:
        after = baseline_manifest(path)
    except (OSError, FileNotFoundError) as exc:
        raise RuntimeError(f"baseline is missing or unreadable: {path}") from exc
    if after != before:
        raise RuntimeError(f"baseline changed: expected {before}, got {after}")


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    atomic_write_bytes(path, content)


def read_cached_json(path: Path, expected_key: str) -> Any | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("cache_key") != expected_key:
        return None
    return value.get("data")


def write_cached_json(path: Path, cache_key_value: str, data: Any) -> None:
    atomic_write_json(path, {"cache_key": cache_key_value, "data": data})
