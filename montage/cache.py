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
    if value.get("cache_key") != expected_key:
        return None
    return value.get("data")


def write_cached_json(path: Path, cache_key_value: str, data: Any) -> None:
    atomic_write_json(path, {"cache_key": cache_key_value, "data": data})
