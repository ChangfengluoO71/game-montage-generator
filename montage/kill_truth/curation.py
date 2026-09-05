"""Persistent editorial curation ledger, separate from kill truth."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

VALID_STATUSES = {"UNREVIEWED", "AUTO_CONFIRMED", "MANUAL_KEEP", "MANUAL_REJECT", "MANUAL_UNKNOWN"}


def init_curation_ledger(sources: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for source in sources:
        source_id = str(source["source_id"])
        entries[source_id] = {
            "source_id": source_id,
            "source_path": str(source.get("source_path", "")),
            "status": "UNREVIEWED",
            "editorial_excluded": False,
            "notes": "",
            "updated_at": None,
        }
    return {
        "schema": "curation-ledger-v1",
        "semantic_truth_independent": True,
        "sources": entries,
    }


def set_curation_status(ledger: Mapping[str, Any], source_id: str, status: str, *, notes: str = "") -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid curation status: {status}")
    payload = {
        "schema": ledger.get("schema", "curation-ledger-v1"),
        "semantic_truth_independent": True,
        "sources": {key: dict(value) for key, value in (ledger.get("sources") or {}).items()},
    }
    if source_id not in payload["sources"]:
        raise KeyError(source_id)
    entry = payload["sources"][source_id]
    entry["status"] = status
    entry["editorial_excluded"] = status == "MANUAL_REJECT"
    entry["notes"] = notes
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    return payload
