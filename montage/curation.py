"""Transparent, source-range editorial exclusions for V2 selection."""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .models import CandidateVariant, PayoffEvent


def excluded_range_reason(
    source: Path, source_start: float, source_end: float, config: PipelineConfig
) -> str | None:
    """Return the configured reason when a source interval overlaps an exclusion."""
    source_name = Path(source).name
    start = float(source_start)
    end = float(source_end)
    for excluded_name, excluded_start, excluded_end, reason in config.v2_excluded_ranges:
        if source_name == excluded_name and end > excluded_start and start < excluded_end:
            return reason
    return None


def event_is_editorially_excluded(event: PayoffEvent, source: Path, config: PipelineConfig) -> bool:
    """Return whether an event timestamp lies in a configured excluded interval."""
    source_name = Path(source).name
    timestamp = float(event.source_time)
    return any(
        source_name == excluded_name and excluded_start <= timestamp < excluded_end
        for excluded_name, excluded_start, excluded_end, _ in config.v2_excluded_ranges
    )


def variant_is_editorially_excluded(variant: CandidateVariant, config: PipelineConfig) -> bool:
    """Reject a generated variant only when one of its authoritative segments overlaps."""
    return any(
        excluded_range_reason(segment.source, segment.source_in, segment.source_out, config) is not None
        for segment in variant.source_segments
    )
