"""Visual fingerprint comparison and duplicate group preference."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from math import ceil
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from .cache import file_fingerprint, read_cached_json, v2_variant_fingerprint_cache_key, write_cached_json
from .config import PipelineConfig, assert_source_read_only, is_within
from .models import Candidate, CandidateVariant, DedupeResult, V2DedupeResult
from .toolchain import Toolchain


def _dhash(frame: np.ndarray) -> int:
    image = frame.reshape(32, 32).astype(np.float32)
    reduced = image.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    comparisons = reduced[:, :-1] > reduced[:, 1:]
    value = 0
    for bit in comparisons.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def fingerprint_candidate(candidate: Candidate, source_for_analysis: Path, toolchain: Toolchain) -> list[int]:
    interval = 2.0
    sample_count = max(2, min(16, int(candidate.duration / interval) + 1))
    result = subprocess.run(
        [
            str(toolchain.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{candidate.source_start:.3f}",
            "-i",
            str(source_for_analysis),
            "-t",
            f"{candidate.duration:.3f}",
            "-vf",
            f"fps={1.0 / interval},scale=32:32:flags=fast_bilinear,format=gray",
            "-frames:v",
            str(sample_count),
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        shell=False,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    data = result.stdout or b""
    frame_size = 32 * 32
    return [_dhash(np.frombuffer(data[offset : offset + frame_size], dtype=np.uint8)) for offset in range(0, len(data) - frame_size + 1, frame_size)]


def _expected_fingerprint_count(variant: CandidateVariant, interval: float) -> int:
    return sum(_fingerprint_sample_count(segment.duration, interval) for segment in variant.source_segments)


def _fingerprint_sample_count(duration: float, interval: float) -> int:
    if duration <= 0:
        return 0
    return max(1, min(16, ceil(duration / interval)))


def _is_complete_fingerprint(value: object, expected_count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == expected_count
        and all(isinstance(item, int) and not isinstance(item, bool) and 0 <= item < (1 << 56) for item in value)
    )


def _fingerprint_segment(
    source_for_analysis: Path, start: float, duration: float, interval: float, toolchain: Toolchain
) -> list[int]:
    sample_count = _fingerprint_sample_count(duration, interval)
    result = subprocess.run(
        [
            str(toolchain.ffmpeg), "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}",
            "-i", str(source_for_analysis), "-t", f"{duration:.3f}",
            "-vf", f"fps={1.0 / interval},scale=32:32:flags=fast_bilinear,format=gray",
            "-frames:v", str(sample_count), "-f", "rawvideo", "pipe:1",
        ],
        shell=False,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        details = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg fingerprinting failed: {details or 'unknown error'}")
    data = result.stdout or b""
    frame_size = 32 * 32
    expected_bytes = sample_count * frame_size
    if len(data) != expected_bytes:
        actual_frames = len(data) / frame_size
        raise RuntimeError(f"FFmpeg fingerprint output expected {sample_count} frames, got {actual_frames:g}")
    return [_dhash(np.frombuffer(data[offset : offset + frame_size], dtype=np.uint8)) for offset in range(0, len(data) - frame_size + 1, frame_size)]


def fingerprint_variant(
    variant: CandidateVariant, source_for_analysis: Path, toolchain: Toolchain, config: PipelineConfig
) -> list[int]:
    """Fingerprint every authoritative source segment, caching against the selected FFmpeg identity."""
    assert_source_read_only(source_for_analysis, config.raw_dir)
    if source_for_analysis.resolve(strict=True) != variant.source_file.resolve(strict=True):
        raise ValueError("fingerprint analysis source must match variant.source_file")
    interval = config.fingerprint_interval
    if interval <= 0:
        raise ValueError("fingerprint_interval must be positive")
    expected_count = _expected_fingerprint_count(variant, interval)
    source_fingerprint = file_fingerprint(source_for_analysis)
    key = v2_variant_fingerprint_cache_key(
        source_fingerprint,
        variant_id=variant.variant_id,
        parent_candidate_id=variant.parent_candidate_id,
        ranges=[(segment.source_in, segment.source_out) for segment in variant.source_segments],
        interval=interval,
        detector_version=config.payoff_detector_version,
        ffmpeg_version=toolchain.ffmpeg_version,
        ffmpeg_path=str(toolchain.ffmpeg.resolve(strict=False)),
    )
    cache_root = config.cache_dir.resolve(strict=False)
    if not is_within(cache_root, config.work_dir) or is_within(cache_root, config.raw_dir):
        raise ValueError(f"V2 fingerprint cache must remain below work_dir: {cache_root}")
    cache_path = cache_root / "dedupe_v2" / f"{key}.json"
    cached = read_cached_json(cache_path, key)
    if _is_complete_fingerprint(cached, expected_count):
        return list(cached)
    values = [
        fingerprint
        for segment in variant.source_segments
        for fingerprint in _fingerprint_segment(source_for_analysis, segment.source_in, segment.duration, interval, toolchain)
    ]
    if not _is_complete_fingerprint(values, expected_count):
        raise RuntimeError(f"computed fingerprint expected {expected_count} values, got {len(values)}")
    write_cached_json(cache_path, key, values)
    return values


def _hamming_similarity(left: int, right: int) -> float:
    return 1.0 - ((left ^ right).bit_count() / 56.0)


def _sequence_similarity(left: Sequence[int], right: Sequence[int]) -> float:
    if not left or not right:
        return 0.0
    short, long = (list(left), list(right)) if len(left) <= len(right) else (list(right), list(left))
    if len(short) == len(long):
        windows = [long]
    else:
        windows = [long[start : start + len(short)] for start in range(len(long) - len(short) + 1)]
    return max(float(np.mean([_hamming_similarity(a, b) for a, b in zip(short, window)])) for window in windows)


def deduplicate_candidates(
    candidates: Sequence[Candidate],
    fingerprints: Mapping[str, Sequence[int]],
    threshold: float,
) -> DedupeResult:
    ordered = sorted(candidates, key=lambda candidate: candidate.candidate_id)
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    similarities: dict[str, float] = {}
    for left_index, left in enumerate(ordered):
        for right_index in range(left_index + 1, len(ordered)):
            right = ordered[right_index]
            similarity = _sequence_similarity(fingerprints.get(left.candidate_id, []), fingerprints.get(right.candidate_id, []))
            similarities[f"{left.candidate_id}:{right.candidate_id}"] = round(similarity, 6)
            if similarity >= threshold:
                union(left_index, right_index)
    grouped: dict[int, list[Candidate]] = {}
    for index, candidate in enumerate(ordered):
        grouped.setdefault(find(index), []).append(candidate)
    groups: list[list[Candidate]] = []
    for group_number, group in enumerate(sorted(grouped.values(), key=lambda value: value[0].candidate_id), start=1):
        duplicate_group = f"dg-{group_number:03d}"
        groups.append(
            [
                replace(
                    candidate,
                    duplicate_group=duplicate_group,
                    uniqueness=1.0 / len(group),
                    fingerprint=list(fingerprints.get(candidate.candidate_id, [])),
                )
                for candidate in group
            ]
        )
    return DedupeResult(groups=groups, similarity=similarities)


def choose_representative(group: Sequence[Candidate], purpose: Literal["fast", "full"]) -> Candidate:
    if not group:
        raise ValueError("Cannot choose a representative from an empty duplicate group")
    if purpose == "fast":
        return max(group, key=lambda candidate: (candidate.human_selection_score, candidate.final_score, -candidate.duration))
    return max(group, key=lambda candidate: (candidate.final_score + min(candidate.duration, 40.0) * 0.0005, candidate.continuity_score))


def _source_segments_overlap(left: CandidateVariant, right: CandidateVariant) -> bool:
    return any(
        left_segment.source == right_segment.source
        and left_segment.source_in < right_segment.source_out
        and right_segment.source_in < left_segment.source_out
        for left_segment in left.source_segments
        for right_segment in right.source_segments
    )


def _event_windows_overlap(left: CandidateVariant, right: CandidateVariant, window_s: float = 4.0) -> bool:
    left_events = [event for event in left.payoff_events if event.type in {"kill", "multikill"}]
    right_events = [event for event in right.payoff_events if event.type in {"kill", "multikill"}]
    return any(abs(left_event.source_time - right_event.source_time) <= window_s for left_event in left_events for right_event in right_events)


def choose_v2_representative(group: Sequence[CandidateVariant], purpose: Literal["fast", "full"]) -> CandidateVariant:
    if not group:
        raise ValueError("Cannot choose a representative from an empty duplicate group")
    if purpose == "fast":
        return max(
            group,
            key=lambda variant: (
                variant.human_selection_prior,
                variant.payoff_score,
                variant.context_integrity_score,
                variant.final_score,
                -variant.duration,
            ),
        )
    return max(
        group,
        key=lambda variant: (
            variant.payoff_score,
            variant.context_integrity_score,
            variant.final_score,
            variant.duration,
        ),
    )


def deduplicate_variants(
    variants: Sequence[CandidateVariant], fingerprints: Mapping[str, Sequence[int]], threshold: float
) -> V2DedupeResult:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("dedupe threshold must be in [0, 1]")
    ordered = sorted(variants, key=lambda variant: variant.variant_id)
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root, right_root = find(left_index), find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root

    similarity: dict[str, float] = {}
    forced_reasons: dict[str, str] = {}
    for left_index, left in enumerate(ordered):
        for right_index in range(left_index + 1, len(ordered)):
            right = ordered[right_index]
            pair = f"{left.variant_id}:{right.variant_id}"
            visual_similarity = _sequence_similarity(
                fingerprints.get(left.variant_id, ()), fingerprints.get(right.variant_id, ())
            )
            similarity[pair] = round(visual_similarity, 6)
            if left.parent_candidate_id == right.parent_candidate_id and _source_segments_overlap(left, right):
                union(left_index, right_index)
                forced_reasons[pair] = "same_parent_source_overlap"
            elif left.parent_candidate_id == right.parent_candidate_id and _event_windows_overlap(left, right):
                union(left_index, right_index)
                forced_reasons[pair] = "same_parent_event_window_overlap"
            elif visual_similarity >= threshold:
                union(left_index, right_index)

    grouped: dict[int, list[CandidateVariant]] = {}
    for index, variant in enumerate(ordered):
        grouped.setdefault(find(index), []).append(variant)
    groups: list[tuple[CandidateVariant, ...]] = []
    representative_ids: dict[str, dict[str, str]] = {}
    for group_number, group in enumerate(sorted(grouped.values(), key=lambda value: value[0].variant_id), start=1):
        duplicate_group = f"dg-v2-{group_number:03d}"
        deduped_group = tuple(
            replace(variant, duplicate_group=duplicate_group, uniqueness=1.0 / len(group)) for variant in group
        )
        groups.append(deduped_group)
        representative_ids[duplicate_group] = {
            "fast": choose_v2_representative(deduped_group, "fast").variant_id,
            "full": choose_v2_representative(deduped_group, "full").variant_id,
        }
    return V2DedupeResult(
        groups=tuple(groups), similarity=similarity, threshold=threshold,
        representative_ids=representative_ids, forced_source_overlap_reasons=forced_reasons,
    )


def write_v2_dedupe_summary(result: V2DedupeResult, path: Path) -> None:
    from .cache import atomic_write_json

    atomic_write_json(path, result.to_dict())
