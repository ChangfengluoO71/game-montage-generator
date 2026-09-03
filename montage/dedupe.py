"""Visual fingerprint comparison and duplicate group preference."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from .models import Candidate, DedupeResult
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
