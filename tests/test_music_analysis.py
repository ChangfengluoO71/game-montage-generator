from montage.music_analysis import (
    build_edit_points,
    choose_preview_music_window,
    infer_music_structure,
)

import numpy as np


def test_edit_points_prioritize_bar_and_strong_beat():
    points = build_edit_points(
        120.0,
        [0.0, 0.5, 1.0, 1.5],
        [0.2],
        [0.1, 0.8, 0.9, 0.2],
        [0.0, 0.5, 1.0, 1.5],
    )

    assert points[0]["type"] == "bar"
    assert any(point["type"] == "strong_beat" for point in points)
    assert all(0.0 <= float(point["strength"]) <= 1.0 for point in points)


def test_structure_contains_low_medium_high_regions_and_confidence():
    result = infer_music_structure(
        [0, 1, 2, 3],
        [0.1, 0.2, 0.9, 1.0],
        [0.0, 1.0, 2.0, 3.0],
        [0, 0, 1, 1],
    )

    assert {region["energy"] for region in result["regions"]} >= {"low_energy", "high_energy"}
    assert all("confidence" in region for region in result["regions"])


def test_preview_window_prefers_build_to_high_energy_transition():
    structure = {
        "regions": [
            {"start": 70, "end": 105, "energy": "medium_energy", "role": "build_up", "confidence": 0.8},
            {"start": 105, "end": 150, "energy": "high_energy", "role": "chorus", "confidence": 0.7},
        ]
    }

    start, end, reason = choose_preview_music_window(structure, 230, 45, 60)

    assert 45 <= end - start <= 60
    assert start < 105 < end
    assert "high" in reason.lower() or "build" in reason.lower()


def test_structure_smooths_frame_scale_energy_noise():
    times = np.arange(0.0, 120.0, 0.1).tolist()
    energy = []
    for time in times:
        base = 0.1 if time < 30 else 0.45 if time < 55 else 0.9
        energy.append(base + (0.04 if int(time * 10) % 2 else -0.04))

    result = infer_music_structure(times, energy, list(np.arange(0.0, 120.0, 0.5)), energy)

    assert len(result["regions"]) <= 6
    assert max(float(region["end"]) - float(region["start"]) for region in result["regions"]) >= 30.0


def test_preview_window_skips_early_micro_transition():
    structure = {
        "regions": [
            {"start": 0, "end": 2, "energy": "low_energy", "role": "intro", "confidence": 0.4},
            {"start": 2, "end": 3, "energy": "medium_energy", "role": "build_up", "confidence": 0.4},
            {"start": 3, "end": 4, "energy": "high_energy", "role": "chorus", "confidence": 0.4},
            {"start": 60, "end": 90, "energy": "medium_energy", "role": "build_up", "confidence": 0.8},
            {"start": 90, "end": 140, "energy": "high_energy", "role": "chorus", "confidence": 0.8},
        ]
    }

    start, end, _ = choose_preview_music_window(structure, 230, 45, 60)

    assert start >= 60 - 30
    assert start < 90 < end
