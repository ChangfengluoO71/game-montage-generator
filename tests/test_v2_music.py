from dataclasses import replace
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from montage.music_analysis import _downbeat_attempts, analyze_music_v2, choose_v2_music_window
from montage.models import EditDecisionList


def make_structure():
    return {
        "regions": [
            {"start": 0.0, "end": 18.0, "energy": "low_energy", "role": "intro", "confidence": 0.4},
            {"start": 18.0, "end": 75.0, "energy": "high_energy", "role": "chorus", "confidence": 0.4},
        ],
        "boundaries": [18.0],
        "confidence": 0.4,
    }


def make_baseline_edit():
    return EditDecisionList(
        kind="preview",
        music_source=Path("fixture.wav"),
        music_in=19.0,
        music_out=74.252,
        duration=55.252,
        music_reason="V1 baseline",
        shots=[],
    )


@pytest.fixture
def music_config(base_config, tmp_path):
    music_file = tmp_path / "music.wav"
    sample_rate = 22050
    times = np.arange(sample_rate * 3) / sample_rate
    samples = (0.3 * np.sin(2 * np.pi * 220 * times) + 0.2 * (np.sin(2 * np.pi * 2 * times) > 0)).astype(np.float32)
    with wave.open(str(music_file), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((samples * 32767).astype("<i2").tobytes())
    return replace(
        base_config,
        music_file=music_file,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
    )


def test_window_stays_at_baseline_without_better_boundary():
    decision = choose_v2_music_window(make_structure(), 19.0, 74.252, 230.0, 0.5)
    assert decision.v2_music_in == 19.0
    assert decision.v2_music_out == 74.252
    assert decision.changed is False


def test_v2_music_uses_separate_artifacts(music_config, fake_toolchain):
    analysis, decision = analyze_music_v2(music_config, fake_toolchain, make_baseline_edit())
    assert (music_config.music_v2_analysis_dir / "beat_map_v2.json").exists()
    assert (music_config.music_v2_analysis_dir / "music_structure_v2.json").exists()
    assert (music_config.music_v2_analysis_dir / "energy_curve_v2.csv").exists()
    assert (music_config.music_v2_analysis_dir / "music_analysis_v2.png").exists()
    assert decision.baseline_music_in == 19.0


def test_percussive_confidence_is_recorded(music_config, fake_toolchain):
    analysis, _ = analyze_music_v2(music_config, fake_toolchain, make_baseline_edit())
    assert "percussive" in analysis.confidence
    assert "phrase" in analysis.confidence


def test_v2_cache_invalidates_when_baseline_window_changes(music_config, fake_toolchain):
    analyze_music_v2(music_config, fake_toolchain, make_baseline_edit())
    first_key = json.loads(music_config.music_v2_cache_path.read_text(encoding="utf-8"))["cache_key"]

    _, decision = analyze_music_v2(
        music_config,
        fake_toolchain,
        replace(make_baseline_edit(), music_in=20.0, music_out=75.0),
    )

    assert decision.baseline_music_in == 20.0
    assert decision.baseline_music_out == 75.0

    analyze_music_v2(replace(music_config, baseline_music_max_shift=0.25), fake_toolchain, make_baseline_edit())
    assert json.loads(music_config.music_v2_cache_path.read_text(encoding="utf-8"))["cache_key"] != first_key


def test_steady_beat_grid_cannot_move_baseline_as_a_phrase():
    structure = {
        "phrase_boundaries": [{"timestamp": 19.2, "confidence": 0.98, "support": "beat_grid_only"}],
        "phrase_confidence": 0.98,
        "regions": [],
    }

    decision = choose_v2_music_window(structure, 19.0, 74.252, 230.0, 0.5)

    assert decision.changed is False
    assert decision.v2_music_in == 19.0


def test_downbeats_remain_low_confidence_heuristic_attempts():
    attempts, confidence = _downbeat_attempts(np.arange(0.0, 32.0, 0.5))

    assert attempts == list(np.arange(0.0, 32.0, 2.0))
    assert confidence <= 0.25
