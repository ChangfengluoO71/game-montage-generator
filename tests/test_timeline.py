from dataclasses import replace
from pathlib import Path

from montage.config import load_config
from montage.models import Candidate, MusicAnalysis, VideoAnalysis
from montage.timeline import build_preview_edit, write_edit_list
from montage.transitions import choose_transition


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def candidate_with_id(candidate_id, duration=7.0, score=0.5):
    return Candidate(
        candidate_id=candidate_id,
        source_file=Path(f"{candidate_id}.mp4"),
        source_start=0.0,
        source_end=duration,
        duration=duration,
        human_selection_score=0.2,
        audio_score=score,
        motion_score=score,
        visual_score=score,
        continuity_score=score,
        final_score=score,
        combat_intensity=score,
        duplicate_group=f"dg-{candidate_id}",
    )


def make_music():
    return MusicAnalysis(
        source_file=Path("music.flac"),
        duration=230.0,
        tempo=117.45,
        beats=[float(index) * 0.5 for index in range(440)],
        strong_beats=[float(index) for index in range(220)],
        bars=[float(index) * 2.0 for index in range(110)],
        onsets=[10.2, 20.5, 35.1],
        edit_points=[
            {"timestamp": 70.0, "strength": 0.7, "type": "phrase"},
            {"timestamp": 85.0, "strength": 0.9, "type": "bar"},
            {"timestamp": 105.0, "strength": 1.0, "type": "section_change"},
            {"timestamp": 110.0, "strength": 0.95, "type": "strong_beat"},
        ],
        energy_times=[70.0, 105.0, 150.0],
        rms=[0.3, 0.7, 0.9],
        onset_strength=[0.2, 0.7, 1.0],
        novelty=[0.1, 0.8, 0.4],
        structure_regions=[
            {"start": 55.0, "end": 105.0, "energy": "medium_energy", "role": "build_up", "confidence": 0.8},
            {"start": 105.0, "end": 160.0, "energy": "high_energy", "role": "chorus", "confidence": 0.85},
        ],
        confidence={"beat": 0.9, "bar": 0.85, "section": 0.8, "onset": 0.6},
        preview_music_in=70.0,
        preview_music_out=125.0,
        preview_reason="build into chorus",
    )


def preview_config():
    base = load_config(PROJECT_ROOT / "config.yaml")
    return replace(base, preview_min_duration=45.0, preview_max_duration=60.0)


def test_preview_energy_rises_and_strongest_shot_is_near_end():
    candidates = [candidate_with_id(f"c{index}", duration=duration, score=score) for index, (duration, score) in enumerate([(5, 0.4), (5, 0.5), (6, 0.6), (8, 0.7), (10, 0.8), (12, 0.95)])]

    edit = build_preview_edit(candidates, make_music(), preview_config())

    assert edit.duration <= 60.0
    assert edit.shots[0].candidate_score < edit.shots[-1].candidate_score


def test_continuous_seven_second_action_is_not_cut_at_every_beat():
    edit = build_preview_edit([candidate_with_id("multi", duration=7.0, score=0.95)], make_music(), preview_config())

    assert len(edit.shots) == 1


def test_edit_shot_contains_required_preview_fields():
    shot = build_preview_edit([candidate_with_id("one", duration=7.0)], make_music(), preview_config()).shots[0].to_dict()

    required = {
        "source",
        "source_in",
        "source_out",
        "duration",
        "candidate_score",
        "duplicate_group",
        "timeline_in",
        "timeline_out",
        "transition",
        "music_target",
        "music_event_type",
        "sync_offset",
        "rationale",
    }
    assert required <= set(shot)


def test_default_transition_is_hard_cut():
    assert choose_transition(None, candidate_with_id("x"), {}) == "hard_cut"


def test_preview_prefers_human_selected_short_clips_over_long_fallbacks():
    long_candidate = candidate_with_id("long", duration=13.0, score=0.2)
    short_candidate = replace(candidate_with_id("short", duration=7.0, score=0.8), source_category="short_clip", human_selection_score=0.42)

    edit = build_preview_edit([long_candidate, short_candidate], make_music(), preview_config())

    assert edit.shots[0].source == short_candidate.source_file


def test_preview_does_not_truncate_a_saved_short_clip_to_fill_remaining_time():
    short_candidate = replace(candidate_with_id("long-short", duration=70.0, score=0.9), source_category="short_clip", human_selection_score=0.42)

    edit = build_preview_edit([short_candidate], make_music(), preview_config())

    assert edit.shots == []


def test_preview_prefers_montage_compatible_saved_clip_lengths():
    long_saved = replace(candidate_with_id("long-saved", duration=48.0, score=0.3), source_category="short_clip", human_selection_score=0.42)
    montage_clip = replace(candidate_with_id("montage", duration=8.0, score=0.8), source_category="short_clip", human_selection_score=0.42)

    edit = build_preview_edit([long_saved, montage_clip], make_music(), preview_config())

    assert edit.shots[0].source == montage_clip.source_file
