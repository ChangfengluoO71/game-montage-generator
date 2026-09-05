from dataclasses import replace
from pathlib import Path

from montage.candidate import generate_candidates, write_candidates
from montage.config import load_config
from montage.dedupe import choose_representative, deduplicate_candidates
from montage.models import Candidate, VideoAnalysis
from montage.ranking import score_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def candidate_with_id(candidate_id, duration=5.0, human=0.0, score=0.5):
    return Candidate(
        candidate_id=candidate_id,
        source_file=Path(f"{candidate_id}.mp4"),
        source_start=0.0,
        source_end=duration,
        duration=duration,
        human_selection_score=human,
        audio_score=score,
        motion_score=score,
        visual_score=score,
        continuity_score=score,
        final_score=score,
        combat_intensity=score,
        uniqueness=1.0,
    )


def candidate_with_scores(human, combat, motion, audio, visual, continuity, unique):
    return Candidate(
        candidate_id="scores",
        source_file=Path("scores.mp4"),
        source_start=0.0,
        source_end=5.0,
        duration=5.0,
        human_selection_score=human,
        audio_score=audio,
        motion_score=motion,
        visual_score=visual,
        continuity_score=continuity,
        combat_intensity=combat,
        uniqueness=unique,
    )


def test_short_clip_gets_human_selection_prior_without_fragmentation(short_record, base_config):
    analysis = VideoAnalysis(
        source_file=short_record.file_path,
        sample_rate=2.0,
        times=[0.0, 1.0, 2.0, 3.0],
        motion=[0.2, 0.8, 0.8, 0.2],
        visual=[0.2, 0.8, 0.8, 0.2],
        audio=[0.2, 0.8, 0.8, 0.2],
        continuity=[0.2, 0.8, 0.8, 0.2],
        activity=[0.2, 0.8, 0.8, 0.2],
    )

    candidates = generate_candidates([short_record], {str(short_record.file_path): analysis}, base_config)

    assert len(candidates) == 1
    assert candidates[0].human_selection_score == 0.85
    assert candidates[0].duration >= 10.0


def test_long_activity_peaks_merge_into_one_window(long_record, base_config):
    analysis = VideoAnalysis(
        source_file=long_record.file_path,
        sample_rate=1.0,
        times=[float(index) for index in range(20)],
        motion=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0] + [0.0] * 9,
        visual=[0.0] * 5 + [1.0, 1.0, 0.0, 1.0, 1.0, 1.0] + [0.0] * 9,
        audio=[0.0] * 5 + [1.0, 1.0, 0.0, 1.0, 1.0, 1.0] + [0.0] * 9,
        continuity=[0.0] * 5 + [1.0, 1.0, 0.0, 1.0, 1.0, 1.0] + [0.0] * 9,
        activity=[0.0] * 5 + [1.0, 1.0, 0.0, 1.0, 1.0, 1.0] + [0.0] * 9,
        candidate_windows=[{"start": 5.0, "end": 12.0}],
    )

    candidates = generate_candidates([long_record], {str(long_record.file_path): analysis}, base_config)

    assert any(candidate.source_end - candidate.source_start >= 6.0 for candidate in candidates)


def test_weighted_score_is_configurable():
    base = load_config(PROJECT_ROOT / "config.yaml")
    config = replace(base, weights={"human_selection_prior": 1.0})
    candidate = candidate_with_scores(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    assert score_candidates([candidate], config)[0].final_score == 1.0


def test_duplicate_group_prefers_saved_short_for_fast():
    short = candidate_with_id("short", duration=20, human=0.4)
    long = candidate_with_id("long", duration=28, human=0.0)

    result = deduplicate_candidates([short, long], {"short": [0, 1, 2], "long": [0, 1, 2]}, threshold=0.9)

    assert choose_representative(result.groups[0], "fast").candidate_id == "short"
    assert result.groups[0][0].duplicate_group == result.groups[0][1].duplicate_group


def test_dedupe_result_keeps_visual_fingerprint_on_candidates():
    candidate = candidate_with_id("fingerprinted")

    result = deduplicate_candidates([candidate], {"fingerprinted": [11, 22, 33]}, threshold=0.9)

    assert result.groups[0][0].fingerprint == [11, 22, 33]
