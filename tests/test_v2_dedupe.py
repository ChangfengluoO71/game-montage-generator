import json
from pathlib import Path
from dataclasses import replace

import pytest

from montage.config import load_config
from montage.dedupe import (
    choose_v2_representative,
    deduplicate_variants,
    fingerprint_variant,
    write_v2_dedupe_summary,
)
from montage.cache import read_cached_json, v2_variant_fingerprint_cache_key
from montage.candidate import write_v2_candidates
from montage.models import CandidateVariant, PayoffEvent, SourceSegment
from montage.ranking import rapid_multikill_score, score_variant


def variant(
    variant_id: str,
    parent_id: str,
    start: float,
    end: float,
    *,
    human: float = 0.0,
    payoff: float = 0.0,
    context: float = 0.0,
    score: float = 0.0,
) -> CandidateVariant:
    source = Path("C:/raw/clip.mp4")
    segment = SourceSegment(source, start, end, end - start)
    event = PayoffEvent(
        event_id=f"event-{variant_id}", type="combat_climax", source_time=(start + end) / 2,
        confidence=0.8, strength=payoff, semantic_confidence=0.4, evidence={"motion_peak": 0.8},
    )
    return CandidateVariant(
        variant_id=variant_id, parent_candidate_id=parent_id, source_file=source,
        source_segments=(segment,), duration=end - start, human_selection_prior=human,
        payoff_score=payoff, combat_intensity=0.5, action_density=0.5, continuity=0.5,
        visual_novelty=0.5, motion=0.5, audio_activity=0.5, danger_score=0.5,
        uniqueness=1.0, final_score=score, duplicate_group=None, payoff_events=(event,),
        primary_anchor=event, secondary_anchors=(), anchor_event_time=event.source_time,
        anchor_event_type=event.type, anchor_event_strength=event.strength,
        anchor_event_confidence=event.confidence, context_integrity_score=context,
        penalty_values={}, source_signature="clip", environment_signature="map",
        weapon_or_view_signature="rifle", condense_reason="", rationale="test",
    )


def test_fast_representative_retains_saved_short_before_quality_metrics():
    saved_short = variant("saved-short", "parent", 0.0, 6.0, human=0.9, payoff=0.1, context=0.1, score=0.1)
    stronger_variant = variant("stronger", "parent", 0.0, 8.0, human=0.1, payoff=1.0, context=1.0, score=1.0)

    assert choose_v2_representative((saved_short, stronger_variant), "fast") is saved_short


def test_same_parent_overlapping_windows_are_forced_into_one_group_with_reason():
    left = variant("left", "parent", 0.0, 6.0)
    right = variant("right", "parent", 4.0, 10.0)

    result = deduplicate_variants((left, right), {"left": [0], "right": [(1 << 56) - 1]}, threshold=0.78)

    assert len(result.groups) == 1
    assert result.similarity["left:right"] == 0.0
    assert result.forced_source_overlap_reasons == {"left:right": "same_parent_source_overlap"}


def test_v2_dedupe_summary_reports_threshold_and_representatives(tmp_path):
    left = variant("left", "left-parent", 0.0, 4.0, human=0.1, payoff=0.4, context=0.4, score=0.4)
    right = variant("right", "right-parent", 10.0, 14.0, human=0.8, payoff=0.3, context=0.3, score=0.3)
    result = deduplicate_variants((left, right), {"left": [3, 4], "right": [3, 4]}, threshold=0.78)
    output = tmp_path / "dedupe_summary_v2.json"

    write_v2_dedupe_summary(result, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["threshold"] == 0.78
    assert payload["representative_ids"] == {"dg-v2-001": {"fast": "right", "full": "left"}}
    assert payload["similarity"] == {"left:right": 1.0}


def test_variant_fingerprint_cache_identity_changes_with_source_ranges_and_detector_version():
    source = {"absolute_path": "C:/raw/clip.mp4", "size": 100, "mtime": 1.0}
    base = v2_variant_fingerprint_cache_key(
        source, variant_id="variant", parent_candidate_id="parent", ranges=[(0.0, 4.0)],
        interval=2.0, detector_version="payoff-v2", ffmpeg_version="8.0", ffmpeg_path="C:/ffmpeg.exe",
    )
    changed_range = v2_variant_fingerprint_cache_key(
        source, variant_id="variant", parent_candidate_id="parent", ranges=[(0.0, 5.0)],
        interval=2.0, detector_version="payoff-v2", ffmpeg_version="8.0", ffmpeg_path="C:/ffmpeg.exe",
    )
    changed_detector = v2_variant_fingerprint_cache_key(
        source, variant_id="variant", parent_candidate_id="parent", ranges=[(0.0, 4.0)],
        interval=2.0, detector_version="payoff-v3", ffmpeg_version="8.0", ffmpeg_path="C:/ffmpeg.exe",
    )

    assert base != changed_range
    assert base != changed_detector


def test_rapid_multikill_bonus_is_capped_and_serialized_as_score_components():
    base = variant("base", "parent", 0.0, 8.0, payoff=0.8, score=0.5)
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    events = (
        PayoffEvent("kill-1", "kill", 2.0, 0.9, 0.9, 0.9, {"kill_feed": 0.9}),
        PayoffEvent("kill-2", "kill", 5.5, 0.9, 0.9, 0.9, {"kill_feed": 0.9}),
        PayoffEvent("multi", "multikill", 6.0, 0.9, 0.9, 0.9, {"reward_score": 0.9}),
    )
    scored = score_variant(replace(base, payoff_events=events), config)

    assert rapid_multikill_score(events, config.rapid_multikill_window_s, config.rapid_multikill_min_events) == 1.0
    assert scored.score_components["rapid_multikill_score"] == 1.0
    assert scored.score_components["rapid_multikill_bonus"] == 0.12
    payload = scored.to_dict()
    assert payload["rapid_multikill_score"] == 1.0
    assert payload["rapid_multikill_bonus"] == 0.12
    assert scored.final_score > base.final_score


def test_isolated_or_widely_separated_kills_receive_no_rapid_bonus():
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    isolated = (
        PayoffEvent("kill-1", "kill", 2.0, 0.9, 0.9, 0.9, {}),
        PayoffEvent("kill-2", "kill", 6.1, 0.9, 0.9, 0.9, {}),
    )
    assert rapid_multikill_score(isolated, config.rapid_multikill_window_s, config.rapid_multikill_min_events) == 0.0


def test_rapid_kill_sequence_outranks_equivalent_isolated_payoff():
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    base = variant("score", "parent", 0.0, 8.0, payoff=0.8)
    isolated = replace(
        base,
        payoff_events=(PayoffEvent("kill", "kill", 2.0, 0.9, 0.9, 0.9, {}),),
    )
    rapid = replace(
        base,
        payoff_events=(
            PayoffEvent("kill-1", "kill", 2.0, 0.9, 0.9, 0.9, {}),
            PayoffEvent("kill-2", "kill", 5.0, 0.9, 0.9, 0.9, {}),
        ),
    )

    assert score_variant(rapid, config).final_score > score_variant(isolated, config).final_score


def test_rapid_multikill_score_has_the_approved_explicit_signature():
    events = (
        PayoffEvent("one", "kill", 1.0, 1.0, 1.0, 1.0, {}),
        PayoffEvent("two", "kill", 4.0, 1.0, 1.0, 1.0, {}),
    )

    assert rapid_multikill_score(events, 4.0, 2) == 1.0


def test_duplicate_event_ids_count_once_for_rapid_multikill():
    events = (
        PayoffEvent("same", "kill", 1.0, 1.0, 1.0, 1.0, {}),
        PayoffEvent("same", "kill", 1.5, 1.0, 1.0, 1.0, {}),
        PayoffEvent("other", "kill", 5.6, 1.0, 1.0, 1.0, {}),
    )

    assert rapid_multikill_score(events, 4.0, 2) == 0.0


def test_score_audit_fields_are_written_to_json_and_csv(tmp_path):
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    scored = score_variant(
        replace(
            variant("audit", "parent", 0.0, 8.0, payoff=0.8),
            payoff_events=(
                PayoffEvent("one", "kill", 1.0, 1.0, 1.0, 1.0, {}),
                PayoffEvent("two", "kill", 4.0, 1.0, 1.0, 1.0, {}),
            ),
        ),
        config,
    )
    json_path = tmp_path / "work" / "candidates.json"
    csv_path = tmp_path / "work" / "candidates.csv"
    write_v2_candidates([scored], json_path, csv_path, config=replace(config, work_dir=tmp_path / "work"))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    csv_payload = csv_path.read_text(encoding="utf-8-sig")
    assert payload["candidates"][0]["rapid_multikill_score"] == 1.0
    assert payload["candidates"][0]["rapid_multikill_bonus"] == 0.12
    assert "rapid_multikill_score" in csv_payload
    assert "rapid_multikill_bonus" in csv_payload


def _filesystem_variant(tmp_path):
    source = tmp_path / "raw" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    return replace(
        variant("fingerprint", "parent", 0.0, 4.0),
        source_file=source,
        source_segments=(SourceSegment(source, 0.0, 4.0, 4.0),),
    )


def test_fingerprint_rejects_ffmpeg_failure_without_caching(tmp_path, base_config, fake_toolchain, monkeypatch):
    item = _filesystem_variant(tmp_path)
    config = replace(base_config, raw_dir=tmp_path / "raw", work_dir=tmp_path / "work")

    monkeypatch.setattr(
        "montage.dedupe.subprocess.run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 1, "stdout": b"", "stderr": b"failure"})(),
    )
    with pytest.raises(RuntimeError, match="FFmpeg fingerprinting failed"):
        fingerprint_variant(item, item.source_file, fake_toolchain, config)
    assert not list((config.cache_dir / "dedupe_v2").glob("*.json"))


def test_fingerprint_rejects_truncated_ffmpeg_output_without_caching(tmp_path, base_config, fake_toolchain, monkeypatch):
    item = _filesystem_variant(tmp_path)
    config = replace(base_config, raw_dir=tmp_path / "raw", work_dir=tmp_path / "work")
    monkeypatch.setattr(
        "montage.dedupe.subprocess.run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": bytes(2 * 32 * 32), "stderr": b""})(),
    )

    with pytest.raises(RuntimeError, match="expected 3 frames"):
        fingerprint_variant(item, item.source_file, fake_toolchain, config)
    assert not list((config.cache_dir / "dedupe_v2").glob("*.json"))


@pytest.mark.parametrize("payload", ["{not-json", {"cache_key": "wrong", "data": [1]}])
def test_stale_or_corrupt_fingerprint_cache_is_rebuilt(tmp_path, base_config, fake_toolchain, monkeypatch, payload):
    item = _filesystem_variant(tmp_path)
    config = replace(base_config, raw_dir=tmp_path / "raw", work_dir=tmp_path / "work")
    config.cache_dir.joinpath("dedupe_v2").mkdir(parents=True)
    key = v2_variant_fingerprint_cache_key(
        {"absolute_path": str(item.source_file.resolve()), "size": 6, "mtime": item.source_file.stat().st_mtime},
        variant_id=item.variant_id, parent_candidate_id=item.parent_candidate_id, ranges=[(0.0, 4.0)],
        interval=config.fingerprint_interval, detector_version=config.payoff_detector_version,
        ffmpeg_version=fake_toolchain.ffmpeg_version, ffmpeg_path=str(fake_toolchain.ffmpeg.resolve(strict=False)),
    )
    cache_path = config.cache_dir / "dedupe_v2" / f"{key}.json"
    cache_path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("montage.dedupe._fingerprint_segment", lambda *args: [123, 456, 789])

    assert fingerprint_variant(item, item.source_file, fake_toolchain, config) == [123, 456, 789]
    assert read_cached_json(cache_path, key) == [123, 456, 789]


def test_fingerprint_enforces_raw_source_match_and_work_cache_boundary(tmp_path, base_config, fake_toolchain, monkeypatch):
    item = _filesystem_variant(tmp_path)
    config = replace(base_config, raw_dir=tmp_path / "raw", work_dir=tmp_path / "work")
    other = tmp_path / "raw" / "other.mp4"
    other.write_bytes(b"other")

    with pytest.raises(ValueError, match="match variant.source_file"):
        fingerprint_variant(item, other, fake_toolchain, config)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="below raw"):
        fingerprint_variant(item, outside, fake_toolchain, config)
    monkeypatch.setattr(type(config), "cache_dir", property(lambda self: tmp_path / "cache"))
    with pytest.raises(ValueError, match="below work_dir"):
        fingerprint_variant(item, item.source_file, fake_toolchain, config)
