from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from montage.kill_truth.curation import init_curation_ledger, set_curation_status
from montage.kill_truth.calibration import _frame_calibration_metrics
from montage.kill_truth.metrics import match_event_times
from montage.kill_truth.models import (
    DetectedSkull,
    KillSequence,
    OwnKillEvent,
    SkullRowState,
)
from montage.kill_truth.panel_state import TemporalStateMachine
from montage.kill_truth.profile import HudProfile, v6_cache_key
from montage.kill_truth.refinement import resolve_count_jump
from montage.kill_truth.skull_detector import SkullDetector
from montage.kill_truth.skull_row import build_skull_row
from montage.kill_truth.scanner import _even_crop_bounds, iter_cropped_frames


def _skull(x: int, y: int, kind: str = "NORMAL") -> DetectedSkull:
    return DetectedSkull(
        bbox=(x, y, 20, 24),
        center=(x + 10.0, y + 12.0),
        kind=kind,
        template_score=0.98,
        color_score=0.92 if kind == "HEADSHOT" else 0.0,
    )


def _state(
    timestamp: float,
    count: int,
    *,
    present: bool = True,
    headshots: int = 0,
    y: int = 20,
) -> SkullRowState:
    detections = tuple(
        _skull(10 + index * 30, y, "HEADSHOT" if index >= count - headshots else "NORMAL")
        for index in range(count)
    )
    return SkullRowState(
        timestamp=timestamp,
        panel_present=present,
        skull_count=count,
        normal_count=count - headshots,
        headshot_count=headshots,
        geometry_score=0.96 if present else 0.0,
        row_bbox=(10, y, max(20, count * 30 - 10), 24) if count else None,
        confidence=0.97 if present else 0.0,
        detections=detections,
        panel_structure_score=0.94 if present else 0.0,
    )


def test_v6_models_round_trip_and_nullable_impact():
    event = OwnKillEvent(
        event_id="src-e1",
        source_id="src",
        source_path=Path(r"D:\raw\a.mp4"),
        sequence_id="src-seq-1",
        type="OWN_KILL",
        confirmation_time=15.923,
        impact_time=None,
        sequence_index=1,
        skull_count_before=0,
        skull_count_after=1,
        kill_count_delta=1,
        kill_type="UNKNOWN_KILL",
        confidence=0.992,
        evidence={"skull_template_score": 0.97, "row_geometry_score": 0.96},
        dense_refinement_used=False,
    )
    restored = OwnKillEvent.from_dict(event.to_dict())
    assert restored == event
    assert restored.impact_time is None
    assert "verified_kill" not in event.to_dict()


def test_static_row_is_one_state_and_does_not_repeat_event():
    machine = TemporalStateMachine("src", Path("raw.mp4"), panel_disappear_s=0.6)
    for timestamp in (1.0, 1.067, 1.133, 1.2):
        machine.consume(_state(timestamp, 1))
    assert len(machine.events) == 1
    assert machine.events[0].skull_count_before == 0
    assert machine.events[0].skull_count_after == 1
    assert machine.events[0].kill_count_delta == 1
    assert machine.events[0].type == "OWN_KILL"


def test_count_increase_creates_new_event_even_under_300ms():
    machine = TemporalStateMachine("src", Path("raw.mp4"), panel_disappear_s=0.6)
    machine.consume(_state(9.0, 1))
    machine.consume(_state(9.2, 2))
    machine.consume(_state(10.0, 3))
    machine.consume(_state(10.3, 4))
    assert len(machine.events) == 4
    assert machine.events[3].confirmation_time == pytest.approx(10.3)
    assert machine.events[3].skull_count_before == 3
    assert machine.events[3].skull_count_after == 4


def test_count_drop_from_occlusion_does_not_end_active_sequence():
    machine = TemporalStateMachine("src", Path("raw.mp4"), panel_disappear_s=0.6)
    machine.consume(_state(1.0, 1))
    machine.consume(_state(1.1, 0, present=False))
    machine.consume(_state(1.2, 1))
    machine.consume(_state(1.3, 2))
    assert [(event.skull_count_before, event.skull_count_after) for event in machine.events] == [(0, 1), (1, 2)]


def test_panel_disappearance_ends_sequence():
    machine = TemporalStateMachine("src", Path("raw.mp4"), panel_disappear_s=0.5)
    machine.consume(_state(2.0, 1))
    machine.consume(_state(2.2, 1, present=False))
    assert machine.sequences == []
    machine.consume(_state(2.7, 0, present=False))
    machine.finish()
    assert len(machine.sequences) == 1
    assert machine.sequences[0].sequence_type == "SINGLE_KILL"
    assert machine.sequences[0].event_ids == (machine.events[0].event_id,)


def test_dense_refinement_recovers_2_to_3_to_4():
    request = {
        "source_id": "src",
        "source_path": r"D:\raw\a.mp4",
        "before_count": 2,
        "after_count": 4,
        "coarse_time": 8.0,
        "reason": "coarse_count_jump",
    }
    events = resolve_count_jump(
        request,
        [_state(8.10, 2), _state(8.22, 3), _state(8.31, 4)],
    )
    assert [(event.skull_count_before, event.skull_count_after) for event in events] == [(2, 3), (3, 4)]
    assert all(event.dense_refinement_used for event in events)


def test_true_2_to_4_becomes_simultaneous_delta_two():
    request = {
        "source_id": "src",
        "source_path": r"D:\raw\a.mp4",
        "before_count": 2,
        "after_count": 4,
        "coarse_time": 8.0,
        "reason": "coarse_count_jump",
    }
    events = resolve_count_jump(request, [_state(8.2, 4)])
    assert len(events) == 1
    assert events[0].type == "SIMULTANEOUS_MULTI_KILL"
    assert events[0].kill_count_delta == 2


def test_dense_target_remains_valid_when_panel_later_drops_during_fade():
    request = {
        "source_id": "src",
        "source_path": r"D:\raw\a.mp4",
        "before_count": 0,
        "after_count": 2,
        "coarse_time": 8.0,
        "reason": "initial_multi_count_requires_dense_refinement",
    }
    events = resolve_count_jump(
        request,
        [_state(8.10, 2), _state(8.25, 2), _state(8.40, 1)],
    )
    assert len(events) == 1
    assert events[0].type == "SIMULTANEOUS_MULTI_KILL"
    assert events[0].confirmation_time == pytest.approx(8.10)
    assert events[0].kill_count_delta == 2


def test_row_geometry_rejects_y_misaligned_detections():
    state = build_skull_row(3.0, [_skull(10, 20), _skull(40, 21), _skull(70, 20)])
    assert state.panel_present
    assert state.skull_count == 3
    rejected = build_skull_row(3.0, [_skull(10, 20), _skull(40, 60), _skull(70, 20)])
    assert rejected.skull_count < 3 or rejected.geometry_score < 0.7


def test_headshot_requires_local_color_evidence_not_background():
    normal = _skull(10, 20, "NORMAL")
    orange_background = _skull(40, 20, "NORMAL")
    state = build_skull_row(1.0, [normal, orange_background])
    assert state.headshot_count == 0


def test_profile_normalized_roi_and_cache_invalidation():
    profile = HudProfile(
        profile_id="cfl_bf6_1920x1200_v1",
        width=1920,
        height=1200,
        search_roi=(0.30, 0.55, 0.48, 0.74),
        row_rois=((0.30, 0.56, 0.47, 0.63), (0.32, 0.61, 0.47, 0.72)),
        normal_threshold=0.82,
        headshot_threshold=0.82,
        geometry_threshold=0.62,
        template_paths=(),
    )
    assert profile.pixel_roi(1920, 1200) == (576, 660, 921, 888)
    first = v6_cache_key({"absolute_path": "a", "size": 1, "mtime": 2}, profile, "templates-a")
    second = v6_cache_key({"absolute_path": "a", "size": 1, "mtime": 2}, profile.with_version("v2"), "templates-a")
    assert first != second


def test_curation_reject_overrides_automatic_status():
    ledger = init_curation_ledger([{"source_id": "s1", "source_path": "a.mp4"}])
    ledger = set_curation_status(ledger, "s1", "MANUAL_REJECT")
    assert ledger["sources"]["s1"]["status"] == "MANUAL_REJECT"
    assert ledger["sources"]["s1"]["editorial_excluded"] is True


def test_event_matching_uses_time_tolerance_and_reports_unmatched():
    result = match_event_times([1.0, 2.0, 2.4], [1.2, 3.0], tolerance=0.25)
    assert result["tp"] == 1
    assert result["fp"] == 2
    assert result["fn"] == 1


def test_calibration_metrics_do_not_count_correct_negatives_as_false_positives():
    result = _frame_calibration_metrics(
        [
            {"expected_panel": True, "panel_ok": True, "count_ok": True, "headshot_ok": True},
            {"expected_panel": True, "panel_ok": True, "count_ok": False, "headshot_ok": True},
            {"expected_panel": False, "panel_ok": True},
            {"expected_panel": False, "panel_ok": False},
        ]
    )
    assert result["positive_panel_hits"] == 2
    assert result["positive_full_hits"] == 1
    assert result["negative_panel_correct"] == 1
    assert result["negative_panel_false_positives"] == 1
    assert result["frame_panel_precision"] == pytest.approx(2 / 3)
    assert result["frame_panel_recall"] == 1.0


def test_detector_matches_normal_and_headshot_only_inside_profile_roi():
    cv2 = pytest.importorskip("cv2")
    normal = np.zeros((24, 20, 3), dtype=np.uint8)
    cv2.circle(normal, (10, 10), 8, (225, 225, 225), -1)
    cv2.circle(normal, (7, 10), 2, (10, 10, 10), -1)
    cv2.circle(normal, (13, 10), 2, (10, 10, 10), -1)
    headshot = normal.copy()
    colored = np.all(headshot > 100, axis=2)
    headshot[colored] = (35, 90, 245)
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    frame[35:59, 30:50] = normal
    frame[35:59, 70:90] = headshot
    frame[0:24, 130:150] = normal  # Same visual outside the search ROI.
    profile = HudProfile(
        profile_id="test",
        width=180,
        height=120,
        search_roi=(0.05, 0.20, 0.60, 0.75),
        row_rois=((0.05, 0.20, 0.60, 0.75),),
        normal_threshold=0.72,
        headshot_threshold=0.72,
        geometry_threshold=0.55,
        panel_structure_threshold=0.0,
        orange_color_threshold=0.20,
    )
    state = SkullDetector(profile, normal_templates=(normal,), headshot_templates=(headshot,)).detect(frame, 4.0)
    assert state.panel_present
    assert state.skull_count == 2
    assert state.normal_count == 1
    assert state.headshot_count == 1
    assert all(item.center[0] < 110 for item in state.detections)


def test_cropped_frame_pipe_accumulates_short_reads(monkeypatch):
    class _ChunkedPipe:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.offset = 0

        def read(self, requested: int | None = None) -> bytes:
            if self.offset >= len(self.payload):
                return b""
            size = min(3, requested or len(self.payload), len(self.payload) - self.offset)
            chunk = self.payload[self.offset : self.offset + size]
            self.offset += size
            return chunk

        def close(self):
            return None

    class _Process:
        def __init__(self):
            self.stdout = _ChunkedPipe(bytes(range(12)))
            self.stderr = _ChunkedPipe(b"")

        def wait(self):
            return 0

    monkeypatch.setattr("montage.kill_truth.scanner.subprocess.Popen", lambda *args, **kwargs: _Process())
    class _Toolchain:
        ffmpeg = Path("ffmpeg.exe")

    frames = list(iter_cropped_frames(_Toolchain(), Path("raw.mp4"), fps=1.0, crop_bounds=(0, 0, 2, 2)))
    assert len(frames) == 1
    assert frames[0][1].shape == (2, 2, 3)


def test_rawvideo_crop_bounds_are_even_for_yuv420():
    assert _even_crop_bounds((576, 660, 921, 888)) == (576, 660, 920, 888)
