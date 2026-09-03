from pathlib import Path
from types import SimpleNamespace

import main
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_dry_run_does_not_call_renderer(monkeypatch):
    called = {"render": False, "analysis": False}

    monkeypatch.setattr("main._run_analysis_pipeline", lambda *args, **kwargs: called.__setitem__("analysis", True))
    monkeypatch.setattr("main.render_preview_stage", lambda *args, **kwargs: called.__setitem__("render", True))

    assert main.run_pipeline("all", PROJECT_ROOT / "config.yaml", dry_run=True) == 0
    assert called["analysis"] is True
    assert called["render"] is False


def test_render_fast_and_full_are_blocked_before_preview_approval():
    assert main.run_pipeline("render-fast", PROJECT_ROOT / "config.yaml") != 0
    assert main.run_pipeline("render-full", PROJECT_ROOT / "config.yaml") != 0


def test_cli_exposes_required_commands():
    parser = main.build_parser()

    assert {
        "index",
        "analyze-music",
        "analyze-video",
        "candidates",
        "review",
        "render-preview",
        "render-fast",
        "render-full",
        "all",
    } <= set(parser._subparsers._group_actions[0].choices)


def test_p95_uses_nearest_rank_for_small_samples():
    assert main._p95([0.04, 0.082, 0.23]) == pytest.approx(0.23)


def test_transition_counts_exclude_initial_shot():
    shots = [
        SimpleNamespace(transition="hard_cut"),
        SimpleNamespace(transition="hard_cut"),
        SimpleNamespace(transition="match_cut"),
    ]
    assert main._transition_counts(shots) == {"hard_cut": 1, "match_cut": 1}
