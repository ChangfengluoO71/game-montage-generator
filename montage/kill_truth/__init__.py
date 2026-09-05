"""Battlefield V6 kill-truth primitives.

This package is deliberately independent from the legacy activity/candidate
pipeline.  A skull-row transition is the only path that can create an
``OwnKillEvent``; motion, audio and other HUD changes remain auxiliary data.
"""

from .models import DetectedSkull, KillSequence, OwnKillEvent, SkullRowState

__all__ = ["DetectedSkull", "KillSequence", "OwnKillEvent", "SkullRowState"]
