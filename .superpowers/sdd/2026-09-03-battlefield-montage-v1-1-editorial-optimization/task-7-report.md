# Task 7 implementation report

## Status

DONE — replacement review fix

## Important findings fixed

- `CandidateVariant.final_score` is used once; the stored rapid-multikill bonus is not added again by beam scoring.
- Anchor placement uses the supplied baseline edit’s `music_in`, while preserving the baseline interval in the V2 EDL.
- `V2EditShot` carries source, environment, and weapon/view signatures through Task 6 transition reconstruction.
- EDL validation requires every shot and segment source to be an existing regular file resolved below `raw`.
- Shot ranges must match ordered segments; segments must be chronological, contiguous, non-overlapping, and duration-consistent.
- Preferred ordinary macro duration and `hero_max_duration` are enforced. A longer ordinary shot is accepted only with a documented condensation reason and sufficient context integrity.
- Energy fitting uses each beam state’s selected 45–60s target duration rather than a fixed preview maximum.
- Hero finale preference is reserved for candidates within `hero_quality_margin`, with an explicit closing-shot quality comparison.
- `beam_max_expansions` is enforced as a shared global candidate-expansion budget for the complete `build_v2_preview_edit` search; `expand_beam` retains its standalone per-call cap when no shared counter is supplied.
- Two-segment condensed shots may retain source-time gaps only for `downtime_removed`, `phrase_boundary`, `bar_boundary`, `action_phase_change`, `substantial_spatial_change`, or `substantial_state_change`; overlap, bad ordering, and unexplained gaps remain rejected.
- Legacy `V2EditShot.source_in/source_out` now describe the first source segment, while `source_segments` remains authoritative for the selected content.
- Timeline validation requires first `timeline_in == 0`, exact contiguous boundaries, and final `timeline_out == edit.duration`.
- Removed the unused import introduced by the original implementation.

## Regression coverage

`tests/test_v2_timeline.py` now covers baseline start placement, no double rapid-multikill bonus, signature preservation, missing/directory sources, documented condensed gaps, overlap/order and unexplained-gap rejection, legacy condensed ranges, duration limits, dynamic energy targets, a build-wide expansion bound independent of beam width, contiguous timelines, and Hero quality margins.

## Verification

- `D:/miniconda/python.exe -m pytest -q tests/test_v2_timeline.py` — **28 passed**
- `D:/miniconda/python.exe -m pytest -q` — **130 passed, 2 pre-existing audioread Python 3.13 deprecation warnings**
- `D:/miniconda/python.exe -m compileall -q .` — exit **0**
- `git diff --check` — exit **0**

No media was rendered, and no files under `raw` were modified.
