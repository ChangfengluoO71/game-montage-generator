# Battlefield Montage V1.1 Preview V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a longer, approximately 60–90 second V3 preview that excludes the identified training-range intervals, favors compact rapid-kill sequences, and keeps continuous music underneath audible gameplay audio.

**Architecture:** Preserve the existing V1 baseline and V2 preview. Add transparent, filename-and-time based editorial exclusions to the V2 aggregation path, make rapid-kill condensation a single continuous window, and expose a separate long-preview profile that selects a representative 75–90 second music structure. Fix the V2 audio filter graph by explicitly splitting the game bus before sidechain ducking and final mixing.

**Tech Stack:** Python 3.13, pytest, existing dataclasses/cache pipeline, FFmpeg 8.0 with the tested NVENC executable, ffprobe, NumPy/SciPy/librosa/OpenCV/Matplotlib.

**Spec:** docs/superpowers/specs/2026-09-03-battlefield-montage-v1-1-editorial-optimization-design.md, plus the approved user feedback for Preview V3.

## Global Constraints

- `D:/91/集锦/raw` is permanently read-only; no file below it may be deleted, moved, renamed, overwritten, metadata-modified, or encoded in place.
- Intermediates remain below `D:/91/集锦/work`; finished videos remain below `D:/91/集锦/output`.
- Every run selects and records one tested FFmpeg 8.0 executable and its sibling ffprobe, then reuses those absolute paths for all steps.
- Preserve `D:/91/集锦/output/preview_60s.mp4` and `preview_60s_v2.mp4`; write the new result as `preview_90s_v3.mp4`.
- V3 uses no Video LLM, cloud API, online service, artificial glitch/RGB/zoom/shake/flash/speed-ramp effects, destructive 16:9 crop, upscale, or frame interpolation.
- Music must span the complete rendered V3 duration. Game audio remains present and is allowed to duck music around transients; it must never be used as the only final bus.
- Do not render `Battlefield_Fast_Montage.mp4` or `Battlefield_Full_Highlights.mp4`; stop after V3 preview verification.

## File Map and Interfaces

- Modify `montage/config.py`: add `v2_excluded_ranges`, `v2_max_shots`, and `v2_music_window_policy` with backward-compatible defaults.
- Modify `config.yaml`: record the two approved training-range exclusions, increase the rapid-kill additive cap to `0.18`, and leave the default 45–60 second V2 profile unchanged.
- Add `montage/curation.py`: expose `excluded_range_reason(source, source_start, source_end, config)`, `event_is_editorially_excluded(event, source, config)`, and `variant_is_editorially_excluded(variant, config)`.
- Modify `montage/condense.py`: expose an internal rapid-window helper used by `build_condensed_variants` to retain a tight, continuous multi-kill sequence.
- Modify `montage/ranking.py`: raise the configured rapid-kill bonus cap from `0.12` to `0.18` without double-counting it during beam placement.
- Modify `montage/music_analysis.py`: support representative-window selection for the long profile while retaining the locked baseline behavior for normal V2.
- Modify `montage/beam_timeline.py`: use configurable maximum shot count and the selected V2 music interval throughout beam expansion, target duration, and validation.
- Modify `montage/audio_mix.py`: split the post-volume game bus into independent sidechain and final-mix branches and mix with `duration=longest`.
- Modify `montage/main.py`: filter editorial exclusions before aggregation, add `all-v2-long`, `render-preview-v2-long`, and `verify-preview-v2-long`, and map the long profile to `preview_90s_v3.mp4`.
- Modify tests in `tests/test_condense_ranking.py`, `tests/test_v2_renderer.py`, `tests/test_v2_timeline.py`, and `tests/test_v2_cli_report.py`; add `tests/test_curation.py`.

## Task 1: Lock the new behavior with failing tests

**Files:**
- Modify: `tests/test_v2_renderer.py`
- Modify: `tests/test_condense_ranking.py`
- Modify: `tests/test_v2_timeline.py`
- Modify: `tests/test_v2_cli_report.py`
- Create: `tests/test_curation.py`

**Interfaces:** Tests describe the exact APIs and observable behavior required by Tasks 2–5. Test fixtures use `tmp_path` and never write below the real RAW directory.

- [ ] **Step 1: Add the audio regression test.** Call `build_v2_audio_filter("game", "music", 51.5, base_config())` and assert that the graph contains `asplit=2`, distinct sidechain/mix labels, and `amix=inputs=2:duration=longest`.
- [ ] **Step 2: Add the rapid-window test.** Build a long candidate with two or more `kill`/`multikill` events within four seconds and assert the condensed variant is one segment, includes the first and last rapid events, has a tight duration, and states a continuous rapid-kill rationale.
- [ ] **Step 3: Add curation tests.** Assert that an overlapping source interval returns `training_range`, a non-overlapping interval is allowed, an excluded event is removed, and a variant overlapping an exclusion is rejected.
- [ ] **Step 4: Add the long-profile timeline test.** With a copied config set to 75–90 seconds, 18 maximum shots, and representative music policy, assert the built EDL uses the representative music interval and stays within 75–90 seconds with no more than 18 shots.
- [ ] **Step 5: Run only the new focused tests and observe expected RED failures.**

Run:

```powershell
D:/miniconda/python.exe -m pytest -q tests/test_curation.py tests/test_v2_renderer.py tests/test_condense_ranking.py tests/test_v2_timeline.py tests/test_v2_cli_report.py
```

Expected: failures identify missing curation/config/profile behavior and the current un-split audio graph; existing unrelated tests may remain green.

## Task 2: Add transparent curation and rapid-kill calibration

**Files:**
- Modify: `montage/config.py`
- Modify: `config.yaml`
- Create: `montage/curation.py`
- Modify: `montage/main.py`
- Modify: `montage/condense.py`
- Modify: `montage/ranking.py`
- Modify: `tests/test_curation.py`
- Modify: `tests/test_condense_ranking.py`

**Interfaces:** `v2_excluded_ranges` is a tuple of `(source_name, start, end, reason)`. Curation uses basename equality and half-open interval overlap (`end > excluded_start and start < excluded_end`). Long candidate windows are not discarded wholesale; only detected events and condensed variants overlapping an exclusion are removed.

- [ ] **Step 1: Add config parsing and YAML data.** Parse finite, positive exclusion intervals; configure the two ranges in `Battlefield 6 2026.09.03 - 14.43.43.02.DVR.mp4` at `49.0–55.0` and `68.667–74.667`, both with reason `training_range`. Add `v2_max_shots=14` and `v2_music_window_policy=baseline` defaults. Set `rapid_multikill_bonus_weight=0.18`.
- [ ] **Step 2: Implement the three curation predicates.** Return the matching exclusion reason for overlap, support event timestamps, and inspect every `SourceSegment` in a variant.
- [ ] **Step 3: Filter V2 detected events and generated variants in `run_v2_analysis_pipeline`.** Increment a reportable `editorial_exclusion` rejection count and include the configured exclusions in the V2 cache key/stage version.
- [ ] **Step 4: Implement `_best_rapid_window`.** Find the highest-value cluster of distinct kill/multikill events inside `rapid_multikill_window_s`, add about one second of context before the first and after the last event, clip to the candidate, and return one continuous segment. Use this segment before ordinary anchor condensation for long candidates; retain all clustered events and write a rationale explaining why the sequence stays intact.
- [ ] **Step 5: Change `RAPID_MULTIKILL_MAX_BONUS` and related expectations to `0.18`.** Keep the existing one-time additive bonus and 4-second/two-event qualification.
- [ ] **Step 6: Run focused curation/condense/ranking tests and the existing ranking suite.**

## Task 3: Implement long representative music/timeline profile

**Files:**
- Modify: `montage/config.py`
- Modify: `montage/music_analysis.py`
- Modify: `montage/beam_timeline.py`
- Modify: `montage/main.py`
- Modify: `tests/test_v2_timeline.py`
- Modify: `tests/test_v2_cli_report.py`

**Interfaces:** Normal `all-v2` remains baseline-locked at its existing 45–60 second policy. The new long commands use a copied config with `preview_min_duration=60.0`, `preview_max_duration=90.0`, `v2_max_shots=18`, `v2_music_window_policy=representative`, and `v2_output_name=preview_90s_v3.mp4`.

- [ ] **Step 1: Add representative policy branching to `analyze_music_v2`.** For the long profile, call the existing representative structure chooser with the configured 60–90 second bounds; retain baseline/±0.5 behavior for default V2. Add policy and duration bounds to the music cache key.
- [ ] **Step 2: Make beam expansion, target duration, and validation use `config.v2_max_shots` and the selected music interval.** Use the representative `preview_music_in/out` when policy is `representative`; reject intervals shorter than the requested rendered duration.
- [ ] **Step 3: Add explicit long CLI commands and profile mapping.** Keep dry-run before render, and keep full-output guards unchanged.
- [ ] **Step 4: Run the long-profile timeline/CLI tests, then run all V2 timeline/music/CLI tests.**

## Task 4: Fix continuous music/game-audio mixing

**Files:**
- Modify: `montage/audio_mix.py`
- Modify: `tests/test_v2_renderer.py`

**Interfaces:** `build_v2_audio_filter` preserves the current game-first sidechain strategy, but the post-volume game bus becomes two labels: one feeds `sidechaincompress`, and one feeds the final `amix`. `amix` uses the longer input and the existing explicit trim/pad duration.

- [ ] **Step 1: Implement the minimal graph change.** Add `asplit=2[game_sidechain][game_mix]`, use `[game_sidechain]` for ducking, use `[game_mix]` for final mixing, and change `duration=first` to `duration=longest`.
- [ ] **Step 2: Run the audio regression test and the complete V2 renderer suite.**
- [ ] **Step 3: Generate a small work-only synthetic or cached audio check with FFmpeg 8 and confirm the final mix has the full requested duration and no latter-half silence.**

## Task 5: Run, render, and verify V3

**Files/artifacts:**
- Create/update only below `D:/91/集锦/work` and `D:/91/集锦/output/preview_90s_v3.mp4`.

- [ ] **Step 1: Run the complete test suite, compileall, and diff check.**
- [ ] **Step 2: Record RAW manifest and V1 baseline manifest before the real run.**
- [ ] **Step 3: Run `D:/miniconda/python.exe main.py all-v2-long --dry-run`; inspect `work/analysis/preview/preview_v2_edit.json` before rendering.** Confirm 60–90 seconds, no selected source range overlaps either training-range interval, EDL fields are complete, music interval spans the target duration, and no full montage output exists.
- [ ] **Step 4: Run `D:/miniconda/python.exe main.py all-v2-long` to render only `preview_90s_v3.mp4`.**
- [ ] **Step 5: Run `D:/miniconda/python.exe main.py verify-preview-v2-long` plus explicit FFmpeg 8 full decode, ffprobe duration/geometry/fps/codec checks, packet/frame monotonicity checks, and full-range audio loudness/silence checks. Music must cover the entire output; game transients must remain measurable.
- [ ] **Step 6: Re-check RAW manifest, V1 baseline SHA/size/mtime, V2 full-output absence, and FFmpeg/ffprobe paths in `environment_v2.json` and `pipeline.log`.
- [ ] **Step 7: Write the final V3 report with selected candidates, deduped/excluded items, music interval/reason, shot statistics, transition counts, sync statistics, audio strategy, encoder path, and the absolute preview path. Stop for human review.

## Self-review checklist

- The user’s two false-positive target-range intervals are explicitly excluded without touching RAW.
- Rapid multi-kills remain continuous and receive a larger additive ranking bonus.
- The long profile is opt-in and cannot change the prior 45–60 second V2 command.
- The audio graph no longer consumes one game stream through two filter branches without `asplit`.
- Music duration is validated against the rendered edit duration before render.
- EDL is generated and inspected before the V3 render.
- No task renders either full montage.
