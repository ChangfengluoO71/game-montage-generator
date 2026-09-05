# Battlefield Montage V1.1 Quality Preview V5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a new 120–150 second `preview_quality_v5.mp4` that is dominated by verified Battlefield kills and compact continuous multi-kill sequences, while preserving the existing V4 preview and all RAW media.

**Architecture:** Keep the existing V2/V4 pipeline and cache-compatible behavior as the default. Add an opt-in quality profile that (1) repairs rapid-window boundary handling, (2) expands long saved clips into several non-overlapping verified-kill sequences, (3) propagates the configured human-selection prior through cached variants, (4) ranks by verified kill count and short-window kill density, and (5) rejects any timeline shot without a strict verified-kill anchor. Give the quality profile isolated preview artifacts and a new output name, while reusing unchanged expensive source analysis through cache fingerprints.

**Tech Stack:** Python 3.13, pytest, existing dataclasses and JSON caches, FFmpeg 8.0 with the tested NVENC executable, ffprobe, NumPy/SciPy/librosa/OpenCV/Matplotlib. No cloud services or large visual models.

**Spec:** The approved V1.1 editorial feedback in this conversation, the existing V1.1 design specification, and the root-cause findings from the V4 review.

## Global Constraints

- `D:\91\集锦\raw` is permanently read-only. No file below it may be deleted, moved, renamed, overwritten, encoded in place, or have metadata changed.
- All intermediate files and reports remain below `D:\91\集锦\work`; the only new finished video is `D:\91\集锦\output\preview_quality_v5.mp4`.
- Preserve `preview_60s.mp4`, `preview_60s_v2.mp4`, `preview_90s_v3.mp4`, and `preview_quality_v4.mp4`; do not overwrite them.
- The existing V4 command/profile remains available and unchanged. V5 is an additive quality profile and must never render `Battlefield_Fast_Montage.mp4` or `Battlefield_Full_Highlights.mp4`.
- Every real run selects one tested FFmpeg 8.0 executable and its sibling ffprobe, records both in `environment.json`/the pipeline log, and uses those absolute paths for all steps in that run.
- V5 keeps the native 1920×1200, approximately 60 fps presentation. No upscale, interpolation, destructive 16:9 crop, template speed ramp, glitch, RGB split, spin, zoom, shake, flash spam, or other automatic effect is allowed.
- Music must cover the entire rendered V5 duration. Game audio remains audible at a controlled, uniform low level, with brief ducking of music around verified gunshots, explosions, vehicle cannon fire, and kill transients.
- The EDL must be generated and inspected before rendering. After the V5 preview is rendered and verified, stop for human review.

## Task 1: Lock V5 quality behavior with failing tests

**Files:**

- Modify: `tests/test_condense_ranking.py`
- Modify: `tests/test_v2_timeline.py`
- Modify: `tests/test_v2_cli_report.py`
- Modify: `tests/test_v2_contracts.py` if the existing serialization contract needs an additive field assertion
- Add focused tests only where an existing module is not a natural home

**Interfaces:** New tests use temporary RAW/work/output directories and synthetic `Candidate`, `PayoffEvent`, `CandidateVariant`, `MusicAnalysis`, and `PipelineConfig` values. No test writes to the real RAW tree.

- [ ] **Step 1: Test rapid-window boundary tolerance.** Construct a candidate whose strongest rapid cluster ends exactly at the source end and assert the quality condensation path accepts a one-segment window after millisecond rounding when it retains the verified events and has the configured minimum tail.
- [ ] **Step 2: Test multi-sequence extraction.** Construct one 70-second saved short-clip candidate with two separated clusters of verified kills and assert V5 condensation returns more than one variant, each variant is one continuous segment, the source ranges do not overlap, and each rationale states the retained verified-kill count and trimmed non-event context.
- [ ] **Step 3: Test cached short-clip prior propagation.** Deserialize a cached short-clip variant carrying the stale `0.42` prior through the V5 loader and assert its effective prior is at least the configured `0.85`, while a normal non-short variant is not silently upgraded.
- [ ] **Step 4: Test strict verified-kill classification.** Build events containing generic combat-climax events, weak kill-feed changes, and corroborated kill events; assert only the corroborated kill/multikill events count as verified kills.
- [ ] **Step 5: Test kill-count and density scoring.** Create two otherwise similar variants and assert the quality profile scores the one with more verified kills and the higher short-window kill density first. Assert the default V2 profile keeps the new weights disabled.
- [ ] **Step 6: Test the quality beam gate.** Give the beam a no-kill variant with a high generic quality score and a verified-kill variant with a lower generic score; assert the quality profile rejects the no-kill option and validation rejects an edit shot without the configured minimum verified kills.
- [ ] **Step 7: Test the V5 profile and command surface.** Assert the profile uses 120–150 seconds, a configured 10–24 shot range, representative music policy, a new `preview_quality_v5.mp4` output, and the `all-v2-quality`, `render-preview-v2-quality`, and `verify-preview-v2-quality` commands. Assert the legacy long profile still points to V4’s output.
- [ ] **Step 8: Run the focused tests and observe RED.**

Run:

```powershell
D:/miniconda/python.exe -m pytest -q tests/test_condense_ranking.py tests/test_v2_timeline.py tests/test_v2_cli_report.py
```

Expected result: the new V5 behavior fails before production code is changed; the existing V2/V4 tests should otherwise remain green.

## Task 2: Repair verified-kill condensation and cache semantics

**Files:**

- Modify: `montage/config.py`
- Modify: `montage/condense.py`
- Modify: `main.py`
- Modify: `montage/ranking.py`
- Modify: `config.yaml`

- [ ] **Step 1: Add backward-compatible V5 config fields.** Add a disabled-by-default profile switch and bounded fields for verified-kill semantic/evidence thresholds, minimum verified kills per shot, density window/target, kill-count and density weights, maximum sequences per parent, sequence context, and end-clipped rapid-tail tolerance. Parse finite values with safe clamps; existing profiles retain their current behavior.
- [ ] **Step 2: Add strict verified-kill helpers.** Implement one shared helper that requires `kill`/`multikill`, sufficient semantic confidence, and corroborating kill-feed plus reward evidence. Add helpers for verified count and maximum count inside the configured density window so ranking and condensation use identical semantics.
- [ ] **Step 3: Fix rapid-window boundary handling.** Keep the legacy single-window behavior for default V2, but make the quality path tolerant of three-decimal rounding and a source-end-clipped tail when the cluster contains at least two verified kills and the configured minimum tail. Never accept an event outside the candidate or an overlapping segment.
- [ ] **Step 4: Add bounded multi-window extraction.** For quality-profile candidates longer than the saved-short threshold, enumerate compact verified-kill clusters, rank them by count, strength, tightness, and deterministic time tie-breakers, greedily retain non-overlapping windows, and cap the number of sequences per parent. Include approximately 1.25 seconds of setup/tail where available, trim dead reload/cover intervals, and preserve each cluster as one continuous source segment. Keep complete saved clips at or below the existing 30-second limit intact.
- [ ] **Step 5: Make each quality variant anchor a verified kill.** When a quality window has verified events, choose its primary/secondary anchors from those events instead of a generic combat-climax anchor. Set the configured short-clip prior on fresh and cached short-clip variants so old stale `0.42` artifacts cannot suppress user-saved clips.
- [ ] **Step 6: Add auditable quality score components.** Extend `score_variant` with verified-kill count, normalized kill-count score, rapid-kill count, and normalized density score. Apply the new weights only when the quality profile is active; preserve the existing one-time rapid-multikill bonus and its no-double-count beam behavior.
- [ ] **Step 7: Run the focused condensation/ranking/cache tests and the existing ranking/condense suites.** All new assertions must pass, and legacy default-profile behavior must remain green.

## Task 3: Add the isolated V5 profile, artifact paths, and CLI

**Files:**

- Modify: `montage/config.py`
- Modify: `main.py`
- Modify: `montage/beam_timeline.py`
- Modify: `montage/v2_report.py`
- Modify: `tests/test_v2_cli_report.py`
- Modify: `tests/test_v2_timeline.py`

- [ ] **Step 1: Add V5 artifact properties.** Add a profile-specific work directory and paths for V5 edit JSON, human timeline, sync report, timeline image, Markdown report, candidate JSON/CSV, payoff events, dedupe summary, and environment/baseline manifests as needed. Keep all paths below `work` and ensure `ensure_runtime_dirs` creates only work/output directories.
- [ ] **Step 2: Parameterize V2 analysis/render/report paths without changing legacy defaults.** Route a quality config to its V5 paths while the normal V2 and V4 configs continue using their existing paths. Include profile identity in cache stage/version and all behavior-changing parameters so V5 cannot accept a stale V4 aggregate artifact.
- [ ] **Step 3: Define the quality profile.** Use `preview_min_duration=120.0`, `preview_max_duration=150.0`, representative music selection, `v2_output_name=preview_quality_v5.mp4`, a 10–24 shot bound, at least one verified kill per shot, bounded multi-sequence extraction, quality kill-count/density weights, and a beam budget sufficient for the expanded candidate pool.
- [ ] **Step 4: Add explicit quality commands.** Register and dispatch `all-v2-quality`, `render-preview-v2-quality`, and `verify-preview-v2-quality`. Preserve the existing V2/V4 commands and output guards. Analysis commands must support dry-run/EDL generation without rendering.
- [ ] **Step 5: Make beam expansion and validation enforce the quality gate.** Reject variants with too few verified kills, include kill-density tie-breakers in deterministic ordering, use the representative music interval, and require the final EDL duration to be within 120–150 seconds and fully covered by the selected music window.
- [ ] **Step 6: Extend the report/timeline.** Keep the existing auditable fields and add verified-kill count/density, source sequence rationale, and explicit quality-profile/music-window reasons. Ensure the report identifies excluded/deduped variants and never describes a generic impact as a verified kill.
- [ ] **Step 7: Run all V5 CLI/timeline/report tests plus the complete V2 contract suite.** Confirm legacy tests remain green.

## Task 4: Generate and inspect the V5 EDL before render

**Files/artifacts:** Only below `D:\91\集锦\work` and the existing read-only inputs.

- [ ] **Step 1: Run `compileall` and the complete pytest suite.** Fix only failures caused by this implementation and keep the V4 behavior covered.
- [ ] **Step 2: Record a pre-run RAW manifest and preserve the prior V4 output/checksum state.** Do not touch any source file.
- [ ] **Step 3: Run the analysis-only quality command.**

```powershell
D:/miniconda/python.exe main.py all-v2-quality --dry-run
```

- [ ] **Step 4: Inspect the V5 EDL and candidate artifacts before rendering.** Confirm duration is 120–150 seconds; every shot has at least one strict verified kill; the first shot is not a no-kill opener; recent high-kill sources such as the 2026-09-01 21-14-19 and 2026-08-28 13-34-31 recordings are considered when their clusters pass verification; no training-range interval is selected; duplicate groups do not repeat; no shot contains an unexplained gap; and `music_out - music_in` covers the full target duration.
- [ ] **Step 5: Review the generated human timeline.** Confirm each placement has source range, score, verified-kill rationale, music section/energy, sync offset, and restrained hard-cut/match-cut transition information. If the EDL still admits a no-kill or low-density opener, fix the ranking/gate with a regression test before rendering.

## Task 5: Render, verify, and stop for human review

**Files/artifacts:** Create only `D:\91\集锦\output\preview_quality_v5.mp4` plus work-only render/debug artifacts.

- [ ] **Step 1: Render only the V5 preview.**

```powershell
D:/miniconda/python.exe main.py render-preview-v2-quality
```

- [ ] **Step 2: Run the quality verification command and an explicit FFmpeg 8 decode.** Check duration, 1920×1200 geometry, approximately 60 fps, monotonic timestamps, playable video/audio streams, and absence of full-montage outputs.
- [ ] **Step 3: Measure audio over the full duration and in multiple windows.** Confirm music is present through the final window, game transients remain measurable, the controlled game bus does not become excessively loud, and the final true peak/loudness remains within the configured safe bounds.
- [ ] **Step 4: Generate a work-only contact sheet or representative frame sample and inspect it.** Confirm no training-range/menu material, no template transition spam, and no obvious non-kill opening. Confirm compact consecutive kills remain visually complete and non-event reload/cover gaps are trimmed rather than cutting through the action.
- [ ] **Step 5: Compare post-run RAW manifest and preserved V4 output state.** Any difference in RAW is a hard failure. Confirm FFmpeg/ffprobe absolute paths and NVENC result are recorded in the V5 environment report and pipeline log.
- [ ] **Step 6: Write the final V5 report and stop.** Report video analysis, music structure, selected candidates, deduped candidates, music interval and reason, average/min/max shot duration, transition counts, sync offset mean/P95, audio strategy/measurements, encoder path, and the complete V5 output path. Do not render either full montage until the user has watched this result.

## Self-review checklist

- No implementation step writes below `raw`, and the pre/post manifest check is explicit.
- The two previously excluded training-range recordings remain excluded without deleting their files.
- The actual root causes are covered: boundary rounding, one-variant-per-long-source condensation, stale cached short-clip prior, and generic-score dominance over verified kills.
- Every quality-profile shot has a strict verified kill; generic combat-climax events cannot satisfy the gate.
- Multiple windows from one long saved clip are non-overlapping and are still subject to visual/audio dedupe.
- V5 music covers the complete edit, and V4/default V2 commands remain behavior-compatible.
- Tests are added before production changes, focused RED is observed, then focused and full GREEN runs are recorded.
- There are no placeholders, TODOs, unbounded candidate expansions, or hidden full-montage renders.
