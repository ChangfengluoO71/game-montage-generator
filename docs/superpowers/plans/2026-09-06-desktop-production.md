# Desktop first-cut production implementation plan

> Agentic workers: execute sequential milestones with regression checks and review. Implementation model: GPT-5.6-luna, as requested by the user.

**Goal:** A real desktop button runs detection, exports an editable project, and renders a playable MP4 from read-only RAW.

**Architecture:** Keep PySide6; a QThread worker calls a Qt-independent orchestration service. Resolve built-in Battlefield 6 rules to calibrated V6 skull-row profiles by exact source resolution; resolve custom rules to template matching. Adapt project clips to the existing v1 FFmpeg renderer; defer v2 ranking and beam editing.

**Spec:** `docs/WORKFLOW_HANDOFF.md` plus the user's M1-M5 acceptance requirements.

## Global constraints

- Worktree: v1.1-editorial-optimization. Integrate by non-force push HEAD:main and verify remote SHA.
- RAW `D:/91/集锦/raw` is read-only. Runtime files go to `D:/91/集锦/work` and `D:/91/集锦/output`; screenshots are not committed.
- Write source containing Chinese with apply_patch or UTF-8 file tools, never shell heredocs.
- For each coherent change run py_compile, full pytest, and offscreen MontageLab smoke. Audit all old field references after model changes.
- No-event, unsupported-resolution, missing-template, decoder, and renderer failures must surface truthfully. Never fabricate detections to produce a video.
- Preserve legacy engine workflows and all desktop rules, sample paths, IDs, metadata, editing and audio settings through round trips.

## M1: Contract adapter

Files: montage/workflow.py, montage/desktop_app.py, tests/test_workflow.py, tests/test_desktop_app.py.

- [ ] Test actual ProfileWizard.save_profile output for Apex imports through MontageWorkflow.import_json, including two rules appended to the same game.
- [ ] Add an explicit multi-rule engine representation while retaining the legacy detector interface; implement desktop import/export and engine round trips without losing rule metadata.
- [ ] Validate finite durations, normalized ROI and thresholds; preserve engine example compatibility and reject malformed configurations with useful messages.
- [ ] Correct same-game append behavior and unique rule identifiers; audit detector/profile old-field references.
- [ ] Run all required checks, document output and commit.

## M2: Background scanning

Files: new montage/generation.py and montage/generation_worker.py, montage/kill_truth/scanner.py, montage/desktop_app.py, new tests/test_generation.py and worker tests.

- [ ] Define request/result/event data for selected workflow, raw folder, optional source subset, music, work/output paths and render options. Validate all write destinations outside RAW, including selected source folder.
- [ ] Index current requested sources, avoiding unrelated cached indexes. Match calibrated profiles by resolution and verify templates exist.
- [ ] Implement custom normalized-ROI template matching with temporal deduplication and rule provenance; handle unicode paths, invalid/constant/oversize templates and decoder failures.
- [ ] Add throttled frame-level progress to V6 and template scan, forwarding log/error/progress signals through QThread. Persist source ledger and events for every run.
- [ ] Wire real start, completion/failure and safe window-close lifecycle; test UI responsiveness and worker errors without blocking modal test dialogs.
- [ ] Run all required checks, document output and commit.

## M3: Editable project

Files: montage/generation.py or focused project builder, tests/test_generation.py, tests/test_workflow_project.py.

- [ ] Convert event windows to bounded TimelineClip ranges, merge source-local nearby windows, apply the documented bridge rule, and never merge across sources.
- [ ] Export workflow snapshot, events, source ledger and EditableProject with contiguous timeline positions, music_source and actual render_settings.
- [ ] Test event boundary clipping, merge/bridge behavior, multiple sources and rules, empty events and project import round trip.
- [ ] Run required checks, document output and commit.

## M4: Real production render

Files: montage/generation.py or new focused project renderer, montage/ffmpeg_renderer.py as necessary, production verification script and regression tests.

- [ ] Map TimelineClip to v1 renderer contracts, preserve selected music and editing durations, support absent music/audio, use existing encoder discovery/fallback, clamp fade to available duration.
- [ ] Report real render stage/segment progress and atomic output success. Do not claim decode success from metadata alone.
- [ ] Run one complete real RAW video through fresh scan, events, project and MP4; select a short calibrated recording for bounded production. Save ffprobe JSON and full ffmpeg decode-to-null verification.
- [ ] Record source size/mtime before and after, output paths, counts, duration, codec and commands; run required checks and commit code/evidence report only.

## M5: UI completion and final acceptance

Files: montage/desktop_app.py, tests/test_desktop_app.py, production evidence docs.

- [ ] Bind selected game explicitly, preserve selection across profile refresh and restart, persist music and use a clean filesystem path separate from file-count display.
- [ ] Disable duplicate generation and conflicting edits while running; display progress, log, errors, result card and working open-output-directory action. Implement actual workflow import/export and project export where exposed.
- [ ] Exercise actual button-driven worker on the real single-source run and capture progress/result screenshots via Qt grab into work evidence directory.
- [ ] Run py_compile, full pytest and offscreen smoke; review final diff and fix regressions.
- [ ] Commit all intended files, non-force push HEAD:main, verify git ls-remote origin refs/heads/main equals local HEAD.

## Assessment notes

The checked-in desktop source differs from the handoff: save_profile currently overwrites same-game rules; import_profile only shows a message. QSettings native Windows fileName may be a registry path, so application data storage must be checked against the actual local Apex profile location. The v1 renderer currently requires music; optional-music support must be explicit and tested. V6 scan_source returns PARTIAL_ERROR instead of raising, and has no progress callback yet; orchestration must not silently accept incomplete scans. Calibrated detection is evidence of HUD feedback, not a new claim of benchmark accuracy.
