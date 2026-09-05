# Frontend Game Montage Generator Plan

## Goal

Turn the frozen local workflow into a reusable local-first application: users import RAW media and music, select or create a game detector profile, run AI candidate generation, continue editing on a timeline, and export both video and shareable configuration.

## Phase 1 — CLI/core foundation

- Keep `MontageWorkflow` and `EditableProject` as the interchange contracts.
- Add explicit adapters: detector adapter, media preflight adapter, event-window builder, renderer adapter, music-source adapter.
- Add project versioning and migration functions.
- Add a source ledger with `unused`, `partially_used`, `used`, `duplicate_review`, and `excluded` states.
- Add MP4 audio extraction metadata and cache keys.
- Add golden fixtures for normalized ROI and event-window merging.

## Phase 2 — Local API/worker

- Expose scan, candidate generation, project save/load, audio extraction, render, and cancel operations through a local API.
- Run FFmpeg/OpenCV in a worker process with progress events and structured errors.
- Never upload or mutate RAW by default.
- Use job IDs and resumable artifacts under a project directory.

## Phase 3 — Frontend MVP

- Project creation and RAW folder import.
- Music import with MP4 audio-track detection.
- Game profile selection and profile wizard.
- Upload template crops and full-frame positive/negative samples.
- Normalized ROI editor with resolution previews.
- Timing/audio/output controls.
- Candidate review list and video preview.
- Editable timeline: trim, reorder, remove, restore, merge, split, mark duplicate.
- Render progress and output history.

## Phase 4 — Profile sharing

Export/import a ZIP containing:

```text
game.json
detector.json
editing_rules.json
audio_output.json
profiles/
templates/
samples/
README.md
```

Do not include RAW media. Validate schema, paths, and compatibility on import.

## Phase 5 — Advanced review

- Cross-RAW temporal perceptual duplicate candidates.
- Side-by-side review with source offsets.
- Manual decisions stored in the project timeline.
- Profile calibration report with positive/negative sample metrics.
- Optional beat-aware placement after event selection, never instead of event evidence.

## Product boundary

The AI produces an editable project and candidate timeline, not an irreversible final MP4. Users remain able to continue editing after generation. Game-specific code is limited to detector adapters and profile data; timing, audio, canvas normalization, project persistence, and rendering remain reusable.

## Immediate next implementation slice

1. Add a project directory serializer and schema migration hooks.
2. Add a detector adapter protocol and Battlefield skull-row adapter.
3. Add event-window builder driven by `EditRules`.
4. Add a local worker command that accepts a workflow JSON and project JSON.
5. Build a thin web UI over the worker API after those contracts stabilize.
