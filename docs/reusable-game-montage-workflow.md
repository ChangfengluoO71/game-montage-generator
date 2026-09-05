# Reusable Game Montage Workflow

`montage/workflow.py` defines the game-agnostic configuration contract for a future Game Montage Generator.

## Capabilities

- Game-specific detector configuration without hard-coding Battlefield logic.
- Normalized ROI coordinates that map across native resolutions.
- Uploadable detector templates plus full-frame positive and negative samples.
- Adjustable event window rules: pre-roll, post-roll, merge gap, long-gap bridge, fade tail.
- Export/import as UTF-8 JSON for sharing or backup.
- MP4/MKV audio extraction through the first audio stream, without modifying source media.
- Stable audio/output settings for gameplay-first mixing.

## Project interchange contract

The future frontend should save an editable project rather than only an MP4:

```text
project.json
source-ledger.json
event-index.json
candidate-manifest.json
timeline.json
render-settings.json
outputs/
```

`timeline.json` should retain source path, source in/out, timeline in/out, event ids, AI reason, and user review state. This lets users continue editing after AI generation and export a project or workflow profile separately from media.

## Resolution strategy

- Use normalized ROIs and native-resolution scanning.
- Scale templates by source height and permit local scale search.
- Keep 16:10 sources distortion-free.
- For 16:9 output, scale to target height and center-crop to the project canvas.
- Store profile id, source resolution, calibration state, and output policy in the ledger.

## Safety boundaries

RAW is read-only. Configuration exports contain rules, templates, and sample references, not the user's full RAW library. Duplicate candidates remain reviewable rather than being silently deleted.

The existing Battlefield V6 pipeline remains the first adapter; other games add detector profiles and calibration data, not a new editing engine.
