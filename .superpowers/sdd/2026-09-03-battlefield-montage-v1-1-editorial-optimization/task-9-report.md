# Task 9 report: V2 visualization, comparison report, CLI, and verification

## Files

- `montage/timeline_visualization.py`: music energy/beat/phrase markers, shot spans, anchors, cuts, event labels, and confidence plot.
- `montage/v2_report.py`: auditable V1/V2 ranges, shot/source/anchor/sync/transition diagnostics, dedupe, rapid-multikill, audio/toolchain, and caveat reporting.
- `main.py`: analysis-only V2 orchestration, artifact gates, V2-only render path, strict verification, logging, and CLI commands.
- `montage/config.py`: versioned environment and baseline-manifest artifact paths.
- `README.md`: V2 dry-run/render/verify usage and safety boundaries.
- `tests/test_v2_cli_report.py`: focused parser, report, plot, and model contract tests.

## Results

- Focused Task 9 and related checks: passed.
- Full regression suite: 164 passed (the prior 159 count was stale).
- `compileall -q .`: passed.
- `git diff --check`: passed.
- No real media, render, or full-output command was run.

## Replacement-fix coverage

- Normalized all music evidence to the zero-based EDL edit clock and made dense-shot labels compact/staggered.
- Added explicit V1/V2/delta comparison records for shot, duration, source, anchor, sync, transition, stationarity, downtime, repetition, recent-source, rejection, and rapid-multikill diagnostics while retaining V2 top-level keys.
- Passed cached or freshly computed per-source audio evidence into payoff detection.
- Reloaded cached payoff events and dedupe diagnostics on variant-cache hits without overwriting them with empty values.
- Strengthened V2 source preflight/verification to probe each distinct regular source with selected ffprobe, require finite duration and audio/video streams, validate every range, and retain RAW/baseline integrity checks.

## Commit hash

Implementation commit hash before the report-only amendment: `43d2f0fb54e6b28c7e3ca054ba764a3047c55c1c`.

## Caveats

- The analysis pipeline expects the existing V1 baseline/music/analysis artifacts and does not synthesize media fixtures.
- Verification requires a real V2 output and selected FFmpeg/ffprobe; it was not exercised against media in this task.
- Report metrics are explicitly diagnostic evidence; phrase/section semantics remain heuristic.

## Immutability confirmation

No media or RAW files were changed. No files below `raw` were created, deleted, moved, renamed, overwritten, or metadata-modified. The V1 baseline was not changed, and no Fast Montage or Full Highlights output was created.
