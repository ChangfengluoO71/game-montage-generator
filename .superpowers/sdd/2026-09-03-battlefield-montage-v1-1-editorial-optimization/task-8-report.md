# Task 8 report

## Files

- `montage/v2_renderer.py`
- `montage/audio_mix.py`
- `montage/config.py`
- `tests/test_v2_renderer.py`

## Tests and results

- `D:/miniconda/python.exe -m pytest -q tests/test_v2_renderer.py`: 13 passed after the review fix.
- Focused regression suites: 94 passed.
- `D:/miniconda/python.exe -m pytest -q`: 146 passed, 2 pre-existing `audioread` Python 3.13 deprecation warnings.
- `D:/miniconda/python.exe -m compileall -q montage tests`: passed.
- `git diff --check`: passed.

## Audio strategy

V2 keeps the game bus present and audible, reads music once from `music_in` for the EDL duration, and applies smooth sidechain compression with configurable attack/release. Safe compatible edges may use one 100–250 ms audio-only J/L overlap; impact edges use direct boundaries. Audio is normalized and resampled to 48 kHz AAC, with duration trimming/padding.

## Geometry and encoder behavior

Each source segment is trimmed under `work/cache/render_v2/<unique-run>`, fitted with bounded `min(iw,1920)` / `min(ih,1200)` scaling and padding, and encoded with source-cadence passthrough. `h264_nvenc` is selected only when the tested `Toolchain` reports runtime support; otherwise the configured `libx264` fallback is used. Final video concatenation is hard-cut-only; no video xfade or visual effects are generated.

## Safety and media status

EDL validation, source/range/path checks, RAW destination rejection, unique work-temporary output, nonzero-size verification, and V2-only replacement are enforced. The V1 baseline remains untouched. No RAW files, generated media, or real-media renders were changed or run.

## Review fixes

- Before any V2 segment render, every distinct EDL RAW source is preflighted with the selected toolchain `ffprobe`. Sources must be regular files below `raw`, contain video and audio streams, report a finite positive duration, and contain every requested `source_out` within the probed duration plus a 20 ms safety tolerance. A failed preflight stops before segment generation.
- `compile_v2_segment_argv` accepts only strict descendants of `work_dir`. `compile_v2_final_argv` accepts a strict `work_dir` descendant or exactly the configured V2 output path. Both public APIs reject RAW, the immutable baseline, the output directory itself, and arbitrary external destinations.
- Added mocked-ffprobe and public-argv destination-policy regression coverage; removed the unused renderer cache import.

## Commits

Original Task 8 commit: `81d5c75f8ea9597c98ca718017ee8ee1bd027fac`

Review-fix commit: recorded by the implementing changeset.
