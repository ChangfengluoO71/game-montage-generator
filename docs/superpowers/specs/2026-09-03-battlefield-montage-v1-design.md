# Battlefield Montage V1 Design Specification

**Date:** 2026-09-03  
**Status:** Approved for implementation  
**Milestone:** Produce and verify one real `preview_60s.mp4`, then stop.

## Goal

Build a Windows-local, non-destructive Battlefield highlight pipeline that indexes the existing recordings, treats manually saved short clips as high-priority candidates, extracts additional candidates from the three long recordings, removes duplicate events, analyzes the test song structurally, creates a beat-aware but gameplay-first 45–60 second edit decision list, and renders:

`D:\91\集锦\output\preview_60s.mp4`

The first milestone does not render the complete Fast Montage or Full Highlights outputs.

## User constraints that are part of the design

- `D:\91\集锦\raw` is permanently read-only. The program must not delete, move, rename, overwrite, modify metadata, or encode in place there.
- All generated data goes under `D:\91\集锦\work`; all finished video goes under `D:\91\集锦\output`.
- All Windows subprocesses use argument arrays with `shell=False` (or the platform default equivalent). Unicode, Japanese, Chinese, full-width punctuation, spaces, and brackets must remain valid path characters.
- The selected FFmpeg executable is discovered once at run start, runtime-tested, recorded in `work/analysis/environment.json` and `work/logs/pipeline.log`, and reused for the entire run. Its sibling `ffprobe` is preferred.
- The currently working FFmpeg 8.0 installation is preferred over the broken PATH-first FFmpeg 4.3.1 installation. If no NVENC runtime passes, the run falls back to CPU and records the reason.
- Short clips at or below 90 seconds are treated as human-selected material. Their priority is high, but obvious black/menu/loading/static tails can still be trimmed. A 15–30 second action is not broken into one-second shots merely to follow beats.
- Only long recordings are required to receive proxy generation and primary candidate extraction. Short clips are analyzed by low-rate direct decode unless a cache already contains a proxy.
- Gameplay continuity outranks music structure; music structure outranks exact beat alignment; transition effects are last.
- Hard cuts are the default. Match cuts, J/L audio overlap, rare crossfades, and chapter-only dip-to-black are optional. Glitch, RGB split, spin, zoom transition, shake, flash spam, and template speed ramps are out of scope.
- The first preview is generated from a representative song section containing a build-up, section transition, and high-energy material when the analysis can identify one. The chosen `music_in` and `music_out` and the reason are reported.
- The preview must preserve the audited 1920×1200 16:10 composition as far as possible, avoid upscaling and interpolation, and keep game audio clearly audible alongside music.

## Scope and stop condition

The first run includes these stages:

```text
Environment
  -> Index
  -> Music Analysis
  -> Video Analysis
  -> Candidate Ranking
  -> Dedupe
  -> 60-second EDL
  -> preview_60s.mp4
```

After verifying the preview, the program stops. `Battlefield_Fast_Montage.mp4` and `Battlefield_Full_Highlights.mp4` are not rendered in this milestone, even though their renderer and EDL builders may exist so the boundary is explicit and future work can reuse the same artifacts.

## Architecture

The project is a small Python package with single-responsibility modules. Python performs metadata collection, signal analysis, candidate selection, dedupe, timeline construction, and manifest generation. FFmpeg performs decoding, thumbnail/preview extraction, source-accurate shot rendering, audio mixing, and final encoding. No cloud service, Video LLM, MoviePy, or online API is used.

### Runtime context and toolchain pinning

`montage.toolchain` resolves all `ffmpeg` candidates returned by Windows `where.exe`, plus configured paths. For each candidate it obtains the version and checks for the required encoders. It runs a harmless 1280×720, 0.5-second null-output encode to test `h264_nvenc` and `hevc_nvenc`; the first viable candidate is selected by highest parsed version, with FFmpeg 8.0 winning on the audited machine. The selected executable and its sibling `ffprobe` are stored in an immutable `RunContext` passed to all stages.

The subprocess wrapper accepts `Sequence[str]`, never a shell command string, and logs a redacted argv plus return code. Each failed media probe is recorded and skipped instead of aborting the complete run.

### Directory layout

```text
battlefield-montage/
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
├── montage/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── toolchain.py
│   ├── cache.py
│   ├── media_index.py
│   ├── proxy.py
│   ├── video_analysis.py
│   ├── audio_analysis.py
│   ├── music_analysis.py
│   ├── candidate.py
│   ├── dedupe.py
│   ├── ranking.py
│   ├── timeline.py
│   ├── transitions.py
│   ├── audio_mix.py
│   ├── ffmpeg_renderer.py
│   └── review.py
└── tests/
```

The configured runtime directories are:

```text
work/cache
work/proxy
work/analysis
work/analysis/music
work/analysis/preview
work/highlights
work/review
work/logs
output
```

## Data flow and artifacts

### Environment audit

`environment.json` records Python version, OS, GPU query, all discovered FFmpeg candidates, selected FFmpeg/ffprobe absolute paths, versions, encoder availability, NVENC runtime results, and the selection reason. It is written under `work/analysis` and mirrored in the run log.

### Media index

`media_index.py` recursively recognizes `.mp4`, `.mkv`, `.mov`, and `.webm`. The current corpus is 44 MP4 files. Each successful record contains:

```text
file_path, file_name, file_size, duration, width, height, fps,
codec, bitrate, audio_codec, audio_channels, audio_sample_rate,
creation_time, category, fingerprint
```

The fingerprint includes normalized absolute path, file size, and mtime. Results are saved to `work/analysis/media_index.json` and `media_index.csv`. Probe errors are saved in the JSON error list and the log.

### Proxy policy

Only files longer than 300 seconds receive video proxies. A proxy uses a 960×540 bounding box, keeps the source presentation timestamps and frame cadence with passthrough mode, and includes a mono analysis audio track. Proxy paths are cache-keyed by the source fingerprint. Short clips are decoded at low sample rate directly from the source so they are not needlessly re-encoded.

The proxy is never used as the final video source. All selected shots use source paths and source timestamps from the media index.

### Video and audio analysis

Video frames are decoded through the selected FFmpeg process into a low-resolution grayscale pipe at the configured analysis rate. The analysis computes normalized frame difference, a cheap motion estimate, brightness/black-screen ratio, entropy, scene-change spikes, and a combined visual activity curve. Audio is decoded to a cached analysis WAV and produces RMS, peak, spectral flux, onset strength/density, and transient activity. The combined curve is smoothed so isolated peaks do not become isolated highlights.

Long-recording candidates are contiguous activity regions with a minimum 3-second combat core and a maximum 30-second core, merging nearby peaks separated by up to 3 seconds. Configured pre-roll and post-roll are applied after merging. Short clips normally remain one candidate, with only an evidence-based trim of leading/trailing non-content frames.

The video analysis writes per-source cached JSON plus `activity_analysis.png`, showing component curves, the combined curve, candidate cores, and selected windows.

### Music analysis

The fixed FLAC path is decoded by `librosa` at analysis rate for DSP only; the source file is not changed. The analysis creates:

- `work/analysis/music/beat_map.json`: beat timestamps, estimated tempo, bar/downbeat attempts, strong beats, ordinary beats, onsets, strength, and confidence values.
- `work/analysis/music/music_structure.json`: section boundaries, low/medium/high energy regions, confidence, and the selected preview section.
- `work/analysis/music/energy_curve.csv`: time-binned RMS/onset/novelty/normalized energy.
- `work/analysis/music/music_analysis.png`: waveform, RMS, onset envelope, beat markers, and section boundaries.

The first structure pass uses beat periodicity, phrase-sized grouping, novelty peaks, and robust energy quantiles. It may label sections as `low_energy`, `medium_energy`, and `high_energy` without pretending to know exact verse/chorus semantics. Every inferred section and downbeat carries a confidence value; low confidence is reported rather than fabricated into precision.

### Candidate scoring and dedupe

Candidate records contain the requested fields plus component diagnostics:

```text
candidate_id, source_file, source_start, source_end, duration,
human_selection_score, audio_score, motion_score, visual_score,
continuity_score, combat_intensity, uniqueness, duplicate_group,
fingerprint, final_score, rationale
```

The configurable default score is:

```text
0.30 * human_selection_prior
+ 0.20 * combat_intensity
+ 0.10 * motion
+ 0.10 * audio_activity
+ 0.10 * visual_activity
+ 0.10 * continuity
+ 0.10 * uniqueness
```

The implementation keeps both the requested output field names and the more explicit component names so later tuning is possible without changing the EDL contract.

Candidate fingerprints use low-resolution grayscale dHash/pHash samples. Similar candidate sequences are grouped with a duration-aware sliding comparison so a saved short clip can match a contained section of a long recording. Within a duplicate group, Fast prefers the human-selected short clip; Full may choose the longer-context candidate. One group can be used at most once in a single edit.

Results are saved to `work/analysis/highlight_candidates.json` and `.csv`. `dedupe_summary.json` records groups, similarity values, and the selected representative for the preview.

### Preview EDL

`timeline.py` makes `work/analysis/preview_edit.json` before any render. Every shot contains at least:

```json
{
  "source": "D:\\91\\集锦\\raw\\clip.mp4",
  "source_in": 14.3,
  "source_out": 19.7,
  "duration": 5.4,
  "candidate_score": 0.82,
  "duplicate_group": "dg-003",
  "timeline_in": 0.0,
  "timeline_out": 5.4,
  "transition": "hard_cut",
  "music_target": 8.01,
  "music_event_type": "strong_beat",
  "sync_offset": 0.041,
  "rationale": "Medium-intensity opening establishes combat before the build-up."
}
```

The preview selection chooses one representative 45–60-second music window that includes a detected build-up/section boundary and high-energy region. It assigns shots by narrative energy rather than sorting final score descending: medium material opens, activity rises toward the chosen chorus/high-energy section, and the strongest available candidate closes the preview. For the first preview, candidates up to 20 seconds are preferred when enough coherent saved clips exist; longer coherent shots remain eligible when they are the better continuity choice. A normal beat can align an action inside a shot but cannot force a cut. A continuous multi-kill remains intact even when it crosses several beats.

`preview_timeline.txt` is generated from the same EDL and is intended for manual inspection. It includes the source range, score, music section/energy, sync event and offset, transition, and rationale for every shot.

## Rendering and audio

The renderer uses a `Timeline -> Shot -> Transition -> AudioAutomation` intermediate model and compiles one manageable FFmpeg argv per shot. Each shot is source-trimmed, normalized into a 1920×1200 fit-with-padding frame, and encoded at the source cadence without interpolation. Padding is used instead of destructive cropping so HUD and composition remain intact. The preview output uses H.264; `h264_nvenc` is selected when the pinned runtime test succeeded, otherwise `libx264` is logged and used.

The game track remains present. A smooth, low-ratio sidechain/automation stage ducks music against the game track with approximately 20–80ms attack and 250–800ms release. Candidate activity controls a second broad automation level: ordinary shots keep music around a reduced mix level, while combat/explosion regions reduce music an additional few dB and allow game audio to lead briefly. The final mix is limited with `loudnorm` toward approximately -14 LUFS and -1 dBTP true peak. AAC 48kHz is used for the preview.

Segments are concatenated through a generated concat manifest rather than one unbounded Windows command line. Temporary segment files are placed under `work/cache/render/<run-id>`; final output is written atomically to `output/preview_60s.mp4` only after FFmpeg exits successfully. No output path is ever resolved inside `raw`.

## CLI

The first implementation exposes:

```text
python main.py index
python main.py analyze-music
python main.py analyze-video
python main.py candidates
python main.py review
python main.py render-preview
python main.py render-fast
python main.py render-full
python main.py all
```

`all --dry-run` executes audit, indexing, analysis, candidate generation, dedupe, review, and EDL generation, but refuses all final video rendering. `render-preview` is the only render command used in this milestone. Full output commands exist as guarded future paths and are not called by the first run.

## Cache and failure behavior

Expensive artifacts are keyed by a JSON fingerprint containing absolute path, size, mtime, stage name, and relevant configuration/version. Unchanged sources do not repeat ffprobe, proxy, video analysis, audio extraction, music analysis, or candidate fingerprints. Changed configuration invalidates only the affected stage. Atomic JSON/CSV/image writes prevent partial cache files.

All stages log one of `INDEX`, `PROXY`, `ANALYZE`, `DEDUPE`, `MUSIC`, `EDIT`, or `RENDER` with timestamps. A corrupt media file produces a structured error and continues. A missing required tool or music file is a run-level error with a clear remediation message. A failed NVENC test is not fatal when CPU encoding is available.

## Testing and acceptance

Tests are written before production code. Unit tests cover path safety, toolchain selection, fingerprint invalidation, media categorization, feature normalization, candidate window merging, dedupe preference, music edit-point hierarchy, narrative ordering, EDL schema, and renderer argv safety. Synthetic media fixtures are generated outside `raw` for integration tests.

The real smoke set is three short clips and the three long recordings only as needed for proxy/analysis cost. Before the preview render, the following must exist:

```text
work/analysis/environment.json
work/analysis/media_index.json
work/analysis/media_index.csv
work/analysis/music/beat_map.json
work/analysis/music/music_structure.json
work/analysis/music/music_analysis.png
work/analysis/highlight_candidates.json
work/analysis/highlight_candidates.csv
work/analysis/preview_edit.json
work/analysis/preview_timeline.txt
```

The preview verification checks that `raw` fingerprints, names, mtimes, and file counts are unchanged; the output probes successfully; audio and video streams exist; duration is 45–60 seconds; resolution is 1920×1200; frame rate remains approximately 60fps; no selected shot is outside source bounds; and the report contains the requested analysis, dedupe, timing, transition, mix, and toolchain fields.

The milestone is complete only when the verified preview exists at the exact requested path. At that point the process stops and waits for human viewing.
