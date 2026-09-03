# Battlefield Montage V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a non-destructive Windows Battlefield highlight pipeline that analyzes the audited corpus and produces one verified, music-aware `D:\91\集锦\output\preview_60s.mp4`, then stops.

**Architecture:** A typed Python orchestration layer owns configuration, cache keys, FFmpeg tool selection, DSP/video features, candidate ranking, dedupe, and an explicit EDL. FFmpeg is invoked through argument arrays for decoding and segmented source rendering; the selected FFmpeg 8.0 executable and sibling ffprobe are pinned for the whole run. Long videos get proxies, short clips are analyzed directly and retain their human-selection prior.

**Tech Stack:** Python 3.13, standard library subprocess/pathlib/dataclasses/json/csv/logging, PyYAML, NumPy, SciPy, librosa, SoundFile, Matplotlib, pytest, FFmpeg 8.0 with NVENC when runtime-tested.

**Spec:** `docs/superpowers/specs/2026-09-03-battlefield-montage-v1-design.md`

## Global Constraints

- `D:\91\集锦\raw` is permanently read-only and must never be deleted, moved, renamed, overwritten, metadata-modified, or encoded in place.
- All generated data goes under `D:\91\集锦\work`; all finished video goes under `D:\91\集锦\output`.
- All Windows subprocesses use argument arrays with `shell=False`; Unicode and bracketed paths are passed as individual arguments.
- The selected FFmpeg executable is discovered once, runtime-tested, recorded in `work/analysis/environment.json` and `work/logs/pipeline.log`, and reused for the entire run; its sibling ffprobe is preferred.
- Prefer the tested FFmpeg 8.0 installation over the PATH-first FFmpeg 4.3.1 installation; fall back to CPU encoding only after recording the NVENC failure.
- Clips at or below 90 seconds get a high human-selection prior and are kept as coherent candidates; only obvious non-content head/tail may be trimmed.
- Long recordings get primary activity/audio/motion extraction and proxy generation; short clips are directly low-sample-rate analyzed without mandatory proxy creation.
- Gameplay continuity outranks music structure; music structure outranks exact beat alignment; transition effects are last.
- Hard cuts are the default; no automatic glitch, RGB split, spin, zoom transition, shake, flash spam, or template speed ramp.
- Do not render the complete Fast Montage or Full Highlights during this milestone. Stop after the verified 45–60 second preview.
- Preview output stays 1920×1200, approximately source cadence, with no upscale or frame interpolation and no destructive 16:9 crop.

---

## File Map

Create these files in the new project unless a later task explicitly modifies them:

```text
main.py
config.yaml
requirements.txt
README.md
montage/__init__.py
montage/config.py
montage/models.py
montage/toolchain.py
montage/cache.py
montage/media_index.py
montage/proxy.py
montage/audio_analysis.py
montage/video_analysis.py
montage/music_analysis.py
montage/candidate.py
montage/dedupe.py
montage/ranking.py
montage/transitions.py
montage/timeline.py
montage/audio_mix.py
montage/ffmpeg_renderer.py
montage/review.py
tests/conftest.py
tests/test_config_cache.py
tests/test_toolchain.py
tests/test_media_index.py
tests/test_video_analysis.py
tests/test_music_analysis.py
tests/test_candidates_dedupe.py
tests/test_timeline.py
tests/test_renderer.py
tests/test_cli.py
```

`models.py` owns serializable data contracts; `config.py` owns YAML/defaults; `toolchain.py` owns executable discovery and safe subprocesses; `cache.py` owns fingerprints and atomic artifacts; the analysis modules never render final media; the timeline modules never call FFmpeg; `ffmpeg_renderer.py` is the only final render boundary; `main.py` is orchestration only.

## Task 1: Bootstrap the project, configuration, models, cache, and read-only guard

**Files:**
- Create: `config.yaml`
- Create: `requirements.txt`
- Create: `montage/__init__.py`
- Create: `montage/models.py`
- Create: `montage/config.py`
- Create: `montage/cache.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config_cache.py`

**Interfaces:**
- `load_config(path: Path) -> PipelineConfig`
- `ensure_runtime_dirs(config: PipelineConfig) -> None`
- `file_fingerprint(path: Path) -> dict[str, object]`
- `cache_key(fingerprint: dict[str, object], stage: str, parameters: dict[str, object]) -> str`
- `atomic_write_json(path: Path, value: object) -> None`
- `is_within(path: Path, root: Path) -> bool`
- `assert_source_read_only(path: Path, raw_dir: Path) -> None`
- Dataclasses `PipelineConfig`, `MediaRecord`, `VideoAnalysis`, `MusicAnalysis`, `Candidate`, `EditShot`, and `EditDecisionList` must each expose `to_dict()` methods used by JSON writers.

- [ ] **Step 1: Write failing tests for configuration, cache invalidation, atomic writes, and raw safety**

```python
def test_config_resolves_audited_unicode_paths(tmp_path):
    config = load_config(Path("config.yaml"))
    assert config.raw_dir == Path(r"D:\91\集锦\raw")
    assert config.music_file.name == "01. 決意の唄.flac"
    assert config.output_width == 1920
    assert config.output_height == 1200

def test_cache_key_changes_when_mtime_changes():
    first = cache_key({"absolute_path": "C:/x.mp4", "size": 10, "mtime": 1.0}, "probe", {})
    second = cache_key({"absolute_path": "C:/x.mp4", "size": 10, "mtime": 2.0}, "probe", {})
    assert first != second

def test_raw_path_is_allowed_only_as_a_source(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "片段 [01].mp4"
    source.write_bytes(b"source")
    assert_source_read_only(source, raw)
    with pytest.raises(ValueError):
        assert_source_read_only(raw / "nested" / "generated.mp4", raw)

def test_atomic_json_write_replaces_complete_file(tmp_path):
    target = tmp_path / "analysis.json"
    atomic_write_json(target, {"ok": True})
    assert json.loads(target.read_text(encoding="utf-8"))["ok"] is True
    assert not list(tmp_path.glob("*.tmp"))
```

- [ ] **Step 2: Run the focused tests and verify they fail because the project modules do not exist**

Run: `python -m pytest tests/test_config_cache.py -q`  
Expected: collection failure naming the missing `montage` modules or symbols.

- [ ] **Step 3: Write the minimal models/config/cache implementation**

Use YAML defaults for the audited paths, `video_extensions`, `long_clip_threshold: 300`, `short_clip_threshold: 90`, `output_width: 1920`, `output_height: 1200`, `preview_min_duration: 45`, `preview_max_duration: 60`, and `preview_preferred_shot_max_duration: 20`. Resolve all configured paths with `Path(...).expanduser()` without touching source files. `atomic_write_json` writes beside the target with a unique `.tmp` name, flushes, replaces the target, and cleans the temporary file on error. `assert_source_read_only` accepts an existing path under `raw` only for reads and rejects any generated target under `raw`.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `python -m pytest tests/test_config_cache.py -q`  
Expected: all tests pass with no warnings.

- [ ] **Step 5: Add the dependency and user-facing configuration documentation**

`requirements.txt` must contain `numpy`, `scipy`, `librosa`, `soundfile`, `matplotlib`, `PyYAML`, and `pytest`. `config.yaml` must expose `raw_dir`, `work_dir`, `output_dir`, `music_file`, `proxy_resolution`, `short_clip_threshold`, `long_clip_threshold`, `pre_roll`, `post_roll`, `analysis_fps`, `highlight_min_duration`, `highlight_max_duration`, `weights`, `fast_montage_target`, `full_highlight_target`, `preview_min_duration`, `preview_max_duration`, `preview_preferred_shot_max_duration`, `nvenc`, and `audio_mix` without hard-coding weights in ranking code.

## Task 2: Pin a working FFmpeg/ffprobe toolchain and build the media index

**Files:**
- Create: `montage/toolchain.py`
- Create: `montage/media_index.py`
- Create: `tests/test_toolchain.py`
- Create: `tests/test_media_index.py`

**Interfaces:**
- `discover_toolchain(config: PipelineConfig, runner: Callable[..., CompletedProcess] | None = None) -> Toolchain`
- `run_command(argv: Sequence[str], *, capture_output: bool = True, check: bool = True, logger: logging.Logger | None = None) -> CompletedProcess`
- `build_media_index(config: PipelineConfig, toolchain: Toolchain, logger: logging.Logger) -> list[MediaRecord]`
- `write_media_index(records: Sequence[MediaRecord], json_path: Path, csv_path: Path) -> None`
- `probe_media(path: Path, toolchain: Toolchain) -> MediaRecord`

- [ ] **Step 1: Write failing tests for executable selection, argv safety, media categories, and CSV output**

```python
def test_toolchain_prefers_highest_viable_version(monkeypatch, config):
    candidates = [Path(r"D:\old\ffmpeg.exe"), Path(r"D:\new\ffmpeg.exe")]
    monkeypatch.setattr("montage.toolchain.find_ffmpeg_candidates", lambda _: candidates)
    monkeypatch.setattr("montage.toolchain.probe_candidate", lambda path, *_: fake_tool(path, "8.0" if "new" in str(path) else "4.3.1", path.name == "ffmpeg.exe"))
    selected = discover_toolchain(config)
    assert selected.ffmpeg == candidates[1]
    assert selected.ffprobe == Path(r"D:\new\ffprobe.exe")

def test_command_keeps_unicode_path_as_one_argument(monkeypatch):
    seen = {}
    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr("montage.toolchain.subprocess.run", fake_run)
    run_command(["ffprobe", "-show_format", r"D:\raw\中文 [a].mp4"])
    assert seen["argv"][-1] == r"D:\raw\中文 [a].mp4"
    assert seen["argv"] is not None

def test_media_index_classifies_short_and_long(tmp_path, fake_toolchain, monkeypatch):
    short = tmp_path / "短 [15].mp4"; short.write_bytes(b"x")
    long = tmp_path / "long.mp4"; long.write_bytes(b"y")
    monkeypatch.setattr("montage.media_index.probe_media", lambda path, _: media_record(path, 15 if path == short else 301))
    records = build_media_index(config_for(tmp_path), fake_toolchain, logging.getLogger("test"))
    assert {record.category for record in records} == {"short_clip", "long_clip"}

def test_media_index_csv_has_requested_columns(tmp_path):
    record = media_record(tmp_path / "a.mp4", 20)
    write_media_index([record], tmp_path / "index.json", tmp_path / "index.csv")
    assert "source_start" not in (tmp_path / "index.csv").read_text(encoding="utf-8")
    assert "file_path" in (tmp_path / "index.csv").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests and verify the expected missing-symbol failures**

Run: `python -m pytest tests/test_toolchain.py tests/test_media_index.py -q`  
Expected: failure during collection because `Toolchain`, discovery, and index functions are absent.

- [ ] **Step 3: Implement safe toolchain discovery and media probing**

Use `where.exe ffmpeg` and configured/common sibling paths as individual subprocess arguments. Parse versions numerically, require an adjacent ffprobe when possible, inspect encoder listings, and run a 1280×720 null-output NVENC smoke test. Sort viable candidates by version descending, so the audited FFmpeg 8.0 wins. Store all candidates and the selection reason in `Toolchain.to_dict()`. `run_command` must set `shell=False`, use `text=True`, and log failures without converting argv to a shell string.

`probe_media` invokes the pinned ffprobe with `-of json`, parses the first video/audio streams, computes rational FPS, records creation time, and returns a category based on duration. `build_media_index` scans only the configured extensions, catches per-file probe errors, preserves a list of errors in the JSON envelope, and writes `media_index.json` plus `media_index.csv` atomically. It must not create any path below `raw`.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `python -m pytest tests/test_toolchain.py tests/test_media_index.py -q`  
Expected: all tests pass.

## Task 3: Add long-recording proxies and cached audio/video activity analysis

**Files:**
- Create: `montage/proxy.py`
- Create: `montage/audio_analysis.py`
- Create: `montage/video_analysis.py`
- Create: `tests/test_video_analysis.py`

**Interfaces:**
- `build_proxy(record: MediaRecord, config: PipelineConfig, toolchain: Toolchain) -> Path | None`
- `extract_analysis_audio(source: Path, destination: Path, toolchain: Toolchain, sample_rate: int = 22050) -> Path`
- `analyze_audio_waveform(samples: np.ndarray, sample_rate: int) -> dict[str, list[float] | float]`
- `analyze_video_activity(record: MediaRecord, source_for_analysis: Path, audio_features: dict, config: PipelineConfig, toolchain: Toolchain) -> VideoAnalysis`
- `normalize_signal(values: Sequence[float]) -> list[float]`
- `merge_activity_peaks(activity: Sequence[float], times: Sequence[float], threshold: float, gap: float, min_duration: float, max_duration: float) -> list[tuple[float, float]]`

- [ ] **Step 1: Write failing tests for proxy policy, signal normalization, and contiguous peak merging**

```python
def test_short_clip_does_not_get_proxy(short_record, config, fake_toolchain):
    assert build_proxy(short_record, config, fake_toolchain) is None

def test_long_proxy_is_written_under_work_not_raw(long_record, config, fake_toolchain, monkeypatch):
    monkeypatch.setattr("montage.proxy.run_command", fake_successful_ffmpeg)
    proxy = build_proxy(long_record, config, fake_toolchain)
    assert proxy.parent == config.proxy_dir
    assert not str(proxy).startswith(str(config.raw_dir))

def test_normalize_signal_is_bounded_and_constant_safe():
    assert normalize_signal([2, 4, 6]) == [0.0, 0.5, 1.0]
    assert normalize_signal([3, 3]) == [0.0, 0.0]

def test_nearby_activity_peaks_form_one_combat_segment():
    times = [float(i) for i in range(20)]
    activity = [0.0] * 20
    for index in (5, 6, 8, 9, 10): activity[index] = 1.0
    assert merge_activity_peaks(activity, times, 0.8, 3.0, 3.0, 30.0) == [(5.0, 12.0)]
```

- [ ] **Step 2: Run the tests and verify they fail for missing implementations**

Run: `python -m pytest tests/test_video_analysis.py -q`  
Expected: missing-symbol failures.

- [ ] **Step 3: Implement cached proxy/audio/video analysis**

`build_proxy` returns `None` for duration ≤300 seconds. For long files, invoke the selected FFmpeg with a filter that fits within 960×540, `-fps_mode passthrough`, source audio mapped to mono 22050/48000 analysis audio, and an atomic output under `work/proxy`. Key the path from source fingerprint and relevant proxy settings. The command must pass `record.file_path` as one argv element.

`extract_analysis_audio` decodes a source or proxy audio stream to a cache WAV using the pinned FFmpeg. `analyze_audio_waveform` computes RMS, peak, spectral flux, onset strength/density, and transient density with NumPy/SciPy; use robust percentile normalization rather than assuming the loudest point is the best event.

`analyze_video_activity` reads low-rate grayscale frames from the proxy for long files or the source for short files through a pinned FFmpeg rawvideo pipe. Compute brightness/black ratio, frame difference, local entropy, scene-change spikes, and a combined visual/motion activity curve. Resample audio activity into the same one-second grid, combine and smooth it, and return arrays with source-time timestamps. Store per-source JSON and `activity_analysis.png` atomically under `work/analysis`; use Matplotlib Agg so no UI is required.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `python -m pytest tests/test_video_analysis.py -q`  
Expected: all tests pass.

## Task 4: Analyze the fixed music file and produce confidence-aware structure artifacts

**Files:**
- Create: `montage/music_analysis.py`
- Create: `tests/test_music_analysis.py`

**Interfaces:**
- `analyze_music(config: PipelineConfig, toolchain: Toolchain) -> MusicAnalysis`
- `build_edit_points(tempo: float, beat_times: Sequence[float], onset_times: Sequence[float], energy: Sequence[float], times: Sequence[float]) -> list[dict[str, object]]`
- `infer_music_structure(times: Sequence[float], energy: Sequence[float], beat_times: Sequence[float], onset_strength: Sequence[float]) -> dict[str, object]`
- `choose_preview_music_window(structure: dict[str, object], duration: float, minimum: float, maximum: float) -> tuple[float, float, str]`

- [ ] **Step 1: Write failing tests for edit-point hierarchy, energy regions, and representative-window selection**

```python
def test_edit_points_prioritize_bar_and_strong_beat():
    points = build_edit_points(120.0, [0.0, 0.5, 1.0, 1.5], [0.2], [0.1, 0.8, 0.9, 0.2], [0.0, 0.5, 1.0, 1.5])
    assert points[0]["type"] == "bar"
    assert any(point["type"] == "strong_beat" for point in points)
    assert all(0.0 <= float(point["strength"]) <= 1.0 for point in points)

def test_structure_contains_low_medium_high_regions_and_confidence():
    result = infer_music_structure([0, 1, 2, 3], [0.1, 0.2, 0.9, 1.0], [0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1])
    assert {region["energy"] for region in result["regions"]} >= {"low_energy", "high_energy"}
    assert all("confidence" in region for region in result["regions"])

def test_preview_window_prefers_build_to_high_energy_transition():
    structure = {"regions": [{"start": 70, "end": 105, "energy": "medium_energy", "role": "build_up", "confidence": 0.8}, {"start": 105, "end": 150, "energy": "high_energy", "role": "chorus", "confidence": 0.7}]}
    start, end, reason = choose_preview_music_window(structure, 230, 45, 60)
    assert 45 <= end - start <= 60
    assert start < 105 < end
    assert "high" in reason.lower() or "build" in reason.lower()
```

- [ ] **Step 2: Run the tests and verify they fail because music functions are absent**

Run: `python -m pytest tests/test_music_analysis.py -q`  
Expected: missing-symbol failures.

- [ ] **Step 3: Implement librosa analysis, confidence, JSON/CSV/PNG output, and window choice**

Load the fixed FLAC only for analysis at 22050Hz mono; never write beside the source. Use `librosa.beat.beat_track`, onset strength, RMS, spectral contrast when available, and a novelty curve. Construct bar/downbeat attempts from the prevailing beat period using 4-beat groups; set lower confidence when beat regularity or structure evidence is weak. Classify robust energy quantiles into low/medium/high and detect novelty/energy boundaries without claiming exact song labels.

Write `beat_map.json`, `music_structure.json`, `energy_curve.csv`, and `music_analysis.png` under `work/analysis/music`. Include the actual 230-second duration, estimated BPM near the audit result, all timestamps, point types, and confidence fields. Choose a 45–60-second window containing a build-up or boundary followed by a high-energy region when possible; otherwise choose the highest-scoring contiguous window and say why in the returned reason.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `python -m pytest tests/test_music_analysis.py -q`  
Expected: all tests pass.

## Task 5: Generate candidates, rank them, and deduplicate short-versus-long events

**Files:**
- Create: `montage/candidate.py`
- Create: `montage/dedupe.py`
- Create: `montage/ranking.py`
- Create: `tests/test_candidates_dedupe.py`

**Interfaces:**
- `generate_candidates(records: Sequence[MediaRecord], analyses: Mapping[str, VideoAnalysis], config: PipelineConfig) -> list[Candidate]`
- `trim_non_content_edges(candidate: Candidate, analysis: VideoAnalysis) -> Candidate`
- `score_candidates(candidates: Sequence[Candidate], config: PipelineConfig) -> list[Candidate]`
- `fingerprint_candidate(candidate: Candidate, source_for_analysis: Path, toolchain: Toolchain) -> list[int]`
- `deduplicate_candidates(candidates: Sequence[Candidate], fingerprints: Mapping[str, Sequence[int]], threshold: float) -> DedupeResult`
- `choose_representative(group: Sequence[Candidate], purpose: Literal["fast", "full"]) -> Candidate`
- `write_candidates(candidates: Sequence[Candidate], json_path: Path, csv_path: Path) -> None`

- [ ] **Step 1: Write failing tests for human priority, coherent long candidates, weighted ranking, and dedupe preference**

```python
def test_short_clip_gets_human_selection_prior_without_fragmentation(short_record, analysis, config):
    candidates = generate_candidates([short_record], {short_record.file_path.as_posix(): analysis}, config)
    assert len(candidates) == 1
    assert candidates[0].human_selection_score >= 0.30
    assert candidates[0].duration >= 10.0

def test_long_activity_peaks_merge_into_one_window(long_record, long_analysis, config):
    candidates = generate_candidates([long_record], {long_record.file_path.as_posix(): long_analysis}, config)
    assert any(candidate.source_end - candidate.source_start >= 6.0 for candidate in candidates)

def test_weighted_score_is_configurable():
    candidate = candidate_with_scores(human=1.0, combat=0.0, motion=0.0, audio=0.0, visual=0.0, continuity=0.0, unique=0.0)
    assert score_candidates([candidate], config_with_weights({"human_selection_prior": 1.0}))[0].final_score == 1.0

def test_duplicate_group_prefers_saved_short_for_fast():
    short = candidate_with_id("short", duration=20, human=0.4)
    long = candidate_with_id("long", duration=28, human=0.0)
    result = deduplicate_candidates([short, long], {"short": [0, 1, 2], "long": [0, 1, 2]}, threshold=0.9)
    assert choose_representative(result.groups[0], "fast").candidate_id == "short"
    assert result.groups[0][0].duplicate_group == result.groups[0][1].duplicate_group
```

- [ ] **Step 2: Run the tests and verify they fail for absent candidate/dedupe modules**

Run: `python -m pytest tests/test_candidates_dedupe.py -q`  
Expected: missing-symbol failures.

- [ ] **Step 3: Implement candidate generation, score normalization, dHash fingerprints, and grouping**

For short clips, make one candidate after removing only leading/trailing black, loading, static scoreboard, or menu intervals; retain a high `human_selection_score`. For long analyses, threshold the smoothed combined activity curve, merge gaps ≤3 seconds, clamp combat cores to 3–30 seconds, then add configured pre/post roll within source bounds. Do not generate one candidate per beat.

Normalize each component robustly to `[0, 1]`; compute `combat_intensity` from the upper activity quantile and `continuity_score` from duration/curve support. Apply the YAML weights exactly, including `uniqueness`, and persist both requested field names and component diagnostics.

Fingerprint low-resolution grayscale samples with a simple 8×8 dHash. Compare the shorter fingerprint against sliding windows of the longer one using mean Hamming similarity. Union similar candidates into deterministic `dg-###` groups. Fast representatives prefer a ≤90-second human-selected candidate; Full representatives may choose a longer candidate only when it has materially more context and a comparable score. Write `highlight_candidates.json` and `.csv` with duplicate groups, fingerprints, and rationales.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `python -m pytest tests/test_candidates_dedupe.py -q`  
Expected: all tests pass.

## Task 6: Build the music-aware, gameplay-first Preview EDL and review assets

**Files:**
- Create: `montage/transitions.py`
- Create: `montage/timeline.py`
- Create: `montage/review.py`
- Create: `tests/test_timeline.py`

**Interfaces:**
- `choose_sync_point(shot: Candidate, music: MusicAnalysis, target: float, tolerance: float = 0.20) -> tuple[float, str, float]`
- `choose_transition(previous: Candidate | None, current: Candidate, analyses: Mapping[str, VideoAnalysis]) -> str`
- `build_preview_edit(candidates: Sequence[Candidate], music: MusicAnalysis, config: PipelineConfig) -> EditDecisionList`
- `write_edit_list(edit: EditDecisionList, json_path: Path, timeline_path: Path) -> None`
- `render_review_assets(candidates: Sequence[Candidate], config: PipelineConfig, toolchain: Toolchain) -> Path`

- [ ] **Step 1: Write failing tests for narrative energy, continuity over beats, EDL schema, and restrained transitions**

```python
def test_preview_energy_rises_and_strongest_shot_is_near_end():
    edit = build_preview_edit(test_candidates(), test_music(), preview_config())
    assert edit.duration <= 60.0
    assert edit.shots[0].candidate_score < edit.shots[-1].candidate_score

def test_continuous_seven_second_action_is_not_cut_at_every_beat():
    edit = build_preview_edit([candidate_with_id("multi", duration=7.0, score=0.95)], test_music_with_many_beats(), preview_config())
    assert len(edit.shots) == 1

def test_edit_shot_contains_required_preview_fields():
    shot = build_preview_edit(test_candidates(), test_music(), preview_config()).shots[0].to_dict()
    assert {"source", "source_in", "source_out", "duration", "candidate_score", "duplicate_group", "timeline_in", "timeline_out", "transition", "music_target", "music_event_type", "sync_offset", "rationale"} <= set(shot)

def test_default_transition_is_hard_cut():
    assert choose_transition(None, candidate_with_id("x"), {}) == "hard_cut"
```

- [ ] **Step 2: Run focused tests and verify they fail because timeline functions are absent**

Run: `python -m pytest tests/test_timeline.py -q`  
Expected: missing-symbol failures.

- [ ] **Step 3: Implement representative music-window selection and EDL writing**

Use music sections to allocate the preview: 4–8 seconds for an establishing opening, 2.5–5 seconds in medium energy, and 1–3 seconds only where the selected candidate itself is short enough and continuity remains intact. Use section/phrase boundaries first, strong beats/downbeats second, and ordinary beats/onsets only for an action alignment target. Do not add cuts solely because a beat exists. Reject duplicate groups already used in the edit and keep the final 10% for the strongest available material. Build `EditShot` instances with explicit rationales and exact source bounds.

`choose_transition` returns `hard_cut` by default and may return `match_cut` only when ending/starting motion direction and strength meet a conservative threshold. It never returns effects outside the allowed set. `write_edit_list` writes `work/analysis/preview_edit.json` and a readable `preview_timeline.txt` with `HH:MM.mmm`, source range, score, section, sync offset, transition, and rationale.

`render_review_assets` creates a thumbnail and low-bitrate preview per candidate under `work/review`, plus a local `review_manifest.json` and `review.html` with KEEP/DROP/STAR buttons persisted in browser local storage. The HTML does not call a server or cloud service.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `python -m pytest tests/test_timeline.py -q`  
Expected: all tests pass.

## Task 7: Compile safe FFmpeg shot renders with game-first audio ducking

**Files:**
- Create: `montage/audio_mix.py`
- Create: `montage/ffmpeg_renderer.py`
- Create: `tests/test_renderer.py`

**Interfaces:**
- `build_audio_filter(game_label: str, music_label: str, game_gain_db: float, music_gain_db: float) -> str`
- `compile_shot_argv(shot: EditShot, music_file: Path, destination: Path, config: PipelineConfig, toolchain: Toolchain) -> list[str]`
- `render_edit(edit: EditDecisionList, music: MusicAnalysis, config: PipelineConfig, toolchain: Toolchain, output_path: Path) -> Path`
- `probe_output(path: Path, toolchain: Toolchain) -> dict[str, object]`

- [ ] **Step 1: Write failing tests for subprocess safety, source bounds, output sizing, and audio filter policy**

```python
def test_compile_argv_keeps_unicode_source_and_never_uses_shell():
    shot = edit_shot(source=Path(r"D:\91\集锦\raw\战斗 [A].mp4"), source_in=2.0, source_out=7.0)
    argv = compile_shot_argv(shot, Path(r"C:\音乐\決意.flac"), Path("segment.mp4"), preview_config(), fake_toolchain)
    assert argv[0].endswith("ffmpeg.exe")
    assert str(shot.source) in argv
    assert "-filter_complex" in argv
    assert "shell=True" not in argv

def test_audio_filter_ducks_music_but_keeps_game_audio():
    graph = build_audio_filter("game", "music", 2.0, -6.0)
    assert "sidechaincompress" in graph
    assert "amix" in graph
    assert "game" in graph and "music" in graph

def test_renderer_rejects_shot_outside_source_duration():
    with pytest.raises(ValueError):
        compile_shot_argv(edit_shot(source_in=10.0, source_out=12.0, source_duration=11.0), music_file, Path("x.mp4"), preview_config(), fake_toolchain)
```

- [ ] **Step 2: Run focused renderer tests and verify they fail for absent renderer symbols**

Run: `python -m pytest tests/test_renderer.py -q`  
Expected: missing-symbol failures.

- [ ] **Step 3: Implement segmented rendering and concat**

Compile one manageable argv per shot. Use `-ss`/`-t` only against the source path, map the source game audio when present, use `scale=1920:1200:force_original_aspect_ratio=decrease,pad=1920:1200:(ow-iw)/2:(oh-ih)/2`, preserve timestamps/cadence without interpolation, and encode with pinned `h264_nvenc` when `Toolchain.nvenc_h264` is true or `libx264` otherwise. Mix the shot’s corresponding music interval with a smooth `sidechaincompress` graph, a broad activity-based music gain, `aresample=async=1:first_pts=0`, and AAC 48kHz 320kbps. Use `loudnorm=I=-14:TP=-1` at the final segment boundary.

Write segments and concat manifests only under `work/cache/render/<run-id>`. Validate every shot source range before invoking FFmpeg. After all segments succeed, concatenate with `-f concat -safe 0` through the pinned FFmpeg and atomically replace the requested output. `probe_output` uses the pinned ffprobe to confirm video/audio streams, duration, 1920×1200 dimensions, and approximate 60fps. It must never resolve an output inside `raw`.

- [ ] **Step 4: Run focused renderer tests and verify they pass**

Run: `python -m pytest tests/test_renderer.py -q`  
Expected: all tests pass.

## Task 8: Add orchestration, environment/report logging, and CLI stage guards

**Files:**
- Create: `main.py`
- Create: `README.md`
- Create: `tests/test_cli.py`

**Interfaces:**
- `run_pipeline(command: str, config_path: Path, dry_run: bool = False) -> int`
- CLI commands: `index`, `analyze-music`, `analyze-video`, `candidates`, `review`, `render-preview`, `render-fast`, `render-full`, `all`
- `write_environment_report(toolchain: Toolchain, config: PipelineConfig, path: Path) -> None`
- `write_preview_report(edit: EditDecisionList, music: MusicAnalysis, candidates: Sequence[Candidate], output_probe: dict[str, object], path: Path) -> None`

- [ ] **Step 1: Write failing tests for dry-run behavior, output guard, and CLI command registration**

```python
def test_all_dry_run_does_not_call_renderer(monkeypatch, config_path):
    called = {"render": False}
    monkeypatch.setattr("main.render_preview_stage", lambda *args: called.__setitem__("render", True))
    assert run_pipeline("all", config_path, dry_run=True) == 0
    assert called["render"] is False

def test_render_fast_and_full_are_blocked_before_preview_approval(config_path):
    assert run_pipeline("render-fast", config_path) != 0
    assert run_pipeline("render-full", config_path) != 0

def test_cli_exposes_required_commands():
    parser = build_parser()
    commands = {action.dest for action in parser._subparsers._group_actions[0].choices.values()}
    assert {"index", "analyze-music", "analyze-video", "candidates", "review", "render-preview", "render-fast", "render-full", "all"} <= set(parser._subparsers._group_actions[0].choices)
```

- [ ] **Step 2: Run CLI tests and verify they fail before `main.py` exists**

Run: `python -m pytest tests/test_cli.py -q`  
Expected: missing-module or missing-parser failures.

- [ ] **Step 3: Implement stage orchestration and the milestone stop guard**

At startup, create only `work` and `output` subdirectories, initialize `pipeline.log`, discover the toolchain once, write `environment.json`, and pass the same context through every stage. The `all` command runs environment, index, music, video, candidates/dedupe, review, and preview EDL. With `--dry-run`, it stops after EDL creation and never calls the renderer. The first milestone allows only `render-preview`; `render-fast` and `render-full` return a clear nonzero message stating that they are intentionally disabled until human approval of the preview.

The CLI must catch per-file stage errors, log them, and continue where safe. Run-level missing music/toolchain errors return nonzero. The report writer must include:

```text
video analysis summary
music structure and confidence
selected candidate ids and source ranges
dedupe groups and representatives
music_in/music_out and selection reason
average/min/max shot duration
Hard Cut/Match Cut/Crossfade counts
mean and P95 absolute beat-sync offset
game-audio/music ducking policy
selected FFmpeg/NVENC absolute paths
exact preview_60s.mp4 path
```

Document Windows setup, FFmpeg discovery, the read-only raw guarantee, CLI examples, cache behavior, and the preview stop condition in `README.md`.

- [ ] **Step 4: Run focused CLI tests and verify they pass**

Run: `python -m pytest tests/test_cli.py -q`  
Expected: all tests pass.

## Task 9: Run the full test suite, real smoke set, and the single Preview milestone

**Files:**
- Modify: `README.md` if real-machine usage findings require corrected command text
- Generate only under: `work/` and `output/`

**Interfaces:**
- The command sequence is the acceptance interface:

```powershell
python -m pytest -q
python main.py all --dry-run
python main.py render-preview
```

- [ ] **Step 1: Run all automated tests before touching real raw media**

Run from `D:\91\集锦\battlefield-montage`: `python -m pytest -q`  
Expected: all tests pass; any failure is fixed with a new failing regression test before changing production code.

- [ ] **Step 2: Capture the immutable RAW manifest before the smoke run**

Record for every recognized raw file: absolute path, file size, mtime, and file name. Save this manifest under `work/analysis/raw_manifest_before.json`; this is generated data and is not placed in `raw`.

- [ ] **Step 3: Run dry-run stages on the real corpus**

Run: `python main.py all --dry-run`  
Expected artifacts include `environment.json`, `media_index.json`, `media_index.csv`, music JSON/CSV/PNG, per-source activity analysis, `highlight_candidates.json`, `highlight_candidates.csv`, dedupe summary, review assets, `preview_edit.json`, and `preview_timeline.txt`; no final preview video is created by this command.

- [ ] **Step 4: Verify the selected real candidates and EDL before rendering**

Read `preview_edit.json` and assert every shot source is inside `raw`, every source range is within the probed duration, no duplicate group occurs twice, total timeline duration is 45–60 seconds, and the music window crosses the reported build/high-energy transition. Confirm the EDL exists before any render command.

- [ ] **Step 5: Render only the requested preview**

Run: `python main.py render-preview`  
Expected: `D:\91\集锦\output\preview_60s.mp4` is created, while `Battlefield_Fast_Montage.mp4` and `Battlefield_Full_Highlights.mp4` are not created.

- [ ] **Step 6: Verify output and RAW immutability with fresh commands**

Run: `python main.py verify-preview` if implemented by the CLI, otherwise run the equivalent probe/test command recorded in README. Compare the before/after raw manifest, probe the output with the pinned ffprobe, and verify video/audio streams, 1920×1200 dimensions, approximately 60fps, 45–60 second duration, no invalid EDL ranges, and a nonzero audio stream. Read the log to confirm one fixed FFmpeg/ffprobe version and actual NVENC/CPU selection for the complete run.

- [ ] **Step 7: Write the milestone report and stop**

Write `work/analysis/preview_report.md` and report the requested A–L fields: video analysis, music structure, chosen candidates, dedupe groups, song interval and reason, average/min/max shot duration, transition counts, mean/P95 sync offset, game-audio policy, selected encoder paths, and the absolute preview path. Do not invoke `render-fast` or `render-full` after this point.

## Plan self-review

- Spec coverage: Tasks 1–2 cover safety, toolchain pinning, environment/index artifacts; Task 3 covers long-only proxy and cheap analysis; Task 4 covers BPM plus beats, strong beats, bars, onsets, energy, sections, confidence, and debug PNG; Task 5 covers short prior, long extraction, scores, fingerprints, duplicate preference, and candidate CSV/JSON; Task 6 covers EDL-before-render, narrative energy, continuity, sync hierarchy, restrained transitions, review HTML, and readable timeline; Task 7 covers source rendering, 1920×1200 preservation, NVENC/CPU choice, audio ducking, loudness, and concat; Task 8 covers CLI, logs, report, dry-run, and milestone guards; Task 9 covers real smoke, preview verification, report, and stop condition.
- Placeholder scan: no step depends on an unfinished placeholder or an unspecified error-handling task. Every step names files, interfaces, commands, and expected outcomes.
- Interface consistency: `PipelineConfig`, `Toolchain`, `MediaRecord`, `VideoAnalysis`, `MusicAnalysis`, `Candidate`, `EditShot`, and `EditDecisionList` are introduced in Task 1 and consumed by later tasks; function names and argument order are repeated consistently in Tasks 2–9.
- Safety review: no task writes to `raw`; all real outputs are under `work` or `output`; the renderer validates source bounds and output location before invocation; full final outputs are guarded and not called in the milestone.
