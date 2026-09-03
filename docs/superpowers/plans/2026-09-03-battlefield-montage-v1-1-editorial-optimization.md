# Battlefield Montage V1.1 Editorial Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a payoff-aware, context-safe V2 editorial pipeline that generates and verifies D:/91/集锦/output/preview_60s_v2.mp4 while preserving the V1 baseline and stopping before full montage rendering.

**Architecture:** Keep V1 artifacts and commands usable, add versioned V2 models and artifacts, and isolate payoff detection, condensation, beam-search timeline, audio continuity, visualization, and reporting behind explicit V2 functions. The V2 renderer composes continuous music with a conservative game-audio timeline and uses ordinary gameplay-motivated hard cuts.

**Tech Stack:** Python 3.13, NumPy/SciPy/librosa/SoundFile/Matplotlib/OpenCV, standard-library dataclasses/json/csv/subprocess/pathlib, PyYAML, pytest, pinned FFmpeg 8.0 with tested h264_nvenc.

**Spec:** docs/superpowers/specs/2026-09-03-battlefield-montage-v1-1-editorial-optimization-design.md

## Global Constraints

- D:/91/集锦/raw remains permanently read-only; no file below it may be deleted, moved, renamed, overwritten, metadata-modified, or encoded in place.
- Intermediates remain below D:/91/集锦/work; finished video remains below D:/91/集锦/output.
- The selected FFmpeg 8.0 executable and sibling ffprobe are discovered once, runtime-tested, recorded, and reused for the entire run; the PATH-first FFmpeg 4.3.1 must not be selected for NVENC work.
- D:/91/集锦/output/preview_60s.mp4 is the immutable V1 baseline; V1.1 writes preview_60s_v2.mp4 and never overwrites the baseline.
- V2 uses the baseline music interval 19.000s–74.252s by default; a ±0.5 second change requires a recorded reason and both ranges in the report.
- V1.1 generates only preview_60s_v2.mp4 and versioned analysis/debug artifacts; complete Fast Montage and Full Highlights files remain absent and guarded.
- Preview geometry remains 1920×1200 at approximately source cadence, with no interpolation, destructive 16:9 crop, or upscale of a smaller source.
- V1.1 uses ROI event evidence, temporal differencing, audio transient evidence, multi-evidence fusion, payoff anchors, Action-to-Beat placement, and a bounded beam search; it uses no Video LLM, cloud API, online service, or required OCR installation.
- OCR is optional; OpenCV non-OCR detection must remain functional and record ocr_available=false when OCR is unavailable.
- Semantic event confidence is separate from meaningful-event confidence. A single ROI change cannot create kill, multikill, vehicle_destroy, or objective semantics.
- Event peaks within event_merge_window_ms default 700ms are merged; strong_anchor_threshold and weak_anchor_threshold default to 0.75 and 0.55 and remain configurable.
- Gameplay quality and continuity outrank music precision, diversity, and transition effects. A natural larger sync offset is preferred over context damage.
- Hard Cut remains the ordinary video transition. Compatibility affects selection and metadata only; no artificial white frame, glitch, RGB split, zoom, shake, warp, motion blur, flash spam, or template speed ramp is allowed.
- The V2 EDL, sync report, and timeline plot must exist before rendering; rendering must verify RAW/baseline integrity and full decode before stopping for human A/B review.

---

## File Map and Stable Interfaces

~~~text
config.yaml
README.md
main.py
montage/config.py
montage/models.py
montage/cache.py
montage/music_analysis.py
montage/payoff_detection.py          new
montage/candidate.py
montage/condense.py                  new
montage/ranking.py
montage/dedupe.py
montage/transitions.py
montage/beam_timeline.py             new
montage/audio_mix.py
montage/v2_renderer.py               new
montage/timeline_visualization.py    new
montage/v2_report.py                 new
tests/test_v2_contracts.py           new
tests/test_v2_music.py               new
tests/test_payoff_detection.py       new
tests/test_condense_ranking.py       new
tests/test_v2_dedupe.py              new
tests/test_v2_timeline.py            new
tests/test_v2_renderer.py            new
tests/test_v2_cli_report.py          new
~~~

The following V2 model names and fields are the contract between tasks:

~~~python
@dataclass(frozen=True)
class PayoffEvent:
    event_id: str
    type: str
    source_time: float
    confidence: float
    strength: float
    semantic_confidence: float
    evidence: dict[str, float]
    detector_flags: tuple[str, ...] = ()
    merged_peak_count: int = 1

@dataclass(frozen=True)
class SourceSegment:
    source: Path
    source_in: float
    source_out: float
    duration: float

@dataclass(frozen=True)
class CandidateVariant:
    variant_id: str
    parent_candidate_id: str
    source_file: Path
    source_segments: tuple[SourceSegment, ...]
    duration: float
    human_selection_prior: float
    payoff_score: float
    combat_intensity: float
    action_density: float
    continuity: float
    visual_novelty: float
    motion: float
    audio_activity: float
    danger_score: float
    uniqueness: float
    final_score: float
    duplicate_group: str | None
    payoff_events: tuple[PayoffEvent, ...]
    primary_anchor: PayoffEvent | None
    secondary_anchors: tuple[PayoffEvent, ...]
    anchor_event_time: float | None
    anchor_event_type: str | None
    anchor_event_strength: float | None
    anchor_event_confidence: float | None
    context_integrity_score: float
    penalty_values: dict[str, float]
    source_signature: str
    environment_signature: str
    weapon_or_view_signature: str
    condense_reason: str
    rationale: str
~~~

V2EditShot extends the V1 shot fields with source_segments, parent_candidate_id, variant_id, payoff_events, anchor_event_time, anchor_event_type, anchor_event_strength, anchor_event_confidence, primary_anchor, secondary_anchors, context_integrity_score, condense_reason, event_timeline, event_sync_offset, cut_sync_offset, transition_compatibility_score, impact_cut, audio_j_cut_ms, and audio_l_cut_ms. V2EditDecisionList stores kind, music_source, baseline_music_in/out, V2 music_in/out, music_reason, duration, and an ordered tuple of V2EditShot values.

## Task 1: Versioned configuration, models, cache identity, and baseline guard

**Files:**
- Modify: config.yaml
- Modify: montage/config.py
- Modify: montage/models.py
- Modify: montage/cache.py
- Create: tests/test_v2_contracts.py

**Interfaces:**
- PipelineConfig gains V2 detector, ROI, condensation, beam, audio-overlap, output, and baseline fields.
- PipelineConfig exposes `music_v2_analysis_dir -> Path` and versioned V2 artifact paths without changing V1 artifact locations.
- Add model to_dict methods for PayoffEvent, SourceSegment, CandidateVariant, V2EditShot, and V2EditDecisionList.
- Preserve payoff_events and anchor_event_time/type/strength/confidence on candidate and shot artifacts.
- Add baseline_manifest(path: Path) -> dict[str, object].
- Add assert_baseline_unchanged(before: dict[str, object], path: Path) -> None.
- Add v2_cache_key(source_fingerprint: dict[str, object], stage: str, parameters: dict[str, object]) -> str.

- [ ] **Step 1: Write failing tests**

~~~python
def test_v2_defaults(base_config):
    assert base_config.event_merge_window_ms == 700
    assert base_config.strong_anchor_threshold == 0.75
    assert base_config.weak_anchor_threshold == 0.55
    assert base_config.beam_width == 16
    assert (base_config.baseline_music_in, base_config.baseline_music_out) == (19.0, 74.252)

def test_v2_shot_contains_source_segments_and_sync_fields(tmp_path):
    shot = make_v2_shot(tmp_path, make_event("e1", 2.0))
    payload = shot.to_dict()
    assert {"source_segments", "primary_anchor", "context_integrity_score",
            "condense_reason", "event_timeline", "event_sync_offset",
            "cut_sync_offset", "transition_compatibility_score",
            "audio_j_cut_ms", "audio_l_cut_ms"} <= set(payload)

def test_baseline_guard_detects_change(tmp_path):
    path = tmp_path / "preview_60s.mp4"
    path.write_bytes(b"baseline")
    before = baseline_manifest(path)
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="baseline"):
        assert_baseline_unchanged(before, path)
~~~

- [ ] **Step 2: Run D:/miniconda/python.exe -m pytest -q tests/test_v2_contracts.py and confirm failure because the V2 fields/helpers are absent.**
- [ ] **Step 3: Add config values: baseline_music_in/out 19.0/74.252, v2_output_name preview_60s_v2.mp4, payoff_analysis_fps 6.0, event_merge_window_ms 700, strong/weak thresholds 0.75/0.55, max_anchor_count_per_candidate 3, preferred macro duration 2–10s, hero maximum 12s, beam_width 16, beam_max_expansions 32, recent source/environment windows 2, baseline_music_max_shift 0.5, audio overlap 100–250ms, impact tail maximum 400ms, all v2_weights, beam_weights, penalty_weights, and normalized roi_profile values.**
- [ ] **Step 4: Implement immutable dataclasses and serialization. Require at least one source segment and validate each segment duration against source_out - source_in within 1ms. Preserve payoff_events and anchor event metadata. Serialize paths as absolute strings and tuples as arrays.**
- [ ] **Step 5: Implement baseline_manifest with resolved path, size, mtime, and SHA-256. Implement v2_cache_key from sorted JSON containing source fingerprint, stage/version, FFmpeg version, and parameters.**
- [ ] **Step 6: Run the focused test again, run git diff --check, and commit:**

~~~powershell
git add config.yaml montage/config.py montage/models.py montage/cache.py tests/test_v2_contracts.py
git commit -m "feat: add v2 editorial contracts and safeguards"
~~~

## Task 2: HPSS-aware music analysis with strict baseline interval lock

**Files:**
- Modify: montage/music_analysis.py
- Modify: montage/config.py
- Create: tests/test_v2_music.py

**Interfaces:**
- Add MusicWindowDecision with baseline_music_in/out, v2_music_in/out, changed, and reason.
- Add analyze_music_v2(config: PipelineConfig, toolchain: Toolchain, baseline_edit: EditDecisionList) -> tuple[MusicAnalysis, MusicWindowDecision].
- Add choose_v2_music_window(structure: dict[str, object], baseline_in: float, baseline_out: float, duration: float, max_shift: float) -> MusicWindowDecision.
- Write V2 artifacts under config.music_v2_analysis_dir: beat_map_v2.json, music_structure_v2.json, energy_curve_v2.csv, and music_analysis_v2.png.

- [ ] **Step 1: Write failing tests.**

~~~python
def test_window_stays_at_baseline_without_better_boundary():
    decision = choose_v2_music_window(make_structure(), 19.0, 74.252, 230.0, 0.5)
    assert decision.v2_music_in == 19.0
    assert decision.v2_music_out == 74.252
    assert decision.changed is False

def test_v2_music_uses_separate_artifacts(music_config, fake_toolchain):
    analysis, decision = analyze_music_v2(music_config, fake_toolchain, make_baseline_edit())
    assert (music_config.music_v2_analysis_dir / "beat_map_v2.json").exists()
    assert decision.baseline_music_in == 19.0

def test_percussive_confidence_is_recorded(music_config, fake_toolchain):
    analysis, _ = analyze_music_v2(music_config, fake_toolchain, make_baseline_edit())
    assert "percussive" in analysis.confidence
    assert "phrase" in analysis.confidence
~~~

- [ ] **Step 2: Run D:/miniconda/python.exe -m pytest -q tests/test_v2_music.py and confirm failure.**
- [ ] **Step 3: Load the V1 EDL and make 19.000–74.252 the initial V2 range. Permit an endpoint shift only when a materially better phrase/section boundary is within ±0.5 seconds; return changed=False otherwise.**
- [ ] **Step 4: Apply librosa.effects.hpss, analyze full and percussive beat/onset hypotheses, compare interval stability, and retain confidence keys beat, strong_beat, bar, downbeat, phrase, section, onset, and percussive. Store high-energy and low-energy regions. Keep section semantics explicitly heuristic.**
- [ ] **Step 5: Write V2-only JSON/CSV/PNG artifacts atomically. Plot RMS, percussive onset, ordinary/strong/downbeat markers, phrase/section boundaries, and the locked/adjusted range with readable labels.**
- [ ] **Step 6: Run focused tests, inspect cache hit behavior, and commit:**

~~~powershell
git add montage/music_analysis.py montage/config.py tests/test_v2_music.py
git commit -m "feat: add hpss-aware v2 music analysis"
~~~

## Task 3: Multi-evidence payoff detection and 700ms event fusion

**Files:**
- Create: montage/payoff_detection.py
- Modify: montage/video_analysis.py
- Modify: montage/audio_analysis.py
- Modify: montage/config.py
- Create: tests/test_payoff_detection.py

**Interfaces:**
- EvidenceSample stores source_time and normalized ROI, motion, novelty, danger, and audio evidence.
- detect_payoff_events(source: Path, source_start: float, source_end: float, analysis: VideoAnalysis, audio: Mapping[str, Sequence[float]], config: PipelineConfig, toolchain: Toolchain) -> list[PayoffEvent].
- fuse_evidence(sample: EvidenceSample, config: PipelineConfig) -> PayoffEvent | None.
- classify_semantics(evidence: dict[str, float], config: PipelineConfig) -> tuple[str, float].
- merge_event_peaks(events: Sequence[PayoffEvent], window_ms: int) -> list[PayoffEvent].
- write_payoff_events(events: Sequence[PayoffEvent], path: Path) -> None.

- [ ] **Step 1: Write failing tests.**

~~~python
def test_lone_killfeed_change_is_not_a_kill():
    sample = EvidenceSample(source_time=12.4, killfeed_change=0.95, motion_peak=0.1)
    event = fuse_evidence(sample, test_config())
    assert event.type in {"visual_transient", "combat_climax"}
    assert event.semantic_confidence < test_config().weak_anchor_threshold

def test_independent_evidence_fuses_without_fake_semantics():
    sample = EvidenceSample(source_time=12.43, reward_roi_change=0.81,
                            crosshair_event=0.76, audio_transient=0.91,
                            motion_peak=0.68, killfeed_change=0.44)
    event = fuse_evidence(sample, test_config())
    assert event.confidence >= 0.75
    assert set(event.evidence) >= {"reward_roi_change", "crosshair_event",
                                   "audio_transient", "motion_peak", "killfeed_change"}

def test_peaks_within_700ms_merge():
    merged = merge_event_peaks([make_event("a", 10.0, 0.72),
                                make_event("b", 10.55, 0.81)], 700)
    assert len(merged) == 1
    assert merged[0].source_time == 10.55
    assert merged[0].merged_peak_count == 2
~~~

- [ ] **Step 2: Run the focused test and confirm failure.**
- [ ] **Step 3: Decode short clips directly and long candidate windows through the long-only proxy/refinement path at payoff_analysis_fps. Pass source_start through every timestamp calculation and keep all destinations under work.**
- [ ] **Step 4: Extract configured normalized ROIs for reward/score, killfeed, crosshair, objective, damage border, plus frame motion/luminance/novelty and nearest audio transient/RMS/impact evidence. Record ocr_available and never require OCR.**
- [ ] **Step 5: Require at least two independent evidence families for kill, multikill, vehicle_destroy, and objective. Fuse independent evidence with diminishing returns, keep confidence separate from semantic_confidence, clamp to [0, 1], and downgrade unsupported events to combat_climax, impact_event, visual_transient, or danger_climax.**
- [ ] **Step 6: Apply the 700ms event refractory/merge rule, sort peaks by source time, merge gaps at most 700ms, preserve strongest timestamp, all evidence channels, detector flags, and merged_peak_count. Cache the detector version, ROI version, thresholds, source fingerprint, raw events, merged events, and confidence histogram.**
- [ ] **Step 7: Run the focused test, inspect a synthetic event JSON, and commit:**

~~~powershell
git add montage/payoff_detection.py montage/video_analysis.py montage/audio_analysis.py montage/config.py tests/test_payoff_detection.py
git commit -m "feat: detect and fuse payoff events"
~~~

## Task 4: Anchor choice, safe condensation, payoff ranking, and penalties

**Files:**
- Create: montage/condense.py
- Modify: montage/candidate.py
- Modify: montage/ranking.py
- Modify: montage/models.py
- Modify: montage/config.py
- Create: tests/test_condense_ranking.py

**Interfaces:**
- select_anchors(events: Sequence[PayoffEvent], candidate_start: float, candidate_end: float, music: MusicAnalysis, config: PipelineConfig) -> tuple[PayoffEvent | None, tuple[PayoffEvent, ...]].
- context_integrity_score(candidate: Candidate, primary: PayoffEvent | None, segments: Sequence[SourceSegment]) -> float.
- build_condensed_variants(candidate: Candidate, events: Sequence[PayoffEvent], music: MusicAnalysis, config: PipelineConfig) -> list[CandidateVariant].
- calculate_penalties(features: dict[str, float], events: Sequence[PayoffEvent], config: PipelineConfig) -> dict[str, float].
- score_variant(variant: CandidateVariant, config: PipelineConfig) -> CandidateVariant.
- write_v2_candidates(variants: Sequence[CandidateVariant], json_path: Path, csv_path: Path) -> None.
- Penalty output includes stationary_ads_penalty, same_view_penalty, downtime_penalty, no_payoff_penalty, repetitive_fire_penalty, same_source_recent_penalty, and same_environment_recent_penalty.

- [ ] **Step 1: Write failing tests.**

~~~python
def test_late_multikill_can_beat_earlier_isolated_kill():
    primary, secondary = select_anchors(
        [make_event("kill", 2.0, 0.92, 0.55, "kill"),
         make_event("multi", 8.0, 0.84, 0.95, "multikill")],
        0.0, 10.0, make_music(), test_config())
    assert primary.event_id == "multi"
    assert secondary[0].event_id == "kill"

def test_stationary_ads_penalty_is_payoff_aware():
    features = {"stationary_ads": 0.9, "motion": 0.1,
                "visual_novelty": 0.1, "danger_escalation": 0.1,
                "repetitive_fire": 0.7}
    no_event = calculate_penalties(features, [], test_config())
    with_event = calculate_penalties(features, [make_event("kill", 3.0, 0.8)], test_config())
    assert with_event["stationary_ads_penalty"] < no_event["stationary_ads_penalty"]

def test_two_segment_condense_requires_reason():
    with pytest.raises(ValueError, match="condense_reason"):
        make_variant_with_two_segments(make_long_candidate(), "")
~~~

- [ ] **Step 2: Run D:/miniconda/python.exe -m pytest -q tests/test_condense_ranking.py and confirm failure.**
- [ ] **Step 3: Select primary/secondary anchors using strength, semantic importance, candidate position, music opportunity, and context cost. Do not use maximum confidence as the sole rule.**
- [ ] **Step 4: For candidates over 12 seconds, generate one continuous anchor-centered 3–10 second variant with setup/action/payoff/tail. Generate at most one additional source range only when measured downtime, phrase/bar boundary, action phase, spatial change, or state change proves a jump cut. Record condense_reason and reject context scores below the configured minimum.**
- [ ] **Step 5: Calculate stationary ADS penalty only when low motion, low novelty, no payoff, and no danger escalation co-occur. Reduce it for kill sequences, explosions, survival, or escalation. Calculate downtime, no-payoff, same-view, and repetitive-fire diagnostics from explicit feature runs.**
- [ ] **Step 6: Score with exactly the configured V1.1 weights: payoff .25, human prior .15, combat .12, action density .12, continuity .10, visual novelty .08, motion .06, audio .05, danger .04, uniqueness .03. Subtract penalties, clamp to [0, 1], and write every component to JSON/CSV.**
- [ ] **Step 7: Run focused tests and commit:**

~~~powershell
git add montage/condense.py montage/candidate.py montage/ranking.py montage/models.py montage/config.py tests/test_condense_ranking.py
git commit -m "feat: add payoff-aware candidate condensation"
~~~

## Task 5: Dedupe condensed variants and retain saved-short preference

**Files:**
- Modify: montage/dedupe.py
- Modify: montage/cache.py
- Modify: montage/models.py
- Create: tests/test_v2_dedupe.py

**Interfaces:**
- deduplicate_variants(variants: Sequence[CandidateVariant], fingerprints: Mapping[str, Sequence[int]], threshold: float) -> V2DedupeResult.
- V2DedupeResult contains groups, pairwise similarity, threshold, representative IDs, and forced source-overlap reasons.
- choose_v2_representative(group: Sequence[CandidateVariant], purpose: Literal["fast", "full"]) -> CandidateVariant.
- fingerprint_variant(variant: CandidateVariant, source_for_analysis: Path, toolchain: Toolchain, config: PipelineConfig) -> list[int].
- write_v2_dedupe_summary(result: V2DedupeResult, path: Path) -> None.

- [ ] **Step 1: Write failing tests for saved-short preference, same-parent grouping, and threshold reporting.**
- [ ] **Step 2: Run D:/miniconda/python.exe -m pytest -q tests/test_v2_dedupe.py and confirm failure.**
- [ ] **Step 3: Fingerprint each source segment sequence with pinned FFmpeg and v2_cache_key containing variant ID, parent ID, ranges, interval, detector version, and source fingerprint.**
- [ ] **Step 4: Compare duration-aware visual sequences, force overlapping variants from one parent/event window into one group, record all similarities and forced-group reasons, and use threshold 0.78 for other pairs.**
- [ ] **Step 5: For the V2 preview representative, sort human_selection_prior before payoff, context, final score, and shorter duration. Permit long context only for the future full purpose; V2 always uses fast preference.**
- [ ] **Step 6: Run focused tests, inspect the dedupe JSON, and commit:**

~~~powershell
git add montage/dedupe.py montage/cache.py montage/models.py tests/test_v2_dedupe.py
git commit -m "feat: deduplicate v2 condensed variants"
~~~

## Task 6: Anchor placement and gameplay-only transition compatibility

**Files:**
- Modify: montage/transitions.py
- Modify: montage/video_analysis.py
- Modify: montage/models.py
- Modify: montage/config.py
- Create: tests/test_v2_timeline.py

**Interfaces:**
- BoundaryDescriptor contains motion direction/strength, luminance, visual tone, ADS state, weapon motion, impact strength, environment signature, and source signature.
- describe_variant_boundary(variant: CandidateVariant, analysis: VideoAnalysis | None, side: Literal["start", "end"]) -> BoundaryDescriptor.
- transition_compatibility_score(previous: BoundaryDescriptor, current: BoundaryDescriptor) -> float.
- choose_anchor_music_target(anchor: PayoffEvent, music: MusicAnalysis, timeline_hint: float, config: PipelineConfig) -> AnchorPlacement.
- protect_context_during_sync(variant: CandidateVariant, target_music_time: float, config: PipelineConfig) -> AnchorPlacement.
- AnchorPlacement contains source_anchor_time, target_music_time, event_type, event_sync_offset, context_integrity_score, source_in, and source_out.
- TransitionDecision contains transition, effect, compatibility_score, impact_cut, audio_j_cut_ms, and audio_l_cut_ms.
- choose_v2_transition(previous: CandidateVariant, current: CandidateVariant) -> TransitionDecision.

- [ ] **Step 1: Write failing tests for primary anchor strong-beat priority, context protection, matching motion, and selection-only hard cuts.**
- [ ] **Step 2: Run the focused test and confirm failure.**
- [ ] **Step 3: Estimate descriptors from the first/last 250–500ms and use neutral values with lower compatibility when evidence is absent.**
- [ ] **Step 4: Target primary anchors to downbeat/strong beat, secondary anchors to ordinary beat/onset, and boundaries to phrase/bar positions. Return event_sync_offset and context_integrity_score.**
- [ ] **Step 5: Reject placements that start after encounter cause, remove necessary aim/approach, or end before payoff. Keep a larger sync error when context would fall below the minimum.**
- [ ] **Step 6: Score motion direction/strength, luminance/tone, ADS/weapon state, impact, source, and environment. Mark impact_cut only for natural gameplay flash/impact; keep rendered transition hard_cut.**
- [ ] **Step 7: Run focused tests and commit:**

~~~powershell
git add montage/transitions.py montage/video_analysis.py montage/models.py montage/config.py tests/test_v2_timeline.py
git commit -m "feat: align payoff anchors and score natural cuts"
~~~

## Task 7: Bounded beam-search V2 timeline and EDL validation

**Files:**
- Create: montage/beam_timeline.py
- Modify: montage/models.py
- Modify: montage/config.py
- Modify: tests/test_v2_timeline.py

**Interfaces:**
- BeamState tracks shots, elapsed, score, used duplicate groups, recent sources/environments, energy fit, context total, and sync diagnostics.
- build_v2_preview_edit(variants: Sequence[CandidateVariant], music: MusicAnalysis, baseline_edit: EditDecisionList, config: PipelineConfig) -> V2EditDecisionList.
- expand_beam(state: BeamState, variants: Sequence[CandidateVariant], music: MusicAnalysis, config: PipelineConfig) -> list[BeamState].
- timeline_energy_target(relative_time: float) -> float.
- validate_v2_edit(edit: V2EditDecisionList, config: PipelineConfig) -> None.

- [ ] **Step 1: Write failing tests for 8–14 shots, source/environment repetition penalties, final Hero Play preference, duplicate groups, and invalid ranges.**
- [ ] **Step 2: Run D:/miniconda/python.exe -m pytest -q tests/test_v2_timeline.py and confirm failure.**
- [ ] **Step 3: Keep at most beam_width states (initial 16) and at most beam_max_expansions (initial 32) options per variant. Consider only nearby phrase/bar, primary strong/downbeat, secondary beat/onset, and context-safe fallback placements.**
- [ ] **Step 4: Score each expansion with configured weights highlight_quality .35, music_fit .25, transition_compatibility .15, diversity .15, energy_curve_fit .10, then subtract context, duplicate, same_source_recent, same_environment_recent, downtime, no-payoff, and sync penalties.**
- [ ] **Step 5: Use the piecewise timeline target: 0–20% build, 20–45% first payoff/rise, 45–60% release, 60–85% escalation, 85–100% strongest payoff/Hero Play. Preserve a stronger finale when quality is comparable.**
- [ ] **Step 6: Populate source_segments, payoff_events, anchor event metadata, primary/secondary anchors, event_timeline, event/cut offsets, transition score, impact metadata, audio overlap metadata, rationale, section, and condense_reason. Write preview_v2_edit.json and preview_v2_timeline.txt before any renderer call.**
- [ ] **Step 7: Validate source paths below raw, source ranges, unique duplicate groups, anchor/rationale presence, jump-cut reasons, 45–60 second duration, and the no-three-long-segments rule.**
- [ ] **Step 8: Run focused tests and commit:**

~~~powershell
git add montage/beam_timeline.py montage/models.py montage/config.py tests/test_v2_timeline.py
git commit -m "feat: build payoff-aware v2 beam timeline"
~~~

## Task 8: Continuous-music V2 renderer and conservative J/L audio

**Files:**
- Create: montage/v2_renderer.py
- Modify: montage/audio_mix.py
- Modify: montage/config.py
- Create: tests/test_v2_renderer.py

**Interfaces:**
- compile_v2_segment_argv(segment: SourceSegment, config: PipelineConfig, toolchain: Toolchain, destination: Path) -> list[str].
- choose_audio_overlap(previous: V2EditShot, current: V2EditShot, config: PipelineConfig) -> tuple[int, int, str].
- compile_v2_final_argv(segment_paths: Sequence[Path], edit: V2EditDecisionList, config: PipelineConfig, toolchain: Toolchain, destination: Path) -> list[str].
- render_v2_edit(edit: V2EditDecisionList, config: PipelineConfig, toolchain: Toolchain, output_path: Path) -> Path.

- [ ] **Step 1: Write failing tests for no-upscale fit, no video xfade/effects, bounded audio overlap, and raw-output rejection.**
- [ ] **Step 2: Run D:/miniconda/python.exe -m pytest -q tests/test_v2_renderer.py and confirm failure.**
- [ ] **Step 3: Render each source segment under work/cache/render_v2 with source trim, scale expressions min(iw,1920)/min(ih,1200), padding, source-cadence passthrough, H.264 NVENC when tested, and game audio only.**
- [ ] **Step 4: Select no overlap or a 100–250ms audio-only overlap based on compatibility/impact/audio density. Record J-cut/L-cut mode; permit an impact tail up to 400ms only on a safe edge.**
- [ ] **Step 5: Concatenate video with hard cuts. Build the game-audio chain with direct boundaries or acrossfade on selected edges. Read the music source once from music_in for edit.duration, sidechain-compress music against the game timeline, mix, loudnorm, resample to 48kHz AAC, and pad/trim audio to video duration.**
- [ ] **Step 6: Reject destinations inside raw; render to a unique temporary output below output and replace only preview_60s_v2.mp4 after a nonzero-size check.**
- [ ] **Step 7: Run focused tests and commit:**

~~~powershell
git add montage/v2_renderer.py montage/audio_mix.py montage/config.py tests/test_v2_renderer.py
git commit -m "feat: render v2 with gameplay-first audio continuity"
~~~

## Task 9: V2 visualization, comparison report, CLI, and verification

**Files:**
- Create: montage/timeline_visualization.py
- Create: montage/v2_report.py
- Modify: main.py
- Modify: README.md
- Modify: montage/config.py
- Create: tests/test_v2_cli_report.py

**Interfaces:**
- render_v2_timeline_plot(edit: V2EditDecisionList, music: MusicAnalysis, path: Path) -> None.
- build_v2_sync_report(edit: V2EditDecisionList, baseline: EditDecisionList, variants: Sequence[CandidateVariant], rejected: dict[str, int]) -> dict[str, object].
- write_v2_report(report: dict[str, object], path: Path) -> None.
- run_v2_analysis_pipeline(config: PipelineConfig) -> V2PipelineState.
- render_preview_v2_stage(config: PipelineConfig) -> Path.
- verify_preview_v2(config: PipelineConfig) -> int.
- V2PipelineState contains config, selected toolchain, baseline manifest, V2 music/window decision, payoff events, variants, dedupe result, V2 EDL, sync report, and rejection counts.
- Add commands all-v2 --dry-run, render-preview-v2, and verify-preview-v2.

- [ ] **Step 1: Write failing tests.**

~~~python
def test_v2_commands_are_exposed():
    parser = main.build_parser()
    assert {"all-v2", "render-preview-v2", "verify-preview-v2"} <= set(parser._subparsers._group_actions[0].choices)

def test_report_contains_baseline_and_event_metrics():
    report = build_v2_sync_report(make_v2_edit(), make_baseline_edit(),
                                  make_variants(), {"stationary_ads": 2})
    assert report["baseline_music_range"] == [19.0, 74.252]
    assert "event_sync_P95" in report
    assert "cut_sync_P95" in report
    assert "transition_compatibility_mean" in report

def test_plot_is_created_below_work(tmp_path):
    path = tmp_path / "preview_v2_timeline.png"
    render_v2_timeline_plot(make_v2_edit(), make_music_with_beats(), path)
    assert path.exists()
~~~

- [ ] **Step 2: Run D:/miniconda/python.exe -m pytest -q tests/test_v2_cli_report.py and confirm failure.**
- [ ] **Step 3: Plot music RMS/energy, strong/downbeat/phrase markers, shot spans, primary/secondary anchors, cuts, event type, and confidence with readable non-overlapping labels.**
- [ ] **Step 4: Compare V1 and V2 for baseline_music_range, v2_music_range, macro_shot_count, average_shot_duration, median_shot_duration, unique_source_count, hero_shot_count, payoff_anchor_count, strong_anchor_count, cut_sync_mean, cut_sync_median, cut_sync_P90, cut_sync_P95, event_sync_mean, event_sync_median, event_sync_P90, event_sync_P95, stationary_ads_duration_ratio, estimated_downtime_ratio, same_environment_consecutive_count, same_source_recent_penalty_count, transition_compatibility_mean, and rejection counts including rejected_by_stationary_ads, rejected_by_downtime, and rejected_by_no_payoff. Label metrics as diagnostic evidence.**
- [ ] **Step 5: Orchestrate V2 analysis without rendering: discover one toolchain, write environment_v2.json, capture baseline manifest, load V1 EDL/music, run Tasks 2–7, write V2 EDL/sync report/plot, and return state.**
- [ ] **Step 6: Require V2 EDL, sync report, and plot before rendering. Render only V2 output, compare RAW/baseline manifests, probe with selected ffprobe, write preview_v2_report.md, and make verify-preview-v2 check streams, duration, geometry, cadence, decode, ranges, dedupe, integrity, and absent full outputs.**
- [ ] **Step 7: Log cache hits/misses, OCR availability, thresholds, music ranges, beam width, encoder, output, and stop condition. Run focused tests and commit:**

~~~powershell
git add montage/timeline_visualization.py montage/v2_report.py main.py README.md montage/config.py tests/test_v2_cli_report.py
git commit -m "feat: add v2 reports visualization and cli gates"
~~~

## Task 10: Real V2 dry-run, render, verification, and stop

**Files:**
- Generated only under D:/91/集锦/work and D:/91/集锦/output.
- No source change is allowed without a new focused regression test and a separate commit.

- [ ] **Step 1: Run the full suite before real media.**

~~~powershell
D:/miniconda/python.exe -m pytest -q
D:/miniconda/python.exe -m compileall -q .
~~~

Expected: all tests pass and compileall exits 0.

- [ ] **Step 2: Run D:/miniconda/python.exe main.py all-v2 --dry-run.** Confirm V2 music artifacts, payoff events, V2 candidates, dedupe, EDL, sync report, and timeline PNG exist while preview_60s_v2.mp4 does not exist. Confirm EDL mtime precedes render start.
- [ ] **Step 3: Inspect preview_v2_sync_report.json and preview_v2_edit.json. Confirm the pool supports 8–14 macro shots, duplicate groups do not repeat, sources remain below raw, anchors/context scores exist, unsupported semantics have multi-evidence support, and the music range remains 19.000–74.252 unless a ±0.5 second reason is recorded.**
- [ ] **Step 4: Run D:/miniconda/python.exe main.py render-preview-v2.** Confirm only output/preview_60s_v2.mp4 is created or replaced and the V1 baseline path/size/mtime/SHA-256 is unchanged.
- [ ] **Step 5: Run D:/miniconda/python.exe main.py verify-preview-v2, then decode every stream using the FFmpeg path from environment_v2.json:**

~~~powershell
$environment = Get-Content -Raw -Encoding utf8 'D:/91/集锦/work/analysis/environment_v2.json' | ConvertFrom-Json
$ffmpeg = [string]$environment.toolchain.ffmpeg
& $ffmpeg -hide_banner -loglevel error -i 'D:/91/集锦/output/preview_60s_v2.mp4' -map 0 -f null NUL
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
~~~

Expected: verify-preview-v2 returns valid=true and full decode exits 0 with no diagnostics.
- [ ] **Step 6: Compare fresh RAW and baseline manifests with recorded manifests. Confirm all RAW path/size/mtime values and baseline path/size/mtime/SHA-256 match, and no complete montage output exists.**
- [ ] **Step 7: Read preview_v2_report.md and confirm it contains the requested A/B metrics, anchor evidence, separate event/cut sync statistics, transition/audio strategy, encoder paths, exact V2 path, and diagnostic caveats.**
- [ ] **Step 8: Do not run render-fast or render-full. Report V1/V2 paths and results, then stop for human A/B review.**

## Implementation Order and Review Checkpoints

1. Tasks 1–2 establish contracts, cache identity, baseline protection, and music interval lock.
2. Tasks 3–5 establish evidence-safe events, anchors, variants, penalties, and dedupe.
3. Tasks 6–7 establish context-safe Action-to-Beat, natural-cut scoring, beam search, EDL, and the no-three-long-segments guard.
4. Tasks 8–9 establish continuous game-first audio, V2 rendering, visualization, reports, CLI, and verification.
5. Task 10 is the only real-media V2 run and stops after preview_60s_v2.mp4.

After each task, run its focused test file, inspect the diff, run git diff --check, and commit. A real-run defect must first receive a focused regression test before the affected stage is rerun.

## Spec Coverage Self-Review

- RAW/baseline immutability and output boundaries: Tasks 1, 8, 9, 10.
- FFmpeg 8.0/ffprobe pinning and same-run reuse: Tasks 1, 3, 8, 9, 10.
- Optional OCR and non-OCR fallback: Task 3.
- Multi-evidence semantics, confidence separation, and 700ms merge: Task 3.
- Strong/weak anchors and primary/secondary selection: Tasks 1, 4, 6, 7.
- Context-integrity condensation and jump-cut reasons: Tasks 4, 6, 7.
- Ranking weights and payoff-aware penalties: Task 4.
- V2 dedupe and saved-short preference: Task 5.
- HPSS, beats, bars/downbeats, phrase/section, energy, confidence: Task 2.
- Baseline music lock and ±0.5 exception: Tasks 2, 7, 9, 10.
- Bounded beam search, diversity, energy progression, final climax: Task 7.
- Selection-only transition compatibility and impact cut: Task 6.
- J/L audio overlap, impact tail, continuous music, no video crossfade: Task 8.
- Versioned artifacts, visualization, A/B metrics, and stop rule: Tasks 9 and 10.
- Test-first development and real verification: every task, especially Task 10.

## Plan Self-Review Checklist

- [x] Every interface name used by a later task is defined in the file map or an earlier task.
- [x] Every V2 source range is represented by SourceSegment and validated before render.
- [x] Every semantic event can be downgraded when evidence is insufficient.
- [x] No task writes to raw or overwrites the V1 baseline.
- [x] V2 render requires EDL, sync report, and timeline plot.
- [x] V2 render has no path to complete Fast Montage or Full Highlights files.
- [x] Every expensive V2 stage has source/parameter/version cache identity.
- [x] The real-media task ends with A/B review, not full rendering.
