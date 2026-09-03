# Battlefield Montage V1.1 Editorial Optimization Design

**Date:** 2026-09-03  
**Status:** Approved design; implementation plan follows after user review  
**Milestone:** Generate and verify D:\91\集锦\output\preview_60s_v2.mp4, then stop for A/B review

## 1. Purpose

V1's technical chain passed, but its editorial result was only baseline quality: approximately 55 seconds were represented by three 17–20 second gameplay segments. The segments were coherent, but the cut points were weakly motivated, the ranking favored sustained combat intensity over payoff, and music synchronization mostly affected shot boundaries rather than the gameplay events themselves.

V1.1 keeps the V1 baseline intact and improves editorial decision quality with a deliberately small local algorithmic layer:

~~~
ROI event evidence
  -> temporal event fusion
  -> payoff anchors
  -> context-safe candidate condensation
  -> payoff-aware ranking
  -> lightweight beam-search timeline
  -> action-to-beat placement
  -> conservative gameplay-compatible audio/cuts
  -> preview_60s_v2.mp4
~~~

V1.1 does not use a Video LLM, cloud API, online service, or required OCR installation.

## 2. Inherited hard constraints

- D:\91\集锦\raw remains permanently read-only. No file below it may be deleted, moved, renamed, overwritten, metadata-modified, or encoded in place.
- Intermediates remain below D:\91\集锦\work; finished video remains below D:\91\集锦\output.
- The selected FFmpeg 8.0 executable and sibling ffprobe are discovered once, runtime-tested, recorded, and reused for the entire run. The PATH-first FFmpeg 4.3.1 must not be selected for NVENC work.
- The existing D:\91\集锦\output\preview_60s.mp4 is the immutable V1 baseline for this iteration. V1.1 writes preview_60s_v2.mp4 and never overwrites the baseline.
- The test music range defaults to the exact V1 EDL range 19.000s–74.252s. A change is allowed only within ±0.5 seconds when a clearly better boundary is demonstrated and both baseline and V2 ranges plus the reason are reported.
- V1.1 generates only the V2 preview and its analysis/debug artifacts. It does not render Battlefield_Fast_Montage.mp4 or Battlefield_Full_Highlights.mp4.
- Preview geometry remains 1920×1200, approximately source cadence, with no interpolation, no destructive 16:9 crop, and no upscale of a smaller source.
- Gameplay quality and continuity remain more important than exact beat precision, diversity, or transition effects.

## 3. Editorial acceptance criteria

The V2 result is accepted for human A/B review only when all of the following hold:

1. The baseline file still exists and its path, size, and mtime are unchanged by V1.1.
2. The V2 preview is 45–60 seconds, contains video and audio, probes successfully, and decodes fully with the pinned FFmpeg.
3. The EDL exists before the V2 render and every selected source is an existing file below raw with a valid source range.
4. The V2 timeline contains approximately 8–14 macro shots when the candidate pool supports it. A result with only three shots averaging about 18 seconds is rejected and the selector must retry with its next beam result or fail clearly.
5. Ordinary shots target 2–6 seconds, complete actions target 6–10 seconds, and a Hero Play is at most approximately 10–12 seconds unless a context-integrity exception is recorded.
6. At least one payoff anchor or explicitly reported weak/unknown event rationale exists for every selected macro shot. No event below weak_anchor_threshold may force shot movement.
7. Every semantic event label is backed by multiple evidence families. A single HUD ROI change cannot independently create a kill, multikill, vehicle_destroy, or objective event.
8. Event peaks within event_merge_window_ms are merged into one event with a peak timestamp, combined confidence, and preserved evidence.
9. A selected shot's context-integrity score is not sacrificed merely to reduce a beat offset. Sync targets are advisory; natural action wins over mathematical zero offset.
10. Full output commands remain guarded and no full montage file is created.

Automatic metrics are debug evidence, not a replacement for human judgment. Lower event-sync P95 alone does not establish that V2 is better.

## 4. Payoff event model

### 4.1 Evidence extraction

payoff_detection.py analyzes short clips directly at a low rate and analyzes long-recording candidate windows using the existing long-only proxy policy, refining promising windows against the original source when needed. All timestamps are source timestamps, not proxy-local timestamps.

The detector uses normalized HUD and action regions so the audited 1920×1200, 2560×1600, 2560×1440, and 1680×1050 sources can share one profile:

- reward/score UI region;
- kill-feed/change region;
- crosshair and center impact region;
- objective/status region;
- damage-state or red-vignette border region;
- full-frame motion/luminance/novelty regions;
- audio transient and low-frequency impact evidence.

The fixed ROIs are configuration data, not scattered constants. A ROI state change is only one evidence input. Temporal differencing, local luminance/color change, persistence across adjacent samples, action motion, audio transient timing, and candidate position are fused before an event is emitted. OpenCV may be used for image operations. OCR is optional; if it is unavailable, the run records ocr_available: false and continues with non-OCR evidence.

### 4.2 Event schema and semantic safety

Every emitted event has this shape:

~~~json
{
  "event_id": "evt-001",
  "type": "combat_climax",
  "source_time": 12.43,
  "confidence": 0.84,
  "strength": 0.88,
  "semantic_confidence": 0.41,
  "evidence": {
    "reward_roi_change": 0.81,
    "crosshair_event": 0.76,
    "audio_transient": 0.91,
    "motion_peak": 0.68,
    "killfeed_change": 0.44
  },
  "detector_flags": ["impact", "reward_candidate"],
  "merged_peak_count": 3
}
~~~

confidence describes confidence that a meaningful event occurred. semantic_confidence separately describes confidence in the event label. A high-energy moment may therefore be combat_climax with high event confidence but low kill semantic confidence.

Allowed event labels, from strongest semantics to safest fallback, are:

~~~
kill
multikill
vehicle_destroy
objective
combat_climax
impact_event
visual_transient
danger_climax
~~~

The semantic labels kill, multikill, vehicle_destroy, and objective require corroborating evidence from at least two independent families, with stronger labels requiring stronger combined evidence. A lone kill-feed change, reward ROI change, or compression artifact must be downgraded to a non-semantic event. The detector keeps all evidence in the artifact so false positives can be audited.

### 4.3 Refractory and merge behavior

Detector peaks inside event_merge_window_ms are fused. The default is 700ms, configurable. The merged event keeps the strongest timestamp, the maximum or fused value for each evidence channel, all contributing detector flags, and the number of merged peaks. A kill-feed update, reward popup, explosion, and audio transient from one action must not become four anchors.

The fusion confidence uses a bounded multi-evidence rule: independent evidence increases confidence, repeated evidence from the same family has diminishing returns, and the result is clamped to [0, 1]. It is not the maximum of one ROI score.

## 5. Candidate enrichment and intelligent condensing

### 5.1 Candidate fields

V1.1 preserves all V1 candidate fields and adds:

~~~
parent_candidate_id
payoff_events
primary_anchor
secondary_anchors
anchor_event_time
anchor_event_type
anchor_event_strength
anchor_event_confidence
context_integrity_score
action_density
danger_score
visual_novelty
stationary_ads_penalty
same_view_penalty
downtime_penalty
no_payoff_penalty
repetitive_fire_penalty
source_signature
environment_signature
weapon_or_view_signature
~~~

primary_anchor is a structured event reference; secondary_anchors contains at most a small number of meaningful events that can support internal action rhythm.

### 5.2 Anchor selection

Anchor selection does not simply choose the event with maximum confidence. Each event receives a utility based on:

- event strength;
- semantic importance, while keeping semantic confidence separate;
- position within the candidate, with preference for an action payoff rather than an arbitrary first transient;
- whether a nearby strong beat/downbeat/onset offers a natural placement;
- context-integrity cost of centering the shot on that event.

The last meaningful multikill, explosion payoff, vehicle destruction, or survival resolution may become a primary anchor even when an earlier isolated kill has a slightly higher raw confidence. Earlier kills or weapon impacts can become secondary anchors. A strong anchor may drive Action-to-Beat placement; a weak anchor can only guide it.

The configurable defaults are strong_anchor_threshold: 0.75 and weak_anchor_threshold: 0.55. They are calibration starting points, not claims of universal accuracy. The run reports the actual thresholds, event counts, semantic downgrades, and confidence distribution.

### 5.3 Context-safe condensing

An original saved clip remains a high-priority source, but it does not automatically remain a single 18-second macro shot. For a candidate longer than 12 seconds, the selector may create a compact representation around one or two payoff phases:

- retain enough setup to understand the encounter;
- retain the action leading to the anchor;
- retain the payoff and a short resolution tail;
- remove explicit waiting, stationary ADS with no escalation, empty travel, repeated aiming, and post-payoff drag.

A condensed representation is normally one continuous source range. If two source ranges from the same parent are used, it is limited to a small number of segments and must record condense_reason as one of:

~~~
downtime_removed
phrase_boundary
bar_boundary
action_phase_change
substantial_spatial_change
substantial_state_change
~~~

No jump cut is allowed without one of these reasons. The implementation must never start after the encounter cause, cut off necessary aiming, start from half an action, or end before the payoff. If the available evidence cannot prove safe condensation, the original continuous range wins.

## 6. Payoff-aware ranking and penalties

The V1.1 default ranking weights are configuration values and must appear in config.yaml:

~~~text
payoff_score             0.25
human_selection_prior    0.15
combat_intensity         0.12
action_density           0.12
continuity               0.10
visual_novelty           0.08
motion                   0.06
audio_activity           0.05
danger_score             0.04
uniqueness               0.03
~~~

The penalty features are not unconditional:

~~~
stationary ADS
+ low motion
+ low novelty
+ no payoff
+ no danger/action escalation
=> significant stationary_ads_penalty
~~~

Stationary ADS with repeated kills, an explosion, a survival payoff, or escalating danger reduces or cancels the penalty. Similar logic applies to downtime and repetitive fire. Penalties remain diagnostics in candidate JSON/CSV so a reviewer can see why a sustained-combat candidate fell.

## 7. Music analysis and Action-to-Beat

V1.1 runs HPSS and uses the percussive component as an additional beat/onset hypothesis. It keeps separate records for:

~~~
beat
strong_beat
downbeat/bar attempt
phrase
section
onset
RMS/energy
confidence
~~~

The full and percussive hypotheses are compared. If downbeat or phrase confidence is weak, the report says so rather than fabricating exact structure.

The V2 music interval is loaded from the V1 baseline EDL and remains 19.000–74.252 by default. A ±0.5 second adjustment is allowed only when a section/phrase boundary is materially better and the sync report contains:

~~~
baseline_music_in
baseline_music_out
v2_music_in
v2_music_out
change_reason
~~~

Action-to-Beat placement maps the primary anchor event to a strong beat or downbeat where possible, maps secondary anchors to ordinary beats/onsets, and uses phrase/bar positions for visual shot boundaries. It does not force every shot boundary onto a beat.

For each selected shot:

~~~
event_timeline = timeline_in + (anchor_source_time - source_in)
event_sync_offset = event_timeline - selected_music_event
cut_sync_offset = timeline_boundary - selected_boundary_event
~~~

The sync target is optimized through timeline placement, context-safe trim, and reasonable pre-roll/post-roll. A visible context break is always more expensive than a larger offset. Median absolute event offset of 80ms and P95 of 160ms are soft targets only; ordinary events may be wider.

## 8. Lightweight beam-search timeline

The timeline no longer greedily sorts candidates by final score. It maintains a bounded beam of partial timelines. beam_width is configurable and defaults to a value between 8 and 32; the initial implementation uses 16. Each candidate expansion considers only a small set of music-compatible placements: a nearby phrase/bar boundary, strong beat/downbeat, or onset fallback.

Each beam state tracks:

~~~
selected shots
elapsed duration
used duplicate groups
recent sources and environment signatures
energy progression
last transition descriptor
context-integrity total
anchor/cut sync diagnostics
~~~

The placement score is:

~~~text
highlight_quality          * 0.35
music_fit                  * 0.25
transition_compatibility   * 0.15
diversity                  * 0.15
energy_curve_fit           * 0.10
~~~

Penalties for repeated recent source, repeated recent environment, duplicate group, context damage, and candidate downtime are applied after the base score. Quality remains dominant: a genuine continuous Hero Play is not rejected solely because it shares a source or environment with its immediately preceding phase.

The target timeline curve is piecewise and observable, not a simple ascending sort:

~~~text
0–20%   build / establish rhythm
20–45%  first payoff and rise
45–60%  brief release
60–85%  escalation
85–100% strongest payoff / Hero Play
~~~

The final approximately 15% receives a positive preference for strong primary anchors and high-quality continuous action. The beam must prefer a slightly weaker early shot to preserve a stronger climax for the ending when the quality difference is not decisive.

## 9. Transition compatibility and audio continuity

transition_compatibility_score is a selection feature, not a request for an effect. It compares:

- ending and starting motion direction/strength;
- luminance and dominant visual tone;
- ADS and weapon-motion state;
- muzzle flash/explosion/impact evidence;
- source/environment signature.

Hard Cut remains the normal rendered video transition. impact_cut is metadata indicating that a natural gameplay flash, muzzle flash, or explosion hides the cut; no artificial white frame, flash, RGB split, zoom, shake, warp, or motion blur is added. Motion compatibility similarly selects an ordinary Hard Cut.

The V2 audio renderer uses a continuous music source for the selected music interval and keeps the game track separate until the final mix. At a safe visual Hard Cut, it may use one conservative audio-only overlap:

- next gameplay audio J-cut: 100–250ms before the visual cut; or
- previous impact/gameplay L-cut: 150–400ms after the visual cut.

The choice is recorded per edge. Different-scene audio is never allowed to overlap for more than one local transition window, and unsafe/noisy edges fall back to a direct audio boundary. Video HUD frames never crossfade. Music ducking remains smooth and game-first, with additional reduction around high-energy game transients.

## 10. V2 data contracts and artifacts

V1.1 writes versioned artifacts without overwriting V1 artifacts:

~~~
work/analysis/music_v2/beat_map_v2.json
work/analysis/music_v2/music_structure_v2.json
work/analysis/music_v2/music_analysis_v2.png
work/analysis/payoff_events_v2.json
work/analysis/highlight_candidates_v2.json
work/analysis/highlight_candidates_v2.csv
work/analysis/dedupe_summary_v2.json
work/analysis/preview/preview_v2_edit.json
work/analysis/preview/preview_v2_timeline.txt
work/analysis/preview/preview_v2_timeline.png
work/analysis/preview/preview_v2_sync_report.json
work/analysis/preview/preview_v2_report.md
output/preview_60s_v2.mp4
~~~

The V2 EDL is a superset of the V1 shot contract. Every macro shot records the original source fields plus the following fields. A normal continuous shot has one `source_segments` entry; an internally condensed shot may have at most two entries from the same parent candidate. The legacy `source`, `source_in`, and `source_out` fields describe the first segment for compatibility, while `source_segments` is authoritative for rendering:

~~~json
{
  "anchor_event_time": 25.4,
  "anchor_event_type": "combat_climax",
  "anchor_event_strength": 0.88,
  "anchor_event_confidence": 0.84,
  "primary_anchor": {"type": "combat_climax", "source_time": 25.4},
  "secondary_anchors": [],
  "source_segments": [
    {"source": "D:\\91\\集锦\\raw\\clip.mp4", "source_in": 18.2, "source_out": 22.4, "duration": 4.2},
    {"source": "D:\\91\\集锦\\raw\\clip.mp4", "source_in": 24.0, "source_out": 27.0, "duration": 3.0}
  ],
  "context_integrity_score": 0.93,
  "condense_reason": "downtime_removed",
  "event_timeline": 18.22,
  "event_sync_offset": -0.041,
  "cut_sync_offset": 0.032,
  "transition_compatibility_score": 0.81,
  "impact_cut": false,
  "audio_j_cut_ms": 120,
  "audio_l_cut_ms": 0
}
~~~

preview_v2_timeline.txt remains human-readable. preview_v2_timeline.png shows the music waveform/energy, strong beat and downbeat markers, video shot spans, primary and secondary anchors, and cut positions on one time axis.

preview_v2_sync_report.json records at minimum:

~~~
baseline_music_range
v2_music_range
macro_shot_count
average_shot_duration
median_shot_duration
hero_shot_count
payoff_anchor_count
strong_anchor_count
cut_sync_mean/median/P90/P95
event_sync_mean/median/P90/P95
transition_compatibility_mean
stationary_ads_duration_ratio
estimated_downtime_ratio
same_environment_consecutive_count
same_source_recent_penalty_count
rejected_by_stationary_ads
rejected_by_downtime
rejected_by_no_payoff
~~~

The V2 report directly compares these metrics with the V1 baseline and labels every metric as diagnostic rather than a quality guarantee.

## 11. Caching and failure handling

Expensive V1.1 stages use cache keys containing the source absolute path, file size, mtime, stage/version, selected FFmpeg version, analysis rate, ROI profile version, and relevant configuration:

- payoff event extraction;
- refined video/audio features;
- HPSS music analysis;
- fingerprints and event fingerprints;
- candidate condensation variants.

If the source fingerprint is unchanged, these stages are reused. A changed ROI profile, detector version, or music analysis version invalidates only its affected artifacts. Corrupt or incomplete cache entries are rebuilt atomically under work.

No event detector failure may modify raw. If a source cannot produce reliable event evidence, its candidate remains eligible with downgraded anchor confidence and an explicit rationale. If there are not enough context-safe candidates to reach 45 seconds, the run fails with a diagnostic rather than fabricating cuts.

## 12. CLI and stop rule

The V1.1 CLI exposes explicit versioned commands:

~~~text
python main.py all-v2 --dry-run
python main.py render-preview-v2
python main.py verify-preview-v2
~~~

all-v2 --dry-run must finish payoff analysis, V2 ranking, dedupe, beam-search EDL, sync report, and timeline visualization before any V2 render. render-preview-v2 is the only render command used in this milestone. render-fast and render-full remain blocked.

The run stops after:

~~~
output/preview_60s_v2.mp4
~~~

has passed probe, full decode, RAW/baseline integrity, EDL range, duration, geometry, audio, cache, and report checks. The next action is human A/B review of preview_60s.mp4 versus preview_60s_v2.mp4.

## 13. Testing strategy

Tests are written before implementation changes. Unit and fixture tests cover:

- multi-evidence event fusion and semantic downgrade;
- 700ms event refractory/merge behavior;
- strong/weak anchor thresholds;
- primary/secondary anchor choice;
- context-integrity protection during Action-to-Beat placement;
- intelligent condensing and mandatory jump-cut reasons;
- payoff-aware stationary ADS penalty;
- source/environment repetition penalties;
- bounded beam width and no-three-long-segments guard;
- HPSS confidence and unchanged baseline music range;
- separate cut/event sync statistics including mean, median, P90, and P95;
- transition compatibility as selection-only metadata;
- conservative J/L audio graph without video crossfade;
- V2 output paths and baseline non-overwrite;
- RAW read-only and FFmpeg 8 pinning invariants.

The real run must additionally verify that the V1 baseline and RAW manifest are unchanged, the V2 EDL precedes rendering, the selected FFmpeg/ffprobe paths are consistent, and no complete montage outputs exist.
