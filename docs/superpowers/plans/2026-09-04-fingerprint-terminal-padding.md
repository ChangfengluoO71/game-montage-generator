# Fingerprint Terminal Padding Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task.

**Goal:** Make V2 fingerprint sampling deterministic for short and fractional-duration source segments without reading outside the requested range.

**Architecture:** Keep end-exclusive `ceil(duration / interval)` as the single expected-count contract. Trim and reset timestamps in FFmpeg's filter graph, sample with `fps`, clone only the terminal in-segment frame through `tpad`, and retain a bounded `-frames:v` raw output plus strict subprocess/output validation.

**Tech Stack:** Python, pytest, selected FFmpeg subprocess, NumPy raw-frame hashing.

**Spec:** `docs/superpowers/specs/2026-09-03-battlefield-montage-v1-1-editorial-optimization-design.md`

## Global Constraints

- Do not modify files under `raw` or render media.
- Preserve cache validation against the same end-exclusive expected count.
- Reject nonzero FFmpeg exit status and truncated raw output without caching it.

---

### Task 1: Reproduce the terminal-frame sampling failure

**Files:**
- Modify: `tests/test_v2_dedupe.py`
- Test: `tests/test_v2_dedupe.py`

- [ ] **Step 1: Write the failing regression test**

Create a 10.242-second filesystem-backed variant, capture the FFmpeg argv, return six synthetic raw frames, and assert the filter graph and bounded count contract:

```python
def test_fingerprint_trims_and_pads_terminal_frame_for_fractional_duration(
    tmp_path, base_config, fake_toolchain, monkeypatch
):
    item = _filesystem_variant(tmp_path)
    item = replace(
        item,
        duration=10.242,
        source_segments=(SourceSegment(item.source_file, 0.0, 10.242, 10.242),),
    )
    config = replace(base_config, raw_dir=tmp_path / "raw", work_dir=tmp_path / "work")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return type("Result", (), {"returncode": 0, "stdout": bytes(6 * 32 * 32), "stderr": b""})()

    monkeypatch.setattr("montage.dedupe.subprocess.run", fake_run)

    assert len(fingerprint_variant(item, item.source_file, fake_toolchain, config)) == 6
    command = calls[0]
    assert "-t" not in command
    assert "-frames:v" in command and command[command.index("-frames:v") + 1] == "6"
    filter_graph = command[command.index("-vf") + 1]
    assert "trim=duration=10.242" in filter_graph
    assert "setpts=PTS-STARTPTS" in filter_graph
    assert "fps=0.5" in filter_graph
    assert "tpad=stop_mode=clone:stop_duration=2" in filter_graph
```

- [ ] **Step 2: Run the focused test and verify it fails against current production code**

Run: `D:/miniconda/python.exe -m pytest -q tests/test_v2_dedupe.py::test_fingerprint_trims_and_pads_terminal_frame_for_fractional_duration`

Expected: FAIL because the current argv contains output-level `-t` and no filter-level trim/tpad graph.

### Task 2: Implement the minimal filter-graph fix

**Files:**
- Modify: `montage/dedupe.py`
- Test: `tests/test_v2_dedupe.py`

- [ ] **Step 1: Replace output-level duration limiting**

Build the graph as `trim=duration=<duration>,setpts=PTS-STARTPTS,fps=<rate>,tpad=stop_mode=clone:stop_duration=<interval>,scale=32:32:flags=fast_bilinear,format=gray`; remove `-t`; retain `-frames:v <sample_count>`.

- [ ] **Step 2: Preserve strict failure and cache behavior**

Keep nonzero return-code rejection and exact raw byte-count validation. Keep `_fingerprint_sample_count` as the source for segment counts, aggregate expected counts, cache validation, and bounded output.

- [ ] **Step 3: Run focused tests**

Run: `D:/miniconda/python.exe -m pytest -q tests/test_v2_dedupe.py tests/test_candidates_dedupe.py`

Expected: PASS.

### Task 3: Verify, report, and commit

**Files:**
- Modify: `D:/91/集锦/work/analysis/preview_report.md`

- [ ] **Step 1: Run the requested verification commands**

Run focused dedupe/fingerprint tests, relevant V2 tests, full pytest, `compileall`, and `git diff --check`; do not render media.

- [ ] **Step 2: Update the progress/report**

Record the real-run root cause (10.242-second container versus five decodable frames at 0.5 fps), the filter trim/setpts/fps/tpad fix, strict failure handling, and the verification results.

- [ ] **Step 3: Commit the fix separately**

Run `git add montage/dedupe.py tests/test_v2_dedupe.py docs/superpowers/plans/2026-09-04-fingerprint-terminal-padding.md` and commit with `fix: pad terminal fingerprint samples within trimmed segments`.
