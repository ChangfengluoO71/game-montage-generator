from pathlib import Path
from subprocess import CompletedProcess

from montage.proxy import build_proxy
from montage.video_analysis import merge_activity_peaks, normalize_signal


def test_short_clip_does_not_get_proxy(short_record, base_config, fake_toolchain):
    assert build_proxy(short_record, base_config, fake_toolchain) is None


def test_long_proxy_is_written_under_work_not_raw(long_record, base_config, fake_toolchain, monkeypatch):
    from dataclasses import replace

    config = replace(
        base_config,
        raw_dir=long_record.file_path.parent,
        work_dir=long_record.file_path.parent.parent / "work",
        output_dir=long_record.file_path.parent.parent / "output",
    )

    def fake_successful_ffmpeg(argv, **kwargs):
        Path(argv[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[-1]).touch()
        return CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("montage.proxy.run_command", fake_successful_ffmpeg)

    proxy = build_proxy(long_record, config, fake_toolchain)

    assert proxy.parent == config.proxy_dir
    assert not str(proxy).startswith(str(config.raw_dir))


def test_normalize_signal_is_bounded_and_constant_safe():
    assert normalize_signal([2, 4, 6]) == [0.0, 0.5, 1.0]
    assert normalize_signal([3, 3]) == [0.0, 0.0]


def test_nearby_activity_peaks_form_one_combat_segment():
    times = [float(index) for index in range(20)]
    activity = [0.0] * 20
    for index in (5, 6, 8, 9, 10):
        activity[index] = 1.0

    assert merge_activity_peaks(activity, times, 0.8, 3.0, 3.0, 30.0) == [(5.0, 12.0)]
