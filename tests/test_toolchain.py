from pathlib import Path
from subprocess import CompletedProcess

from montage.config import load_config
from montage.toolchain import Toolchain, _parse_version, discover_toolchain, run_command


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_toolchain_prefers_highest_viable_version(monkeypatch):
    config = load_config(PROJECT_ROOT / "config.yaml")
    candidates = [Path(r"D:\old\ffmpeg.exe"), Path(r"D:\new\ffmpeg.exe")]

    monkeypatch.setattr("montage.toolchain.find_ffmpeg_candidates", lambda _: candidates)
    monkeypatch.setattr(
        "montage.toolchain.probe_candidate",
        lambda path, *_: Toolchain(
            ffmpeg=path,
            ffprobe=path.with_name("ffprobe.exe"),
            ffmpeg_version="8.0" if "new" in str(path) else "4.3.1",
            ffprobe_version="8.0" if "new" in str(path) else "4.3.1",
            nvenc_h264="new" in str(path),
            nvenc_hevc="new" in str(path),
            selected_reason="test",
            candidates=[],
        ),
    )

    selected = discover_toolchain(config)

    assert selected.ffmpeg == candidates[1]
    assert selected.ffprobe == Path(r"D:\new\ffprobe.exe")


def test_command_keeps_unicode_path_as_one_argument(monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("montage.toolchain.subprocess.run", fake_run)
    run_command(["ffprobe", "-show_format", r"D:\raw\中文 [a].mp4"])

    assert seen["argv"][-1] == r"D:\raw\中文 [a].mp4"
    assert seen["kwargs"]["shell"] is False


def test_version_parser_accepts_ffprobe_output():
    assert _parse_version("ffprobe version 8.0-full_build-www.gyan.dev") == "8.0"
