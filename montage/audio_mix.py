"""FFmpeg audio automation graphs for game-first music mixing."""

from __future__ import annotations


def build_audio_filter(
    game_label: str,
    music_label: str,
    game_gain_db: float,
    music_gain_db: float,
) -> str:
    return (
        f"[{game_label}]volume={game_gain_db:.3f}dB[game_gain];"
        f"[{music_label}]volume={music_gain_db:.3f}dB[music_gain];"
        "[music_gain][game_gain]sidechaincompress="
        "threshold=0.04:ratio=2.4:attack=50:release=500:makeup=1[ducked_music];"
        "[game_gain][ducked_music]amix=inputs=2:duration=first:"
        "dropout_transition=0:normalize=0,aresample=async=1:first_pts=0,"
        "loudnorm=I=-14:TP=-1[aout]"
    )


def build_v2_audio_filter(
    game_label: str,
    music_label: str,
    game_gain_db: float,
    music_gain_db: float,
    *,
    duration: float,
    attack_ms: int = 50,
    release_ms: int = 500,
    target_lufs: float = -14.0,
    true_peak_db: float = -1.0,
) -> str:
    """Build a gameplay-first mix; the game bus is never ducked or gated."""
    if duration <= 0:
        raise ValueError("audio duration must be positive")
    return (
        f"[{game_label}]aresample=48000,volume={game_gain_db:.3f}dB[game_bus];"
        "[game_bus]asplit=2[game_sidechain][game_mix];"
        f"[{music_label}]aresample=48000,atrim=duration={duration:.3f},"
        f"volume={music_gain_db:.3f}dB[music_bus];"
        "[music_bus][game_sidechain]sidechaincompress="
        f"threshold=0.035:ratio=4.0:attack={int(attack_ms)}:release={int(release_ms)}:"
        "makeup=1:mix=1[ducked_music];"
        "[game_mix][ducked_music]amix=inputs=2:duration=longest:"
        "dropout_transition=0:normalize=0,"
        f"loudnorm=I={target_lufs:.1f}:TP={true_peak_db:.1f},"
        "aresample=48000:async=1:first_pts=0,atrim=duration="
        f"{duration:.3f},apad=whole_dur={duration:.3f}[aout]"
    )
