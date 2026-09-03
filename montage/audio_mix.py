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
