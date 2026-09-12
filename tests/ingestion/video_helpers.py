"""ffmpeg helpers shared by the video-adapter tests -- generates tiny synthetic clips
so tests exercise real ffprobe/ffmpeg calls without needing actual cricket footage.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def make_color_clip(path: Path, color: str, duration: float = 1.0, size: str = "64x64", fps: int = 10) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={size}:d={duration}:r={fps}",
            "-pix_fmt", "yuv420p", str(path),
        ],
        capture_output=True, check=True,
    )
    return path


def concat_clips(paths: list[Path], out_path: Path) -> Path:
    list_file = out_path.with_suffix(".txt")
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in paths), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_path)],
        capture_output=True, check=True,
    )
    return out_path
