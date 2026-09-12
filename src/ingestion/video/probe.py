"""ffprobe wrapper -- reads the fps/width/height DeliveryClip needs, without pulling in a CV dependency."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    duration_s: float


def probe(video_path: str) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-show_entries", "format=duration",
            "-of", "json", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    parsed = json.loads(result.stdout)
    stream = parsed["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else float(num)
    duration = stream.get("duration") or parsed.get("format", {}).get("duration")
    return VideoInfo(fps=fps, width=int(stream["width"]), height=int(stream["height"]), duration_s=float(duration))
