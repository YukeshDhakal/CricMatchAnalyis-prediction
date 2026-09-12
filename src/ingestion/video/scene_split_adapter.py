"""Heuristic v0 splitter for one continuous recording (a whole innings/over, not pre-cut).

Detects hard scene cuts with ffmpeg's `scene` filter and uses them as delivery
boundaries; broadcast footage typically cuts to a replay or graphic between
balls, so a scene cut is a reasonable proxy for "one delivery ended." If the
cut count doesn't match the known number of deliveries in that innings (an
uncut phone recording, a coverage style with no between-ball cuts, a missed or
spurious cut), this falls back to a uniform time split across the known
delivery count -- deliberately naive, same spirit as the video engine's own
"heuristic segmenter v0" for shot classification: a working default to replace
once real timing data (broadcast timecodes, on-pitch audio cues) is available.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..contracts import Delivery, DeliveryClip, DeliveryRef, Source
from .base import VideoAdapter
from .probe import probe

_SCENE_TIME_RE = re.compile(r"pts_time:(?P<t>[\d.]+)")


def detect_scene_cuts(video_path: str, threshold: float = 0.3) -> list[float]:
    """Returns timestamps (seconds) of detected hard cuts, via ffmpeg's scene-change filter."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-filter:v", f"select='gt(scene,{threshold})',showinfo",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    return [float(m.group("t")) for m in _SCENE_TIME_RE.finditer(result.stderr)]


def _uniform_boundaries(duration_s: float, count: int) -> list[float]:
    step = duration_s / count
    return [round(step * i, 3) for i in range(count + 1)]


def split_by_timestamps(video_path: str, boundaries: list[float], out_dir: Path | str, prefix: str) -> list[Path]:
    """Cuts `video_path` into len(boundaries)-1 segments at the given timestamps.

    Re-encodes rather than stream-copying: a `-c copy` cut can only land on a
    keyframe, which for arbitrary delivery-length boundaries (well under a
    typical GOP) would silently produce empty or misaligned segments.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        out_path = out_dir / f"{prefix}_{i + 1:04d}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(start), "-to", str(end),
                "-i", video_path, "-pix_fmt", "yuv420p", str(out_path),
            ],
            capture_output=True, check=True,
        )
        outputs.append(out_path)
    return outputs


class SceneSplitAdapter(VideoAdapter):
    def ingest(
        self,
        match_id: str,
        video_path: Path | str,
        innings_deliveries: list[Delivery],
        out_dir: Path | str,
        source: Source = Source.USER_UPLOAD,
        scene_threshold: float = 0.3,
    ) -> list[DeliveryClip]:
        """`innings_deliveries` must be one innings' Deliveries, in over/ball order -- this zips segments to them 1:1."""
        video_path = str(video_path)
        count = len(innings_deliveries)
        if count == 0:
            return []

        info = probe(video_path)
        cuts = detect_scene_cuts(video_path, threshold=scene_threshold)
        boundaries = [0.0, *cuts, info.duration_s]
        if len(boundaries) - 1 != count:
            boundaries = _uniform_boundaries(info.duration_s, count)

        prefix = f"{match_id}_inn{innings_deliveries[0].innings}"
        segment_paths = split_by_timestamps(video_path, boundaries, out_dir, prefix)

        clips: list[DeliveryClip] = []
        for delivery, segment_path in zip(innings_deliveries, segment_paths):
            seg_info = probe(str(segment_path))
            clips.append(
                DeliveryClip(
                    delivery=DeliveryRef(
                        match_id=delivery.match_id, innings=delivery.innings, over=delivery.over, ball=delivery.ball
                    ),
                    video_path=str(segment_path),
                    fps=seg_info.fps,
                    width=seg_info.width,
                    height=seg_info.height,
                    source=source,
                )
            )
        return clips
