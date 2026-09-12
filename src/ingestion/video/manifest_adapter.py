"""The real, working user-upload path: clips that are already split one-per-delivery.

This is the common case for nets and club footage (PRD 2.1, "user-uploaded
video") -- a coach or analyst records or exports one file per ball and pairs
it with a small manifest telling us which delivery each file is. No splitting
required, so there's no heuristic here: this just probes each file and attaches
the match/delivery key, matching DeliveryClip exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..contracts import DeliveryClip, DeliveryRef, Source
from .base import VideoAdapter
from .probe import probe


class ManifestClipAdapter(VideoAdapter):
    def ingest(
        self,
        match_id: str,
        clips_dir: Path | str,
        manifest_path: Path | str,
        source: Source = Source.USER_UPLOAD,
    ) -> list[DeliveryClip]:
        """`manifest_path` is a JSON file: {"<filename>": {"innings": int, "over": int, "ball": int}, ...}."""
        clips_dir = Path(clips_dir)
        with open(manifest_path, encoding="utf-8") as f:
            manifest: dict = json.load(f)

        clips: list[DeliveryClip] = []
        for filename, key in manifest.items():
            video_path = clips_dir / filename
            if not video_path.exists():
                raise FileNotFoundError(f"Manifest references missing clip: {video_path}")
            info = probe(str(video_path))
            ref = DeliveryRef(match_id=match_id, innings=key["innings"], over=key["over"], ball=key["ball"])
            clips.append(
                DeliveryClip(
                    delivery=ref,
                    video_path=str(video_path),
                    fps=info.fps,
                    width=info.width,
                    height=info.height,
                    source=source,
                )
            )
        return clips
