"""Reads the video file ingestion hands off into frames the rest of the engine works on."""
from __future__ import annotations

import cv2
import numpy as np

from ..contracts import DeliveryClip


def load_frames(clip: DeliveryClip) -> list[np.ndarray]:
    capture = cv2.VideoCapture(clip.video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open delivery clip: {clip.video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames
