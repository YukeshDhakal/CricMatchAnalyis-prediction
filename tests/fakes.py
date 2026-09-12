from __future__ import annotations

import numpy as np

from video_engine.contracts import Detection, PoseFrame, Track
from video_engine.detection.base import Detector
from video_engine.pose.base import PoseEstimator


class FakeDetector(Detector):
    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, frames: list[np.ndarray]) -> list[Detection]:
        return self._detections


class FakePoseEstimator(PoseEstimator):
    def __init__(self, poses: list[PoseFrame]) -> None:
        self._poses = poses

    def estimate(self, frames: list[np.ndarray], tracks: list[Track]) -> list[PoseFrame]:
        return self._poses
